################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import random
import math
import itertools
import unittest
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

from vescale.dtensor.vescale_utils.checkpoint import _break_ragged_box


def product_limited(lst, repeat, max_count):
    def backtrack(current):
        if len(current) == repeat:
            yield tuple(current)
            return
        for x in lst:
            if current.count(x) < max_count:
                yield from backtrack(current + [x])

    yield from backtrack([])


def produce_test_tensor_shape(nd=(1, 2, 3, 4, 5), threshold=128 * 128):
    assert isinstance(nd, tuple)
    shape_1d = (3, 16, 32, 53)
    shape_2d = itertools.product(shape_1d, repeat=2)
    shape_3d = itertools.product(shape_1d, repeat=3)
    shape_4d = product_limited(shape_1d, repeat=4, max_count=3)
    shape_5d = product_limited(shape_1d, repeat=5, max_count=3)

    all_shape = []
    for i, s in enumerate((((x,) for x in shape_1d), shape_2d, shape_3d, shape_4d, shape_5d)):
        if i + 1 in nd:
            all_shape.append(s)
    iter_chain = itertools.chain(*all_shape)
    return filter(lambda x: math.prod(x) < threshold, iter_chain)


def contiguous_sequences(n: int):
    return (tuple(range(i, j)) for i in range(n) for j in range(i + 1, n + 1))


def get_ragged_shape(shape, ragged_dims):
    ragged_shape = []
    ragged_length = 1
    for i, x in enumerate(shape):
        if i == ragged_dims[0]:
            ragged_length *= x
            ragged_shape.append(None)
        elif i in ragged_dims:
            ragged_length *= x
            continue
        else:
            ragged_shape.append(x)
    ragged_shape[ragged_dims[0]] = ragged_length
    return ragged_shape


def all_hyperrectangles(shape):
    for starts in itertools.product(*(range(n) for n in shape)):
        for ends in itertools.product(*(range(s, n + 1) for s, n in zip(starts, shape))):
            yield tuple(zip(starts, ends))


def unique_random_hyperrectangles(shape, k=20000, min_size=1):
    rects = set()
    faild = 0
    while len(rects) < k:
        rect = []
        for s in shape:
            start = random.randint(0, s - min_size)
            end = random.randint(start, s)
            rect.append((start, end))

        rect = tuple(rect)

        l_rect = len(rects)
        rects.add(rect)
        if l_rect == len(rects):
            faild += 1
        if faild >= k * 0.1:
            break
    return list(rects)


class TestBreakRaggedBox(unittest.TestCase):
    @staticmethod
    def fill_tensor(tensor, sizes_list, offsets_list):
        counter = 0
        for sizes, offsets in zip(sizes_list, offsets_list):
            slices = tuple(slice(o, o + s) for s, o in zip(sizes, offsets))
            rect_t = tensor[slices]
            end = counter + rect_t.numel()
            torch.arange(counter, end, out=rect_t)
            counter = end

    @staticmethod
    def _data_generator(nd, hyperrect_f, num_in_batch=50):
        random.seed(9999)

        def generator():
            for shape in produce_test_tensor_shape(nd=nd):  # brute force test
                for ragged_dims in contiguous_sequences(len(shape)):
                    ragged_shape = get_ragged_shape(shape, ragged_dims)
                    for rect in hyperrect_f(ragged_shape):
                        yield (shape, ragged_dims, ragged_shape, rect)

        batch = []
        for data in generator():
            batch.append(data)
            if len(batch) == num_in_batch:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    @staticmethod
    def _parallel_run_test(works):
        max_number_of_broken_box = 0
        for work in works:
            shape, ragged_dims, ragged_shape, rect = work
            tensor = torch.full(shape, -1, device="cpu", dtype=torch.int64)
            ragged_shape = get_ragged_shape(tensor.shape, ragged_dims)
            ragged_tensor = tensor.view(ragged_shape)
            rect_slice = tuple(slice(s, e) for s, e in rect)
            box_shape = tuple(e - s for s, e in rect)
            box_offset = tuple(s for s, _ in rect)
            sizes_list, offsets_list = _break_ragged_box(
                box_shape, box_offset, ragged_dims, shape, shape, torch.Size([0 for _ in range(len(shape))])
            )
            max_number_of_broken_box = max(max_number_of_broken_box, len(sizes_list))
            TestBreakRaggedBox.fill_tensor(tensor, sizes_list, offsets_list)
            t = ragged_tensor[rect_slice]

            # this checks if there is any overlapping in the produced sizes_list & offsets_list
            golden = set(range(0, t.numel()))
            t_set = set(t.flatten().tolist())
            if not t_set == golden:
                raise AssertionError(
                    f"not equal \n{golden=} \n{t=} \n{ragged_tensor=} \n{ragged_dims=} \n{shape=} \n{ragged_shape=} \n{sizes_list=} \n{offsets_list=} \n{rect=} \n{box_shape=} \n{box_offset=}"
                )

            # this checks if sizes_list & offsets_list exactly match the given rect.
            t.fill_(-1)
            if not torch.all(tensor == -1):
                raise AssertionError(
                    f"not all -1 \n{t=} \n{ragged_tensor=} \n{ragged_dims=} \n{shape=} \n{ragged_shape=} \n{sizes_list=} \n{offsets_list=} \n{rect=} \n{box_shape=} \n{box_offset=}"
                )
        return max_number_of_broken_box

    def _run_nd_test_parallel(self, *, nd, hyperrect_f=all_hyperrectangles, skip_check_max_number_of_broken_box=False):
        data = list(self._data_generator(nd, hyperrect_f))
        print(f"create data {len(data)=}")
        n_workers = max(4, min(math.floor(0.75 * cpu_count()), len(data) // 2))
        ok = []
        ex = None
        futs = []
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                print(f"start parallel work with {n_workers=}")
                futs = [ex.submit(self._parallel_run_test, x) for x in data]
                print(f"produce {len(futs)} jobs")
                for fut in as_completed(futs):
                    ok.append(fut.result())
                    if len(ok) % math.floor(len(data) * 0.1) == 0:
                        print(f"finished {len(ok)=}")
        except Exception as e:
            assert ex is not None
            ex.shutdown(cancel_futures=True)
            for f in futs:
                f.cancel()
            raise RuntimeError(str(e)) from e
        else:
            max_number_of_broken_box = max(ok)
            if skip_check_max_number_of_broken_box:
                print(f"{max_number_of_broken_box=} {nd=}")
            else:
                self.assertEqual(max_number_of_broken_box, 2 ** (max(nd)) - 1)

    def _run_nd_test(self, *, nd):
        # single process debug code
        max_number_of_broken_box = 0
        for shape in produce_test_tensor_shape(nd=nd):
            for ragged_dims in contiguous_sequences(len(shape)):
                tensor = torch.full(shape, -1, device="cpu", dtype=torch.int64)
                ragged_shape = get_ragged_shape(tensor.shape, ragged_dims)
                ragged_tensor = tensor.view(ragged_shape)
                print(f"testing {shape=} {ragged_dims=}")
                for rect in all_hyperrectangles(ragged_tensor.shape):
                    rect_slice = tuple(slice(s, e) for s, e in rect)
                    box_shape = tuple(e - s for s, e in rect)
                    box_offset = tuple(s for s, _ in rect)
                    sizes_list, offsets_list = _break_ragged_box(
                        box_shape, box_offset, ragged_dims, shape, shape, torch.Size([0 for _ in range(len(shape))])
                    )
                    max_number_of_broken_box = max(max_number_of_broken_box, len(sizes_list))
                    self.fill_tensor(tensor, sizes_list, offsets_list)

                    t = ragged_tensor[rect_slice]
                    # msg = f"\n{t=} \n{ragged_tensor=} \n{sizes_list=} \n{offsets_list=} \n{rect=} \n{box_shape=} \n{box_offset=}"
                    self.assertTrue(torch.equal(t, torch.arange(0, t.numel()).view(t.shape)))
                    t.fill_(-1)
                    self.assertTrue(torch.all(tensor == -1))
        self.assertEqual(max_number_of_broken_box, 2 ** (max(nd)) - 1)

    def test_1d_case(self):  # brute force test
        self._run_nd_test_parallel(nd=(1,))

    def test_2d_case(self):  # brute force test
        self._run_nd_test_parallel(nd=(2,))

    def test_3d_case(self):  # random test
        self._run_nd_test_parallel(
            nd=(3,), hyperrect_f=unique_random_hyperrectangles, skip_check_max_number_of_broken_box=True
        )

    def test_4d_case(self):  # random test
        self._run_nd_test_parallel(
            nd=(4,), hyperrect_f=unique_random_hyperrectangles, skip_check_max_number_of_broken_box=True
        )


if __name__ == "__main__":
    unittest.main()
