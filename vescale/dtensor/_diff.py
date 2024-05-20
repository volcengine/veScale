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

import functools
import os
from typing import Callable
import logging



VESCALE_DISABLE_REDISTRIBUTE = os.environ.get("VESCALE_DISABLE_REDISTRIBUTE", "1") == "1"

global VESCALE_SHARDING_SUGGETSION
VESCALE_SHARDING_SUGGETSION = []


def dummy_p2p(func: Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        global VESCALE_DUMMY_P2P
        if VESCALE_DUMMY_P2P:
            msg = f"{get_rank()}: {args}"
            logging.info(msg)
        else:
            if VESCALE_DUMP_INSTRUCTION:
                if vescale_file_to_dump is not None:
                    vescale_file_to_dump.write(f"=========================\nrank:{get_rank()}: {args}, {kwargs}\n")
                    vescale_file_to_dump.flush()
                    msg = f"rank:{get_rank()}: {args}, {kwargs}\n=======================\n"
                    logging.info(msg)
            out = func(*args, **kwargs)
            if VESCALE_DUMP_INSTRUCTION:
                if vescale_file_to_dump is not None:
                    vescale_file_to_dump.write(f"output: {out}\n")
                    vescale_file_to_dump.flush()
                    msg = f"output: {out}\n"
                    logging.info(msg)
            return out

    return wrap


def manage_dump_file(func: Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        if VESCALE_DUMP_INSTRUCTION:
            with open(f"instruction-{get_rank()}.txt", "w+") as file:
                global vescale_file_to_dump
                vescale_file_to_dump = file
                out = func(*args, **kwargs)
        else:
            out = func(*args, **kwargs)

        return out

    return wrap


class DeferReshardMode:
    """
    To enable the DTensor to be PartialSum for performance
    By sometimes, we find there have some optimization chance
    for partial state, so we enable to get a partial DTensor
    by torch ops

    chance one: adjust the reshard AllReduceReassociate
    The AllReduceReassociate can be simplify
    allreduce(x) + allreduce(y) to allreduce(x + y),
    partial add option can be done use this mode
    For Example:

        Partial(x) + Partial(y) -> Shard(out)

        one common operation is that:

        Partial(x) -> Replicate(x)
        Partial(y) -> Replicate(y)
        Replicate(x) + Replicate(y) -> Replicate(out)

    there have 2 allreduce operation while with DeferReshardMode operation can be
        Partial(x) + Partial(y) -> Partial(out)
        Partial(out) -> Replicate(out)

    Usage:

        fwd_shard_plan : {
            "partial_op.output.lazy": [[Replicate()]]
        }

    """

    @staticmethod
    def _push_sharding(placement):
        global VESCALE_SHARDING_SUGGETSION
        VESCALE_SHARDING_SUGGETSION.append(placement)

    @staticmethod
    def _remove_sharding():
        global VESCALE_SHARDING_SUGGETSION
        if len(VESCALE_SHARDING_SUGGETSION) > 0:
            VESCALE_SHARDING_SUGGETSION.pop(0)

    @staticmethod
    def _enable_autoresharding():
        global VESCALE_SHARDING_SUGGETSION
        return len(VESCALE_SHARDING_SUGGETSION) > 0

    @staticmethod
    def _query_sharding():
        global VESCALE_SHARDING_SUGGETSION
        return VESCALE_SHARDING_SUGGETSION[0]
