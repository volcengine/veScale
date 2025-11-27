################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from collections import defaultdict

from torch._guards import detect_fake_mode
from torch.distributed.tensor.debug import CommDebugMode as TorchCommDebugMode
from torch.distributed.tensor.debug._comm_mode import c10d_collective_ops, NATIVE_TO_PY_MAPPING
from torch.distributed.tensor import DTensor as TorchDTensor

from vescale.dtensor import DTensor


class CommDebugMode(TorchCommDebugMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into comms ops, before CommDebugMode sees them.

        # sets up operation-level collective count
        if self.advanced_module_tracker.name not in self.comm_module_operation_counts:
            # dictionary should hold module input and output shape, operations list and collective counter
            self.comm_module_operation_counts[self.advanced_module_tracker.name] = {"operations_list": []}
        operation_dict = {}
        operation_dict["name"] = func

        operation_dict["input_shape"] = []
        operation_dict["input_sharding"] = []
        operation_dict["device_mesh"] = ""

        # tracks if the operation is part of the backward pass
        operation_dict["is_bw"] = self.advanced_module_tracker.is_bw

        # tracks if the operation is part of activation checkpointing
        operation_dict["is_activation_checkpointing"] = self.advanced_module_tracker.activation_checkpointing

        if any(t in (DTensor, TorchDTensor) for t in types):
            for ele in args:
                if isinstance(ele, DTensor):
                    # saves shapes and placements of all DTensor args
                    operation_dict["input_shape"].append(ele.shape)
                    operation_dict["input_sharding"].append(ele.placements)
                    operation_dict["device_mesh"] = str(ele.device_mesh)

            self.comm_module_operation_counts[self.advanced_module_tracker.name]["operations_list"].append(
                operation_dict
            )

            return NotImplemented

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket

        # We have many tests that use CommDebugMode to verify the occurrence of
        # collectives. These tests do so by querying comm_counts with legacy
        # funcol ops as key. For the purpose of native funcol migration, we
        # need these tests to work for both legacy and native funcol. To avoid
        # the need to modify all tests to accommodate the two implementations,
        # we make CommDebugMode translate native funcol ops into legacy funcol
        # ops until the migration finishes.

        if func_packet in self.comm_registry or func_packet in c10d_collective_ops:
            if func_packet in NATIVE_TO_PY_MAPPING:
                func_packet = NATIVE_TO_PY_MAPPING[func_packet]
            self.comm_counts[func_packet] += 1

            key = "forward"
            if self.advanced_module_tracker.is_bw:
                key = "backward"

            # adds collective count to current module
            if self.advanced_module_tracker.name not in self.comm_module_counts:
                self.comm_module_counts[self.advanced_module_tracker.name] = {}
                self.comm_module_counts[self.advanced_module_tracker.name]["forward"] = defaultdict(int)
                self.comm_module_counts[self.advanced_module_tracker.name]["backward"] = defaultdict(int)
            self.comm_module_counts[self.advanced_module_tracker.name][key][func_packet] += 1

            # adds collective count to parent modules
            for par in self.advanced_module_tracker.module_parents_dict[self.advanced_module_tracker.name]:
                # makes sure we aren't double counting when current sub-module hasn't been removed from parents
                if par != self.advanced_module_tracker.name:
                    if par not in self.comm_module_counts:
                        self.comm_module_counts[par] = {}
                        self.comm_module_counts[par]["forward"] = defaultdict(int)
                        self.comm_module_counts[par]["backward"] = defaultdict(int)
                    self.comm_module_counts[par][key][func_packet] += 1

        # if tensor op uses fake tensors, return
        if detect_fake_mode(args):
            return out

        # add tensor operation to module operation list
        self.comm_module_operation_counts[self.advanced_module_tracker.name]["operations_list"].append(operation_dict)

        return out
