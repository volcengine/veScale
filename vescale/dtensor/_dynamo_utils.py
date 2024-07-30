################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from torch._dynamo import allow_in_graph

from vescale.dtensor.api import from_local
from vescale.dtensor.dtensor import DTensor

# dynamo/torch.compile utils for
allow_in_graph(DTensor)
allow_in_graph(from_local)
