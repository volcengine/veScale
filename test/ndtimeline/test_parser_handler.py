################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

import pytest
from vescale.ndtimeline.world_info import WorldInfo
from vescale.ndtimeline.handlers import ParserNDHandler
from vescale.ndtimeline.exceptions import NDHandlerError


def test_normal_input_with_tags():
    metric_name = "test_metric"
    recent_elapsed_raw_parts = [1.0, 3.2, 1.4]
    elapsed = sum(recent_elapsed_raw_parts)
    recent_since_start_raw_parts = [1710332816.6118143, 1710332833.2222, 1710332846.1313]
    single_tag = {"is_test": True}
    tags = [single_tag] * (len(recent_elapsed_raw_parts) - 1) + [{"is_test": False}]
    step_range = range(0, 1)
    world_info = WorldInfo(0, 0)
    callback = ParserNDHandler()
    records = callback(
        metric_name, elapsed, recent_elapsed_raw_parts, recent_since_start_raw_parts, tags, step_range, world_info, {}
    )
    assert len(records) == 1
    assert records[0].step == 0


def test_normal_invalid_input():
    metric_name = "test_metric"
    recent_elapsed_raw_parts = [1.0, 3.2, 1.4]
    elapsed = sum(recent_elapsed_raw_parts)
    recent_since_start_raw_parts = [1710332816.6118143, 1710332846.1313]
    single_tag = {"is_test": True}
    tags = [single_tag] * (len(recent_elapsed_raw_parts) - 1) + [{"is_test": False}]
    step_range = range(0, 1)
    world_info = WorldInfo(0, 0)
    callback = ParserNDHandler()
    with pytest.raises(NDHandlerError):
        callback(
            metric_name,
            elapsed,
            recent_elapsed_raw_parts,
            recent_since_start_raw_parts,
            tags,
            step_range,
            world_info,
            {},
        )
