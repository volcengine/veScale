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

import os
from vescale.ndtimeline.world_info import WorldInfo
from vescale.ndtimeline.handlers import LocalRawNDHandler
from vescale.ndtimeline.variables import LOCAL_LOGGING_PATH


def test_basic_usage():
    h = LocalRawNDHandler(run_id=0, chunk_sz=10, backup_cnt=3)
    file_name = "timeline_run0_raw.log"
    h("test_metric", 1.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    assert os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name))
    for _ in range(4):
        h("test_metric", 1.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    h("test_metric2", 2.0, [1.0], [1.0], [{}], range(0, 1), WorldInfo(0, 0), {})
    assert os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".2"))
    assert not os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".4"))
    os.remove(os.path.join(LOCAL_LOGGING_PATH, file_name))
    for i in range(1, 4):
        os.remove(os.path.join(LOCAL_LOGGING_PATH, file_name + "." + str(i)))
    assert not os.path.exists(os.path.join(LOCAL_LOGGING_PATH, file_name + ".2"))
