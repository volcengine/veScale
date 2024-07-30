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

import os

SOCK_TIMEOUT_CLIENT: float = 2.0  # seconds
SOCK_PARENT_DIR: str = "/opt/tiger/tmp/ndtimeline"
SOCK_PATH: str = os.path.join(SOCK_PARENT_DIR, "ndtimeline.sock")  # /opt/tiger/tmp/ndtimeline/ndtimeline.sock
LOCAL_LOGGING_PATH: str = SOCK_PARENT_DIR
DEFAULT_CUDA_EVENT_POOL_SIZE: int = 20
NDTIMELINE_INNER_GLOBAL_STEP_KEY: str = "_inner_global_step"
NDTIMELINE_STREAM_KEY: str = "stream_key"
NDTIMELINE_FLUSH_SEPCIAL: str = "special"
