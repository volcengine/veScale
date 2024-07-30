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

import logging
import sys
import os


class NDTimelineLogger:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            level = logging.getLevelName(os.getenv("VESCALE_NDTIMELINE_LOG_LEVEL", "INFO"))
            if isinstance(level, str):
                # if NDTIMELINE_LOG_LEVEL has an illegal value
                # logging.getLevelName returns a str `Level xxx`
                level = logging.WARNING
            formatter = logging.Formatter(
                "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d][pid:%(process)d] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(formatter)
            cls.instance = logging.getLogger("ndtimeline")
            cls.instance.addHandler(handler)
            cls.instance.setLevel(level)
            cls.instance.propagate = False
        return cls.instance
