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

from typing import Callable, Tuple, Dict, Any
from functools import wraps

TestFunc = Callable[[object], object]


# wrapper to initialize comms (process group) within emulator
def with_comms_emulator(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # launch
        self.init_emulator_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_emulator_pg()

    return wrapper
