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
import threading


# SynchronizedQueue is used for communications between training process and checkpoint uploading thread
class SynchronizedQueue:
    def __init__(self):
        self._task_done = True
        self._item = None
        self._cond = threading.Condition()

    def put(self, item) -> None:
        with self._cond:
            self._cond.wait_for(lambda: self._task_done)
            self._task_done = False
            self._item = item
            self._cond.notify_all()

    def get(self):
        with self._cond:
            self._cond.wait_for(lambda: self._item is not None)
            item = self._item
            self._item = None
            return item

    def task_done(self):
        with self._cond:
            self._task_done = True
            self._cond.notify_all()

    def join(self, timeout=None) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._task_done, timeout=timeout)
