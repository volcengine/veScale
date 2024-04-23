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
from common_dtensor import DTensorTestBase
from torch.testing._internal.common_utils import run_tests

import os
import pathlib
import re

_ROOT_FOLDER_NAME = "vescale"
_ROOT_SUBFOLDERS_TO_TEST = ("test", "vescale")
_FILE_PATTERN_TO_TEST = "*.py"  # TODO: add cpp
_FILE_NAME_TO_EXCLUDE = ("__init__.py",)
_COPYRIGHT = r"copyright"


class TestCopyright(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    def test_copyright(self):
        # change directory to root
        this_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.join(this_dir, "..", "..")
        os.chdir(root_dir)
        self.assertTrue(os.path.realpath(".").endswith(_ROOT_FOLDER_NAME))
        # recursively find all file path
        file_paths = []
        for folder_name in _ROOT_SUBFOLDERS_TO_TEST:
            folder_path = pathlib.Path(folder_name)
            self.assertTrue(os.path.exists(folder_path))
            file_paths += list(  # noqa: C400
                fp
                for fp in folder_path.rglob(_FILE_PATTERN_TO_TEST)
                if os.path.basename(fp) not in _FILE_NAME_TO_EXCLUDE
            )
        # open each file and check copyright
        failed_file_pathes = []
        copyright = re.compile(_COPYRIGHT, re.IGNORECASE)
        for fp in file_paths:
            print(f"{fp}: ...")
            self.assertTrue(os.path.exists(fp))
            with open(fp) as file:
                content = file.read()
                if not bool(copyright.search(content)):
                    failed_file_pathes.append(fp)
        # if fail, print instruction
        for fp in failed_file_pathes:
            print(f"{fp}: has no `{_COPYRIGHT}`!")
        self.assertTrue(
            len(failed_file_pathes) == 0,
            msg=f"{len(failed_file_pathes)} files has no `{_COPYRIGHT}`!\n"
            f"Follow `HowToAddCopyright.md` to add `{_COPYRIGHT}` on the head of failed files!",
        )


if __name__ == "__main__":
    run_tests()
