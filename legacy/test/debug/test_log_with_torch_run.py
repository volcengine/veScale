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
import unittest
import subprocess

dir_name = "torchrun_scripts"

target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)


def run_script(fname, env=None):
    current_dir = os.getcwd()
    os.chdir(target_dir)
    result = subprocess.run(["torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=4", fname], env=env)
    os.chdir(current_dir)
    return result.returncode


class DebugTorchrunTestSuite(unittest.TestCase):
    def test_simple_std_out(self):
        self.assertEqual(run_script("simple_std_out.py"), 0)

    def test_simple_only_rank1(self):
        self.assertEqual(run_script("simple_only_rank1.py"), 0)

    def test_simple_logging(self):
        self.assertEqual(run_script("simple_logging.py"), 0)

    def test_simple_set_env(self):
        my_env = os.environ
        my_env["VESCALE_DEBUG_MODE"] = "1"
        self.assertEqual(run_script("simple_std_out.py", my_env), 0)


if __name__ == "__main__":
    unittest.main()
