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

import unittest
from unittest import TestCase
from vescale.pipe._schedules.instruction_base import register_instruction, registed_functions, InstructionBuilder


class InstructionRegistrationTest(TestCase):
    def test_pp_registed_function(self):
        """
        Tests instruction registration.
        """

        @register_instruction(name="instruction_one")
        def instruction_one(input):
            print(input)
            return input

        assert "instruction_one" in registed_functions

    def test_instruction_constructor(self):
        """
        Tests instruction construction.
        """

        @register_instruction(name="I1")
        def instruction_one(input):
            return input + 1

        @register_instruction(name="I2")
        def instruction_two(input):
            return input * 2

        @register_instruction(name="B")
        def bubble(input):
            return input

        instructions = {0: "B,I1,I1,I1,I1,I2,I2", 1: "B,I2,I2,I2,I2,I1,I1,I1"}
        builder = InstructionBuilder()
        builder.build_from_dict(instructions)
        builder.draw_instructions()


if __name__ == "__main__":
    unittest.main()
