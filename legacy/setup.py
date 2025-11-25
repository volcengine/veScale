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
import re

try:
    from setuptools import Extension, find_packages, setup  # noqa: F401
except ImportError:
    from distutils.core import Extension, setup  # noqa: F401

NAME = "vescale"
VERSION = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    version_file_path = "vescale/__init__.py"
    with open(version_file_path) as f:
        (version,) = re.findall('__version__ = "(.*)"', f.read())
    VERSION = version

except:  # noqa: E722
    raise ValueError(f"Failed to get version from {version_file_path}.")


def package_files(directory, f=None):
    if isinstance(directory, (list, tuple)):
        l = [package_files(d, f=f) for d in directory]
        return [item for sublist in l for item in sublist]
    directory = os.path.join(BASE_DIR, directory)
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            if callable(f):
                if f(filename):
                    paths.append(os.path.join("..", path, filename))
                continue
            if isinstance(f, str):
                if re.match(f, filename):
                    paths.append(os.path.join("..", path, filename))
                continue
            paths.append(os.path.join("..", path, filename))
    # print(paths)
    return paths


with open("requirements.txt") as f_req:
    required = []
    for line in f_req:
        if line.startswith("-"):
            continue
        line = line.strip()
        required.append(line)

setup(
    name=NAME,
    version=VERSION,
    description="veScale: A PyTorch Native LLM Training Framework",
    author="",
    author_email="",
    url="",
    packages=["vescale"],
    package_data={
        "vescale": package_files(["vescale"]),
    },
    install_requires=required,
    python_requires=">=3.8",
)
