#!/usr/bin/bash
set -e

# build torchvision
git clone -b v0.17.0 https://github.com/pytorch/vision.git
pushd vision
python3 setup.py install
popd
rm -rf vision
python3 -c "import torchvision"
pip3 list | grep torch
