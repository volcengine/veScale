#! usr/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PATCH_NAME="patched_pytorch_v2.2.1_rc3.patch"
PATCH_PATH=$SCRIPT_DIR/$PATCH_NAME 

if [ ! -f "$PATCH_PATH" ]; then
    echo "Error: patch does not exist."
    exit 1
fi

git clone --branch v2.2.1-rc3 --depth 1 https://github.com/pytorch/pytorch.git
pushd pytorch 
git apply $PATCH_PATH
git submodule sync 
git submodule update --init --recursive --depth 1
pip3 install -r requirements.txt
python3 setup.py install

popd
rm -rf pytorch
