#! usr/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PATCH_NAME="patched_torchdistX_9c1b9f.patch"
PATCH_PATH=$SCRIPT_DIR/$PATCH_NAME

if [ ! -f "$PATCH_PATH" ]; then
    echo "Error: patch does not exist."
    exit 1
fi

git clone --depth 1 https://github.com/pytorch/torchdistx.git
pushd torchdistx
git pull
git checkout 9c1b9f5cb2fa36bfb8b70ec07c40ed42a33cc87a
git apply $PATCH_PATH
git submodule sync
git submodule update --init --recursive --depth 1
cmake -DTORCHDIST_INSTALL_STANDALONE=ON  -GNinja -DCAKE_CXX_COMPILER_LAUNCHER=ccache -B build
cmake --build build
pip3 install .

popd
rm -rf torchdistx
