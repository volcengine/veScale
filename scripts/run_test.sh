#!/bin/bash

echo "run all tests (for open source)"

set -ex

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd "$SCRIPT_DIR"/..

# install vescale
pip3 install -r requirements.txt --cache-dir "${HOME}"/.cache/pip && pip3 install -e .

# jump to test folder
pushd test/

export PYTHONPATH=$(pwd):$PYTHONPATH
export VESCALE_SINGLE_DEVICE_RAND="1"

# run test
while IFS= read -r -d '' file
do
    pkill -9 python3 || true # ok if nothing to kill
    pytest -s "${file}"
    pkill -9 python3 || true
done <   <(find . -name 'test_*.py' -not -name 'test_open_llama_*.py' -print0)

# return
popd
popd