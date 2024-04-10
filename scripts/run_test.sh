#!/bin/bash
set -ex

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd "$SCRIPT_DIR"/..

# install vescale
pushd python && pip3 install -r requirements.txt --cache-dir "${HOME}"/.cache/pip && pip3 install -e . && popd

# jump to test folder
pushd test/

PYTHONPATH=$(pwd):$PYTHONPATH

export PYTHONPATH

# run test
while IFS= read -r -d '' file
do
    pkill -9 python3 || true # ok if nothing to kill
    pytest -s "${file}"
    pkill -9 python3 || true
done <   <(find . -name 'test_*.py' -print0)

# return
popd
popd