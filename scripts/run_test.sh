#!/bin/bash
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

pushd $SCRIPT_DIR/..

# install vescale
pushd python && pip3 install -r requirements.txt --cache-dir ${HOME}/.cache/pip && pip3 install . e && popd

# jump to test folder
pushd test/
export PYTHONPATH=$(pwd):$PYTHONPATH

# run test
for file in $(find . -name 'test_*.py')
do  
    pkill -9 python3
    pytest -s ${file}
    pkill -9 python3
done

# return
popd
popd