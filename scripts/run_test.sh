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

# run test
while IFS= read -r -d '' file; do
  echo "=== Running: $file ==="
  pkill -9 -f 'python(3)? .*pytest' 2>/dev/null || true
  pytest "$file"   # add -s if you want to see stdout
  pkill -9 -f 'python(3)? .*pytest' 2>/dev/null || true
done < <(find . -type f -name 'test_*.py' -print0 | sort -z)

# return
popd
popd
