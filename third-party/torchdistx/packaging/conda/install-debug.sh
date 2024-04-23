#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -o errexit

if [[ $(uname -s) == Darwin ]]; then
    filter="-type d -name *.dSYM"
else
    filter="-type f -name *.debug"
fi

find "$SRC_DIR/build" $filter -exec cp -a "{}" "$PREFIX/lib" ";"
