#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -o errexit

# We perform LTO only if no sanitizer is enabled since they do not play well
# together.
if [[ -z "$TORCHDIST_SANITIZERS" ]]; then
    perform_lto=ON
else
    perform_lto=OFF
fi

cmake -GNinja\
      -DCMAKE_BUILD_TYPE=RelWithDebInfo\
      -DCMAKE_INSTALL_PREFIX="$PREFIX"\
      -DCMAKE_INSTALL_LIBDIR=lib\
      -DCMAKE_FIND_FRAMEWORK=NEVER\
      -DTORCHDIST_TREAT_WARNINGS_AS_ERRORS=ON\
      -DTORCHDIST_PERFORM_LTO=$perform_lto\
      -DTORCHDIST_DEVELOP_PYTHON=OFF\
      -DTORCHDIST_SANITIZERS="$TORCHDIST_SANITIZERS"\
      -S "$SRC_DIR"\
      -B "$SRC_DIR/build"

cmake --build "$SRC_DIR/build"

# Extract the debug symbols; they will be part of the debug package.
find "$SRC_DIR/build" -type f -name "libtorchdistx*"\
    -exec "$SRC_DIR/scripts/strip-debug-symbols" --extract "{}" ";"
