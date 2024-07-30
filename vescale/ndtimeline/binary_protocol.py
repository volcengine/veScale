################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

import pickle
import io
import gc
from typing import Any, Callable

from .exceptions import ProtocolValidationError


def dumps(v):
    return pickle.dumps(v, protocol=4)


def loads(binary):
    gc.disable()
    res = pickle.loads(binary)
    gc.enable()
    return res


dumps_fn = dumps
loads_fn = loads


# +---------------------------------------------------------------+

# | Magic Number 1Byte | Protocol Version 1Byte | Reserved 2Byte  |

# +---------------------------------------------------------------+

# |                   Payload  Length  4Byte                      |

# +---------------------------------------------------------------+

# |                          Payload                              |

# +---------------------------------------------------------------+

# | EOF Symbol 1Byte |

# +------------------+

# Both Payload Length and Maigc Number are Little Endian


MAGIC_NUMBER = (0x9C).to_bytes(length=1, byteorder="little")
MAGIC_BYTES_LEN = len(MAGIC_NUMBER)
PROTOCOL_VERSION_0 = (0x0).to_bytes(length=1, byteorder="little")
PROTOCOL_VERSION_BYTES_LEN = len(PROTOCOL_VERSION_0)
RESERVED = b"00"
RESERVED_BYTES_LEN = len(RESERVED)
EOF_SYMBOL = b"\n"
EOF_SYMBOL_BYTES_LEN = len(EOF_SYMBOL)
MAX_PAYLOAD_LEN = 1024 * 1024 * 128  # 128MiB
PAYLOAD_LEN_BYTES_LEN = 4


# encode_package encode payload to package
def encode_package(payload: bytes) -> bytes:
    payload_len = len(payload)
    if payload_len > MAX_PAYLOAD_LEN:
        raise ValueError(f"payload size {payload_len}, larger than max size {MAX_PAYLOAD_LEN}")
    payload_len_bytes = payload_len.to_bytes(length=PAYLOAD_LEN_BYTES_LEN, byteorder="little")
    # memory efficient
    return b"".join([MAGIC_NUMBER, PROTOCOL_VERSION_0, RESERVED, payload_len_bytes, payload, EOF_SYMBOL])


# v: any pickable object
def serialize_to_package(v: Any):
    # payload = pickle.dumps(v, protocol=4)
    payload = dumps_fn(v)
    return encode_package(payload)


def recv_and_validate(recv_func: Callable, preload_data: bytearray) -> bytes:
    magic_bytes = read_or_recv(MAGIC_BYTES_LEN, recv_func, preload_data)
    if magic_bytes != MAGIC_NUMBER:
        raise ProtocolValidationError("MAGIC_NUMBER field is broken")
    pt_version_bytes = read_or_recv(PROTOCOL_VERSION_BYTES_LEN, recv_func, preload_data)
    if pt_version_bytes != PROTOCOL_VERSION_0:
        raise ProtocolValidationError("PROTOCOL_VERSION_0 field is broken")
    reserved_bytes = read_or_recv(RESERVED_BYTES_LEN, recv_func, preload_data)
    if reserved_bytes != RESERVED:
        raise ProtocolValidationError(f"RESERVED field is {reserved_bytes}, should be {RESERVED}")
    payload_len_bytes = read_or_recv(PAYLOAD_LEN_BYTES_LEN, recv_func, preload_data)
    payload_len = int.from_bytes(payload_len_bytes, byteorder="little")
    if payload_len > MAX_PAYLOAD_LEN:
        raise ProtocolValidationError(f"payload_len {payload_len} loger than {MAX_PAYLOAD_LEN}")
    payload = read_or_recv(payload_len, recv_func, preload_data)
    eof = read_or_recv(EOF_SYMBOL_BYTES_LEN, recv_func, preload_data)
    if eof != EOF_SYMBOL:
        raise ProtocolValidationError("EOF field is broken")
    return payload


def recv_to_buf(size: int, recv: Callable, preload_data: bytearray):
    assert len(preload_data) <= size
    buf = io.BytesIO()
    buf.write(preload_data)
    remaining = size - len(preload_data)
    del preload_data[: len(preload_data)]
    while remaining > 0:
        chunk = recv(8192)
        n = len(chunk)
        if n == 0:
            raise BrokenPipeError("recv 0 byte from socket")
        if n <= remaining:
            buf.write(chunk)
            remaining -= n
        else:
            buf.write(chunk[:remaining])
            preload_data.extend(chunk[remaining:])
            return buf.getvalue()
    return buf.getvalue()


def read_or_recv(size: int, recv: Callable, preload_data: bytearray):
    if len(preload_data) >= size:
        res = bytes(preload_data[:size])
        del preload_data[:size]
        return res
    else:
        return recv_to_buf(size, recv, preload_data)
