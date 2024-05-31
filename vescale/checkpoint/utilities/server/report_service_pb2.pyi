################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VeScaleCheckpointGatherRequest(_message.Message):
    __slots__ = ("tag", "rank", "content", "with_result")
    TAG_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    WITH_RESULT_FIELD_NUMBER: _ClassVar[int]
    tag: str
    rank: int
    content: bytes
    with_result: bool
    def __init__(
        self,
        tag: _Optional[str] = ...,
        rank: _Optional[int] = ...,
        content: _Optional[bytes] = ...,
        with_result: bool = ...,
    ) -> None: ...

class VeScaleCheckpointGatherResponse(_message.Message):
    __slots__ = ("contents",)
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class VeScaleCheckpointBroadcastRequest(_message.Message):
    __slots__ = ("tag", "rank", "content", "src_rank")
    TAG_FIELD_NUMBER: _ClassVar[int]
    RANK_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    SRC_RANK_FIELD_NUMBER: _ClassVar[int]
    tag: str
    rank: int
    content: bytes
    src_rank: int
    def __init__(
        self,
        tag: _Optional[str] = ...,
        rank: _Optional[int] = ...,
        content: _Optional[bytes] = ...,
        src_rank: _Optional[int] = ...,
    ) -> None: ...

class VeScaleCheckpointBroadcastResponse(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...

class VeScaleCheckpointGetStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class VeScaleCheckpointGetStatusResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: bytes
    def __init__(self, status: _Optional[bytes] = ...) -> None: ...
