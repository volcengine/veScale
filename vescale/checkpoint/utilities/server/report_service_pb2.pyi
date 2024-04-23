from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OmniStoreGatherRequest(_message.Message):
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

class OmniStoreGatherResponse(_message.Message):
    __slots__ = ("contents",)
    CONTENTS_FIELD_NUMBER: _ClassVar[int]
    contents: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, contents: _Optional[_Iterable[bytes]] = ...) -> None: ...

class OmniStoreBroadcastRequest(_message.Message):
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

class OmniStoreBroadcastResponse(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...

class OmniStoreGetStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OmniStoreGetStatusResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: bytes
    def __init__(self, status: _Optional[bytes] = ...) -> None: ...
