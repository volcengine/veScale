from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OmniStoreWriteRequest(_message.Message):
    __slots__ = ("content", "name")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    name: str
    def __init__(self, content: _Optional[bytes] = ..., name: _Optional[str] = ...) -> None: ...

class OmniStoreWriteResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OmniStoreReadRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OmniStoreReadResponse(_message.Message):
    __slots__ = ("content",)
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    def __init__(self, content: _Optional[bytes] = ...) -> None: ...

class OmniStoreRenameRequest(_message.Message):
    __slots__ = ("src", "dst", "overwrite")
    SRC_FIELD_NUMBER: _ClassVar[int]
    DST_FIELD_NUMBER: _ClassVar[int]
    OVERWRITE_FIELD_NUMBER: _ClassVar[int]
    src: str
    dst: str
    overwrite: bool
    def __init__(self, src: _Optional[str] = ..., dst: _Optional[str] = ..., overwrite: bool = ...) -> None: ...

class OmniStoreRenameResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OmniStoreRemoveRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OmniStoreRemoveResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class OmniStoreListdirRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OmniStoreListdirResponse(_message.Message):
    __slots__ = ("names",)
    NAMES_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, names: _Optional[_Iterable[str]] = ...) -> None: ...

class OmniStoreExistsRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OmniStoreExistsResponse(_message.Message):
    __slots__ = ("exists",)
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    exists: bool
    def __init__(self, exists: bool = ...) -> None: ...
