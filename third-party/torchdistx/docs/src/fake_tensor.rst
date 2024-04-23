.. currentmodule:: torchdistx.fake

Fake Tensor
===========
Fake tensors, similar to meta tensors, carry no data; however, unlike meta
tensors which report ``meta`` as their device, fake tensors act as if they were
allocated on a real device. The following example shows how the two tensors
types differ:

::

    >>> import torch
    >>>
    >>> from torchdistx.fake import fake_mode
    >>>
    >>> # Meta tensors are always "allocated" on the `meta` device.
    >>> a = torch.ones([10], device="meta")
    >>> a
    tensor(..., device='meta', size(10,))
    >>> a.device
    device(type='meta')
    >>>
    >>> # Fake tensors are always "allocated" on the specified device.
    >>> with fake_mode():
    ...     b = torch.ones([10])
    ...
    >>> b
    tensor(..., size(10,), fake=True)
    >>> b.device
    device(type='cpu')

Fake tensors, like meta tensors, rely on the meta backend for their operation.
In that sense meta tensors and fake tensors can be considered close cousins.
Fake tensors are just an alternative interface to the meta backend and have
mostly the same tradeoffs as meta tensors.

API
---
The API consists mainly of the ``fake_mode()`` function that acts as a Python
context manager. Any tensor constructed within its scope will be forced to be
fake.

.. autofunction:: fake_mode

There are also two convenience functions offered as part of the API:

.. autofunction:: is_fake
.. autofunction:: meta_like

Use Cases
---------
Fake tensors were originally meant as a building block for :doc:`deferred_init`.
However they are not necessarily bound to that use case and can also be used for
other purposes. For instance they serve as a surprisingly good learning tool for
inspecting large model architectures that cannot fit on a consumer-grade PC:

::

    >>> import torch
    >>>
    >>> from transformers import BlenderbotModel, BlenderbotConfig
    >>>
    >>> from torchdistx.fake import fake_mode
    >>>
    >>> # Instantiate Blenderbot on a personal laptop with 8GB RAM.
    >>> with fake_mode():
    ...     m = BlenderbotModel(BlenderbotConfig())
    ...
    >>> # Check out the model layers and their parameters.
    >>> m
    BlenderbotModel(...)
