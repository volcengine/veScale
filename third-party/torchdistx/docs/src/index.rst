:github_url: https://github.com/pytorch/torchdistx

Torch Distributed Experimental
==============================
Torch Distributed Experimental, or in short torchdistX, contains a collection of
experimental features for which our team wants to gather feedback from our users
before introducing them in the core PyTorch Distributed package. In a sense
features included in torchdistX can be considered in an incubation period.

.. note::
   Please be advised that all features in torchdistX are subject to change and,
   although our team will make its best effort, we do not guarantee any API or
   ABI compatibility between releases. This means you should exercise caution if
   you plan to use torchdistX in production.

Installation
------------
Check out `this section in our README <https://github.com/pytorch/torchdistx/blob/main/README.md#Installation>`_
for installation instructions.

Documentation
-------------
.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Torch Distributed Experimental

   Index <self>

.. toctree::
   :maxdepth: 2
   :caption: Features

   fake_tensor
   deferred_init
   slow_momentum_fsdp
   gossip_grad

.. toctree::
   :maxdepth: 1
   :caption: Design Notes

   fake_tensor_and_deferred_init
