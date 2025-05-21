.. _driver/jpeg:

``jpeg`` Driver
=====================

The ``jpeg`` driver specifies a TensorStore backed by a jpeg.
The read volume is indexed by "height" (y), "width" (x), "channel".

.. json:schema:: driver/jpeg

.. json:schema:: TensorStoreUrl/jpeg

Auto detection
--------------

This driver supports :ref:`auto-detection<auto-driver>` based on the
signature at the start of the file.
