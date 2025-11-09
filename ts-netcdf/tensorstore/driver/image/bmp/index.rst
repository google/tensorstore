.. _driver/bmp:

``bmp`` Driver
=====================

The ``bmp`` driver specifies a TensorStore backed by a BMP image file.
The read volume is indexed by "height" (y), "width" (x), "channel".
The ``bmp`` driver is experimental.

.. json:schema:: driver/bmp

.. json:schema:: TensorStoreUrl/bmp

Auto detection
--------------

This driver supports :ref:`auto-detection<driver/auto>` based on the
signature at the start of the file.
