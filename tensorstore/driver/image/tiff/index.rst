.. _driver/tiff:

``tiff`` Driver
=====================

The ``tiff`` driver specifies a TensorStore backed by a TIFF image file.
The read volume is indexed by "height" (y), "width" (x), "channel".

This driver is currently experimental and only supports a very limited subset
of TIFF files.


.. json:schema:: driver/tiff

.. json:schema:: TensorStoreUrl/tiff

Auto detection
--------------

This driver supports :ref:`auto-detection<auto-driver>` based on the
signature at the start of the file.
