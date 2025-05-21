.. _driver/png:

``png`` Driver
=====================

The ``png`` driver specifies a TensorStore backed by a png image file.
The read volume is indexed by "height" (y), "width" (x), "channel".

.. json:schema:: driver/png

.. json:schema:: TensorStoreUrl/png

Auto detection
--------------

This driver supports :ref:`auto-detection<auto-driver>` based on the
signature at the start of the file.
