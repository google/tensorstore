.. _driver/webp:

``webp`` Driver
=====================

The ``webp`` driver specifies a TensorStore backed by a webp image file.
The read volume is indexed by "height" (y), "width" (x), "channel".

The ``webp`` driver supports either 3 (RGB) or 4 (RGBA) channel WebP images.
Lossy WebP images are encoded as YUV 422, incurring some perceptual loss.

.. json:schema:: driver/webp

.. json:schema:: TensorStoreUrl/webp

Auto detection
--------------

This driver supports :ref:`auto-detection<auto-driver>` based on the
signature at the start of the file.
