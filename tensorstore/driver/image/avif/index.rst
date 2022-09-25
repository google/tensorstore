.. _driver/avif:

``avif`` Driver
=====================

The ``avif`` driver specifies a TensorStore backed by an avif image file.
The read volume is indexed by "height" (y), "width" (x), "channel".

The ``avif`` driver supports between 1 and 4 channel AVIF images. While AVIF
images are encoded in YUV planes with varying bit representations, the ``avif``
driver converts them, regardless of their encoded depth, to an 8-bit
Grey/GreyA/RGB/RGBA equivalent image.

.. json:schema:: driver/avif

