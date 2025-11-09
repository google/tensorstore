.. _driver/downsample:

``downsample`` Driver
=====================

Virtual read-only view that performs downsampling.

Downsampling is performed on-the-fly to compute exactly the positions of the
downsampled view that are read.

.. json:schema:: driver/downsample

The following downsampling methods are supported:

.. json:schema:: DownsampleMethod


Downsampling origin
-------------------

Downsampling is always performed with respect to an origin of 0 in all
dimensions in the :json:schema:`~driver/downsample.base` TensorStore.
To use a different origin, translate the domain by specifying an
:json:schema:`IndexTransform` via the
:json:schema:`TensorStore.transform` property of
:json:schema:`~driver/downsample.base`.

Translating the domain of the downsampled view by a given
:literal:`offset` is equivalent to translating the domain of
:json:schema:`~driver/downsample.base` by :python:`offset *
downsample_factors`.  However, there is no translation of the
downsampled view that is exactly equivalent to translating the domain
of :json:schema:`~driver/downsample.base` by an offset that is not a
multiple of :json:schema:`~driver/downsample.downsample_factors`.
