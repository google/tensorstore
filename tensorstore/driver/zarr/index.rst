.. _zarr-driver:

``zarr`` Driver
===============

`Zarr <https://github.com/zarr-developers/zarr-python>`_ is a chunked
array storage format based on the `NumPy data type model
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_.

The ``zarr`` driver provides access to Zarr-format arrays backed by
any supported :ref:`key_value_store`.  It supports reading, writing,
creating new arrays, and resizing arrays.

Zarr supports arrays with `structural data types
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_
specifying multiple named fields that are packed together.
TensorStore fully supports such arrays, but each field must be opened
separately.

.. json:schema:: driver/zarr

Compressors
-----------

Chunk data is encoded according to the
:json:schema:`driver/zarr.metadata.compressor` specified in the metadata.

.. json:schema:: driver/zarr/Compressor

The following compressors are supported:

.. json:schema:: driver/zarr/Compressor/zlib
.. json:schema:: driver/zarr/Compressor/blosc
.. json:schema:: driver/zarr/Compressor/bz2

Codec
-----

Within a :ref:`schema`, the compression parameters may also be specified
separately as a :json:schema:`Codec`:

.. json:schema:: driver/zarr/Codec

Limitations
-----------

Filters are not supported.

Datetime/timedelta data types are not supported.

TensorStore supports the non-standard `bfloat16
<https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ data type as an
extension.  On little endian platforms, the official `Zarr Python library
<https://github.com/zarr-developers/zarr-python>`_ is capable of reading arrays
created with the ``bfloat16`` data type provided that a bfloat16 numpy data type
has been registered.  The TensorStore Python library registers such a data type,
as does TensorFlow and JAX.
