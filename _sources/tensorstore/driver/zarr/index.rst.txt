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

.. json-schema:: schema.yml
   :title: JSON Schema

Compressors
-----------

Chunk data is encoded according to the :json-member:`compressor` specified in the metadata.

.. json-schema:: schema.yml#/definitions/compressor

The following compressors are supported:

.. json-schema:: schema.yml#/definitions/compressor-zlib
.. json-schema:: schema.yml#/definitions/compressor-blosc
.. json-schema:: schema.yml#/definitions/compressor-bz2

Limitations
-----------

Filters are not supported.

Datetime/timedelta data types are not supported.
