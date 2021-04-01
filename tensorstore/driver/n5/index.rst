.. _n5-driver:

``n5`` Driver
=============

The ``n5`` driver provides access to `N5 <https://github.com/saalfeldlab/n5>`_ arrays backed by any supported :ref:`key_value_store`.  It supports reading, writing, creating new datasets, and resizing datasets.

.. json:schema:: driver/n5

Compression
-----------

.. json:schema:: driver/n5#/definitions/compression

The following compression methods are supported:

.. json:schema:: driver/n5/Compression/raw
.. json:schema:: driver/n5/Compression/gzip
.. json:schema:: driver/n5/Compression/bzip2
.. json:schema:: driver/n5/Compression/xz
.. json:schema:: driver/n5/Compression/blosc

Codec
-----

Within a :ref:`schema`, the compression parameters may also be specified
separately as a :json:schema:`Codec`:

.. json:schema:: driver/n5/Codec

Limitations
-----------

Datasets with varlength chunks are not supported.

The N5 specification does not define a fill value, but TensorStore
assumes a fill value of 0.
