.. _n5-driver:

``n5`` Driver
=============

The ``n5`` driver provides access to `N5 <https://github.com/saalfeldlab/n5>`_ arrays backed by any supported :ref:`key_value_store`.  It supports reading, writing, creating new datasets, and resizing datasets.

.. json-schema:: schema.yml
   :title: JSON Schema

Compression
-----------

.. json-schema:: schema.yml#/definitions/compression

The following compression methods are supported:

.. json-schema:: schema.yml#/definitions/compression-raw
.. json-schema:: schema.yml#/definitions/compression-gzip
.. json-schema:: schema.yml#/definitions/compression-bzip2
.. json-schema:: schema.yml#/definitions/compression-xz
.. json-schema:: schema.yml#/definitions/compression-blosc

Limitations
-----------

Datasets with varlength chunks are not supported.

The N5 specification does not define a fill value, but TensorStore
assumes a fill value of 0.
