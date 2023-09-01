
.. _zip-kvstore-driver:

``zip`` Key-Value Store driver
======================================================

The ``zip`` driver implements support for reading from 
`ZIP <https://en.wikipedia.org/wiki/ZIP_(file_format)>`_ format
files on top of a base key-value store. (Not all ZIP features are supported.)

.. json:schema:: kvstore/zip

Example JSON specifications
---------------------------

.. code-block:: json

   { "driver": "zip",
     "kvstore": "gs://my-bucket/path/to/file.zip" }

Limitations
-----------

Writing is not supported, nor are all ZIP compression formats.
