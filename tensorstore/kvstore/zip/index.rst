
.. _zip-kvstore-driver:

``zip`` Key-Value Store driver
======================================================

The ``zip`` driver implements support for reading from
`ZIP <https://en.wikipedia.org/wiki/ZIP_(file_format)>`_ format
files on top of a base key-value store. (Not all ZIP features are supported.)

.. json:schema:: kvstore/zip

.. json:schema:: KvStoreUrl/zip

Example JSON specifications
---------------------------

.. code-block:: json

   { "driver": "zip",
     "kvstore": "gs://my-bucket/path/to/file.zip" }

Auto detection
--------------

This driver supports :ref:`auto-detection<auto-driver>` based on the
signature at the end of the file.

.. warning::

   To limit the amount of data that is read, auto-detection is only
   supported if the end of file comment does not exceed 4096
   bytes. ZIP archives with comments up to the maximum length of 65535
   bytes are still supported without auto-detection, however.

Limitations
-----------

Writing is not supported, nor are all ZIP compression formats.
