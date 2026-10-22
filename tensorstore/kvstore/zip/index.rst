
.. _kvstore/zip:

``zip`` Key-Value Store driver
======================================================

The ``zip`` driver implements support for reading and writing
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

This driver supports :ref:`auto-detection<driver/auto>` based on the
signature at the end of the file.

.. warning::

   To limit the amount of data that is read, auto-detection is only
   supported if the end of file comment does not exceed 4096
   bytes. ZIP archives with comments up to the maximum length of 65535
   bytes are still supported without auto-detection, however.

Transactions and Write Amplification
------------------------------------

This driver supports writing and deleting keys inside the ZIP archive. Since
ZIP archives do not support random modification, each write or delete
operation may rewrite the entire ZIP archive. Therefore, using transactions
is strongly recommended to batch multiple write operations and minimize write
amplification.

Limitations
-----------

Not all ZIP features and compression formats are supported:

Compression formats
  - **Reading**: Support is limited to ``deflate``, ``bzip2``, ``zstd``,
    ``xz``, and ``store`` (no compression).
  - **Writing**: Support is limited to ``deflate``, ``zstd``, ``xz``, and
    ``store``. Other compression methods (such as ``lzma``) are not supported.

Standard ZIP encryption, strong encryption, and AES encryption are not
supported.
