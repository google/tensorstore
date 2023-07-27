.. _file-kvstore-driver:

``file`` Key-Value Store driver
===============================

The ``file`` driver uses the filesystem as a key-value store directly.  A key
directly specifies a path under a given root directory; the value is stored as
the file contents.

Locking provided by the filesystem is used to safely allow concurrent access
from multiple processes.  (The locking protocol used does not block readers.)
Provided that shared locking is supported, concurrent access from multiple
machines to a network filesystem is also safe.

.. json:schema:: kvstore/file

.. json:schema:: KvStoreUrl/file

.. json:schema:: Context.file_io_concurrency

.. json:schema:: Context.file_io_sync

Durability of writes
--------------------

By default, this driver ensures all writes are durable, meaning that committed
data won't be lost in the event that the process or machine crashes.

In cases where durability is not necessary, faster write performance may be
achieved by setting :json:schema:`Context.file_io_sync` to :json:``false``.

.. code-block:: json

   {"driver": "file",
    "path": "/local/path/",
    "file_io_sync": false}

Limitations
-----------

.. note::

   This driver is only supported on Windows 10 RS1 or later, due to its reliance
   on file operations with POSIX semantics.
