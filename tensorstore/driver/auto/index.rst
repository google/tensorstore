.. _driver/auto:

``auto`` Driver
===============

Many of the supported :ref:`TensorStore drivers<tensorstore-drivers>`
support format auto-detection: an existing TensorStore can be opened
by specifying just a :ref:`key-value store<key_value_store>`, and the
appropriate TensorStore driver is determined automatically.

Both TensorStore drivers, such as :ref:`zarr3<driver/zarr3>` or
:ref:`jpeg<driver/jpeg>`, and key-value store drivers, such as
:ref:`zip<kvstore/zip>` or :ref:`ocdbt<kvstore/ocdbt>`,
can be auto-detected.

Format auto-detection is used implicitly whenever a
:json:schema:`KvStore JSON spec<KvStore>` or :json:schema:`KvStore
URL<KvStoreUrl>` is specified in place of a :json:schema:`TensorStore
JSON spec<TensorStore>`.

Format auto-detection can also be requested explicitly using the
:json:schema:`auto driver identifier<driver/auto>` or the
:json:schema:`TensorStoreUrl/auto`.

.. note::

   Auto-detection is performed when opening the TensorStore.  The
   detected driver can be determined by querying the Spec of the open
   TensorStore.

.. warning::

   Auto-detection involves additional read requests to determine the
   format and adds latency to the open operation.  While the number of
   bytes read is small, in cases where limiting the number of read
   requests, and/or limiting the latency of open operations is
   important, auto-detection should be avoided.

.. hint::

   In distributed execution settings, where the same TensorStore may
   be opened concurrently by many processes, if auto-detection is
   required, it is recommended to first auto-detect the format from a
   single process (e.g. a controller process) and then use the
   resolved spec to open the TensorStore from worker processes.

.. json:schema:: driver/auto

.. json:schema:: TensorStoreUrl/auto

Examples
--------

Auto-detecting an array
^^^^^^^^^^^^^^^^^^^^^^^

A :ref:`zarr3<driver/zarr3>` TensorStore can be detected from its path:

  >>> # Create new array
  >>> await ts.open("file:///tmp/dataset.zarr|zarr3",
  ...               dtype="int32",
  ...               shape=[5],
  ...               create=True)
  TensorStore({
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
    },
    'driver': 'zarr3',
    'dtype': 'int32',
    'kvstore': {'driver': 'file', 'path': '/tmp/dataset.zarr/'},
    'metadata': {
      'chunk_grid': {'configuration': {'chunk_shape': [5]}, 'name': 'regular'},
      'chunk_key_encoding': {'name': 'default'},
      'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
      'data_type': 'int32',
      'fill_value': 0,
      'node_type': 'array',
      'shape': [5],
      'zarr_format': 3,
    },
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })
  >>> # Open with auto-detection
  >>> await ts.open("file:///tmp/dataset.zarr")
  TensorStore({
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
    },
    'driver': 'zarr3',
    'dtype': 'int32',
    'kvstore': {'driver': 'file', 'path': '/tmp/dataset.zarr/'},
    'metadata': {
      'chunk_grid': {'configuration': {'chunk_shape': [5]}, 'name': 'regular'},
      'chunk_key_encoding': {'name': 'default'},
      'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
      'data_type': 'int32',
      'fill_value': 0,
      'node_type': 'array',
      'shape': [5],
      'zarr_format': 3,
    },
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })

Explicitly constructing a :py:obj:`~tensorstore.Spec` demonstrates the
explicit syntax for using the ``auto`` driver:

  >>> ts.Spec("file:///tmp/dataset|auto")
  Spec({'driver': 'auto', 'kvstore': {'driver': 'file', 'path': '/tmp/dataset'}})
  >>> ts.Spec("file:///tmp/dataset")
  Spec({'driver': 'auto', 'kvstore': {'driver': 'file', 'path': '/tmp/dataset'}})

Chaining TensorStore adapters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorStore adapters like :ref:`cast<driver/cast>` can also be
used in conjunction with format auto-detection:

  >>> ts.Spec("file:///tmp/dataset.zarr|cast:int64")
  Spec({
    'base': {
      'driver': 'auto',
      'kvstore': {'driver': 'file', 'path': '/tmp/dataset.zarr'},
    },
    'driver': 'cast',
    'dtype': 'int64',
  })
  >>> ts.Spec("file:///tmp/dataset.zarr|auto|cast:int64")
  Spec({
    'base': {
      'driver': 'auto',
      'kvstore': {'driver': 'file', 'path': '/tmp/dataset.zarr'},
    },
    'driver': 'cast',
    'dtype': 'int64',
  })
  >>> await ts.open("file:///tmp/dataset.zarr|cast:int64")
  TensorStore({
    'base': {
      'driver': 'zarr3',
      'dtype': 'int32',
      'kvstore': {'driver': 'file', 'path': '/tmp/dataset.zarr/'},
      'metadata': {
        'chunk_grid': {
          'configuration': {'chunk_shape': [5]},
          'name': 'regular',
        },
        'chunk_key_encoding': {'name': 'default'},
        'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
        'data_type': 'int32',
        'fill_value': 0,
        'node_type': 'array',
        'shape': [5],
        'zarr_format': 3,
      },
    },
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
    },
    'driver': 'cast',
    'dtype': 'int64',
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })

Multiple auto-detection steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple steps of auto-detection are also possible.  Here, a
:ref:`zarr3<driver/zarr3>` TensorStore at the root of an
:ref:`OCDBT<kvstore/ocdbt>` database can also be detected just
from the path to the OCDBT database.

  >>> # Create new array within new OCDBT database
  >>> await ts.open("file:///tmp/dataset.ocdbt|ocdbt|zarr3",
  ...               dtype="int32",
  ...               shape=[5],
  ...               create=True)
  TensorStore({
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
      'ocdbt_coordinator': {},
    },
    'driver': 'zarr3',
    'dtype': 'int32',
    'kvstore': {
      'base': {'driver': 'file', 'path': '/tmp/dataset.ocdbt/'},
      'config': {
        'compression': {'id': 'zstd'},
        'max_decoded_node_bytes': 8388608,
        'max_inline_value_bytes': 100,
        'uuid': '...',
        'version_tree_arity_log2': 4,
      },
      'driver': 'ocdbt',
    },
    'metadata': {
      'chunk_grid': {'configuration': {'chunk_shape': [5]}, 'name': 'regular'},
      'chunk_key_encoding': {'name': 'default'},
      'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
      'data_type': 'int32',
      'fill_value': 0,
      'node_type': 'array',
      'shape': [5],
      'zarr_format': 3,
    },
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })
  >>> # Open with auto-detection
  >>> await ts.open("file:///tmp/dataset.ocdbt")
  TensorStore({
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
      'ocdbt_coordinator': {},
    },
    'driver': 'zarr3',
    'dtype': 'int32',
    'kvstore': {
      'base': {'driver': 'file', 'path': '/tmp/dataset.ocdbt/'},
      'config': {
        'compression': {'id': 'zstd'},
        'max_decoded_node_bytes': 8388608,
        'max_inline_value_bytes': 100,
        'uuid': '...',
        'version_tree_arity_log2': 4,
      },
      'driver': 'ocdbt',
    },
    'metadata': {
      'chunk_grid': {'configuration': {'chunk_shape': [5]}, 'name': 'regular'},
      'chunk_key_encoding': {'name': 'default'},
      'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
      'data_type': 'int32',
      'fill_value': 0,
      'node_type': 'array',
      'shape': [5],
      'zarr_format': 3,
    },
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })

Note that auto-detection fails if the zarr array is *not* at the root
of the OCDBT database:

  >>> # Create new array within new OCDBT database
  >>> await ts.open(
  ...     "file:///tmp/dataset2.ocdbt|ocdbt:path/within/database|zarr3",
  ...     dtype="int32",
  ...     shape=[5],
  ...     create=True)
  TensorStore({
    'context': {
      'cache_pool': {},
      'data_copy_concurrency': {},
      'file_io_concurrency': {},
      'file_io_locking': {},
      'file_io_mode': {},
      'file_io_sync': True,
      'ocdbt_coordinator': {},
    },
    'driver': 'zarr3',
    'dtype': 'int32',
    'kvstore': {
      'base': {'driver': 'file', 'path': '/tmp/dataset2.ocdbt/'},
      'config': {
        'compression': {'id': 'zstd'},
        'max_decoded_node_bytes': 8388608,
        'max_inline_value_bytes': 100,
        'uuid': '...',
        'version_tree_arity_log2': 4,
      },
      'driver': 'ocdbt',
      'path': 'path/within/database/',
    },
    'metadata': {
      'chunk_grid': {'configuration': {'chunk_shape': [5]}, 'name': 'regular'},
      'chunk_key_encoding': {'name': 'default'},
      'codecs': [{'configuration': {'endian': 'little'}, 'name': 'bytes'}],
      'data_type': 'int32',
      'fill_value': 0,
      'node_type': 'array',
      'shape': [5],
      'zarr_format': 3,
    },
    'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
  })
  >>> # Open with auto-detection
  >>> await ts.open("file:///tmp/dataset2.ocdbt")
  Traceback (most recent call last):
      ...
  ValueError: FAILED_PRECONDITION: Error opening "auto" driver: Failed to detect format for "" in OCDBT database at local file "/tmp/dataset2.ocdbt/"...

.. _multi-step-auto-detection-algorithm:

Multi-step auto-detection algorithm
-----------------------------------

Given a base key-value store, auto-detection of the final TensorStore
driver proceeds as follows:

1. A :ref:`single auto-detection step<single-auto-detection-step>` is
   applied to the current base key-value store, which results in a
   list of candidate formats.

2. There are no candidate formats, or more than one candidate format,
   auto-detection fails with an error.

3. Two kinds of drivers can be detected:

   a. If the detected format is a TensorStore driver, it is applied to
      the current base key-value store, opened, and auto-detection is
      complete.

   b. If the detected format is a key-value store adapter driver, it
      is applied to the current base key-value store, and opened.  The
      adapted key-value store becomes the new base key-value store and
      detection continues at step 1.

.. _single-auto-detection-step:

Single-step auto-detection algorithm
------------------------------------

1. If the base key-value store potentially specifies a single file
   (i.e. it has a non-empty path not ending ``/``), *single
   file-format detection* is attempted.

   a. Each single-file format that supports auto-detection specifies
      the number of bytes at the beginning and end of the file that
      are required for auto-detection.

   b. The prefix and suffix of the file is requested, using the
      maximum prefix/suffix length required by any format for
      auto-detection.

   c. If the file is not found, directory format detection continues
      at step 2.

   d. If the file is found, the single-file formats that match the
      prefix and suffix read from the file are returned as candidates.

2. If the base key-value store refers to a directory, *directory
   format detection* is attempted.

   a. Each directory format that supports auto-detection specifies one
      or more relative paths that should be checked to determine if
      they are present.

   b. The complete set of relative paths required by any directory
      format is checked.

   c. The directory formats that match (based on the set of relative
      paths that are present) are returned as candidates.
