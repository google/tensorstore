Tutorial
========

Reading and writing a local N5 dataset
--------------------------------------

Create a new :ref:`N5<n5-driver>` dataset on the local filesystem using the
:ref:`file-kvstore-driver`:

   >>> import tensorstore as ts
   >>> import numpy as np
   >>> dataset = ts.open({
   ...     'driver': 'n5',
   ...     'kvstore': {
   ...         'driver': 'file',
   ...         'path': '/tmp/dataset',
   ...     },
   ...     'metadata': {
   ...         'compression': {
   ...             'type': 'gzip'
   ...         },
   ...         'dataType': 'uint32',
   ...         'dimensions': [1000, 20000],
   ...         'blockSize': [100, 100],
   ...     },
   ...     'create': True,
   ...     'delete_existing': True,
   ... }).result()

Asynchronously write to to a sub-region:

   >>> write_future = dataset[80:82, 99:102].write([[1, 2, 3], [4, 5, 6]])

Wait for the write to complete using :py:obj:`tensorstore.Future.result`:

   >>> write_future.result()

In an `async<async>` function (or with top-level `await<await>` support),
`await<await>` can also be used for interoperability with `asyncio`:

   >>> await write_future

Subscript assignment can also be used to write synchronously:

   >>> dataset[80:82, 99:102] = [[1, 2, 3], [4, 5, 6]]

Read back a larger region that contains the region that was written (positions
not written have the fill value of ``0``):

   >>> dataset[80:83, 99:102].read().result()
   array([[1, 2, 3],
          [4, 5, 6],
          [0, 0, 0]], dtype=uint32)

Reading the Janelia FlyEM Hemibrain dataset
-------------------------------------------

This example demonstrates accessing the `Janelia FlyeEM Hemibrain 1.1
segmentation <https://www.janelia.org/project-team/flyem/hemibrain>`_ using the
:ref:`neuroglancer-precomputed-driver`.

While this dataset is public, the :ref:`gcs-kvstore-driver` currently requires
that you supply :ref:`Google Cloud credentials<gcs-authentication>`.

Open the dataset asynchronously to obtain a :py:obj:`tensorstore.Future`:

.. doctest::

   >>> import tensorstore as ts
   >>> import numpy as np
   >>> dataset_future = ts.open({
   ...     'driver': 'neuroglancer_precomputed',
   ...     'kvstore': {
   ...         'driver': 'gcs',
   ...         'bucket': 'neuroglancer-janelia-flyem-hemibrain',
   ...     },
   ...     'path': 'v1.1/segmentation',
   ...     # Use 100MB in-memory cache.
   ...     'context': {
   ...         'cache_pool': {
   ...             'total_bytes_limit': 100_000_000
   ...         }
   ...     },
   ...     'recheck_cached_data': 'open',
   ... })
   >>> dataset_future
   <tensorstore.Future object at 0x...>

Wait for the open to complete:

.. doctest::

   >>> dataset = dataset_future.result()
   >>> dataset
   TensorStore({
     'context': {'cache_pool': {'total_bytes_limit': 100000000}},
     'driver': 'neuroglancer_precomputed',
     'dtype': 'uint64',
     'kvstore': {
       'bucket': 'neuroglancer-janelia-flyem-hemibrain',
       'driver': 'gcs',
     },
     'multiscale_metadata': {'num_channels': 1, 'type': 'segmentation'},
     'path': 'v1.1/segmentation',
     'recheck_cached_data': 'open',
     'scale_index': 0,
     'scale_metadata': {
       'chunk_size': [64, 64, 64],
       'compressed_segmentation_block_size': [8, 8, 8],
       'encoding': 'compressed_segmentation',
       'key': '8.0x8.0x8.0',
       'resolution': [8.0, 8.0, 8.0],
       'sharding': {
         '@type': 'neuroglancer_uint64_sharded_v1',
         'data_encoding': 'gzip',
         'hash': 'identity',
         'minishard_bits': 6,
         'minishard_index_encoding': 'gzip',
         'preshift_bits': 9,
         'shard_bits': 15,
       },
       'size': [34432, 39552, 41408],
       'voxel_offset': [0, 0, 0],
     },
     'transform': {
       'input_exclusive_max': [34432, 39552, 41408, 1],
       'input_inclusive_min': [0, 0, 0, 0],
       'input_labels': ['x', 'y', 'z', 'channel'],
     },
   })

In an `async<async>` function, a `tensorstore.Future` is also compatible with
`await<await>`.

.. doctest::

   >>> dataset = await dataset_future
   
.. doctest::

   >>> dataset.domain
   { "x": [0, 34432), "y": [0, 39552), "z": [0, 41408), "channel": [0, 1) }

There is only a single channel, so create a 3-d view without the
:python:`'channel'` dimension:

.. doctest::

   >>> dataset_3d = dataset[ts.d['channel'][0]]
   >>> dataset_3d.domain
   { "x": [0, 34432), "y": [0, 39552), "z": [0, 41408) }
   
Create a view of a 100x100x1 slice from the middle, without performing any I/O:

.. doctest::

   >>> x = dataset_3d[15000:15100, 15000:15100, 20000]
   >>> x
   TensorStore({
     'context': {'cache_pool': {'total_bytes_limit': 100000000}},
     'driver': 'neuroglancer_precomputed',
     'dtype': 'uint64',
     'kvstore': {
       'bucket': 'neuroglancer-janelia-flyem-hemibrain',
       'driver': 'gcs',
     },
     'multiscale_metadata': {'num_channels': 1, 'type': 'segmentation'},
     'path': 'v1.1/segmentation',
     'recheck_cached_data': 'open',
     'scale_index': 0,
     'scale_metadata': {
       'chunk_size': [64, 64, 64],
       'compressed_segmentation_block_size': [8, 8, 8],
       'encoding': 'compressed_segmentation',
       'key': '8.0x8.0x8.0',
       'resolution': [8.0, 8.0, 8.0],
       'sharding': {
         '@type': 'neuroglancer_uint64_sharded_v1',
         'data_encoding': 'gzip',
         'hash': 'identity',
         'minishard_bits': 6,
         'minishard_index_encoding': 'gzip',
         'preshift_bits': 9,
         'shard_bits': 15,
       },
       'size': [34432, 39552, 41408],
       'voxel_offset': [0, 0, 0],
     },
     'transform': {
       'input_exclusive_max': [15100, 15100],
       'input_inclusive_min': [15000, 15000],
       'input_labels': ['x', 'y'],
       'output': [
         {'input_dimension': 0},
         {'input_dimension': 1},
         {'offset': 20000},
         {},
       ],
     },
   })
   >>> x.domain
   { "x": [15000, 15100), "y": [15000, 15100) }

Read the slice asynchronously using the `tensorstore.TensorStore.read` method to
obtain a `tensorstore.Future`:

.. doctest::

   >>> read_future = x.read()

Wait for the read to complete:

.. doctest::

   >>> read_future.result()
   array([[1194100437, 1194100437, 1194100437, ..., 1408314276, 1408314276,
           1408314276],
          [1194100437, 1194100437, 1194100437, ..., 1408314276, 1408314276,
           1408314276],
          [1194100437, 1194100437, 1194100437, ..., 1161117856, 1161117856,
           1161117856],
          ...,
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053],
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053],
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053]], dtype=uint64)

Conversion to an :py:obj:`numpy.ndarray` also implicitly performs a synchronous
read (which hits the in-memory cache since the same region was just retrieved):

.. doctest::

   >>> np.array(dataset_3d[15000:15100, 15000:15100, 20000])
   array([[1194100437, 1194100437, 1194100437, ..., 1408314276, 1408314276,
           1408314276],
          [1194100437, 1194100437, 1194100437, ..., 1408314276, 1408314276,
           1408314276],
          [1194100437, 1194100437, 1194100437, ..., 1161117856, 1161117856,
           1161117856],
          ...,
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053],
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053],
          [1132030694, 1132030694, 1132030694, ..., 5813054053, 5813054053,
           5813054053]], dtype=uint64)
