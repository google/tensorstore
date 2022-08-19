.. _neuroglancer-precomputed-driver:

``neuroglancer_precomputed`` Driver
===================================

The ``neuroglancer_precomputed`` driver provides access to `Neuroglancer Precomputed format <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_ volumes backed by any supported :ref:`key_value_store`.  It supports reading, writing, and creating new volumes.

Multiscale volumes are supported, but each scale must be opened individually.

.. json:schema:: driver/neuroglancer_precomputed

Example JSON specifications
---------------------------

.. code-block:: json
   :caption: Example: Opening the first (or only) scale of an existing multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
   }


.. code-block:: json
   :caption: Example: Opening an existing scale by index.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "scale_index": 1
   }


.. code-block:: json
   :caption: Example: Opening an existing scale by resolution.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "scale_metadata": {
       "resolution": [4, 4, 40]
     }
   }

.. code-block:: json
   :caption: Example: Opening an existing scale by dimension units.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "schema": {
       "dimension_units": ["4nm", "4nm", "40nm", null]
     }
   }

.. code-block:: json
   :caption: Example: Opening an existing scale by key.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "scale_metadata": {
       "key": "4_4_40"
     }
   }

.. code-block:: json
   :caption: Example: Creating a new scale in an existing multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "scale_metadata": {
       "size": [40000, 50000, 10000],
       "encoding": "compressed_segmentation",
       "compressed_segmentation_block_size": [8, 8, 8],
       "chunk_size": [64, 64, 64],
       "resolution": [8, 8, 40]
     }
   }

.. code-block:: json
   :caption: Example: Creating a new multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": "gs://my-bucket/path/to/volume/",
     "multiscale_metadata": {
       "type": "segmentation",
       "data_type": "uint64",
       "num_channels": 1
     },
     "scale_metadata": {
       "size": [40000, 50000, 10000],
       "encoding": "compressed_segmentation",
       "compressed_segmentation_block_size": [8, 8, 8],
       "chunk_size": [64, 64, 64],
       "resolution": [8, 8, 40]
     }
   }


Mapping to TensorStore Schema
-----------------------------

.. admonition:: Example with unsharded raw encoding
   :class: example

   For the following `info <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#info-json-file-specification>`_ metadata file:

   .. doctest-output:: code-block json

      >>> def get_metadata(spec={}, **kwargs):
      ...     context = ts.Context()
      ...     kvstore = {'driver': 'memory'}
      ...     ts.open(dict(spec,
      ...                  driver='neuroglancer_precomputed',
      ...                  kvstore=kvstore),
      ...             context=context,
      ...             create=True,
      ...             **kwargs).result()
      ...     return ts.open(
      ...         {
      ...             'driver': 'json',
      ...             'kvstore': kvstore,
      ...             'path': 'info'
      ...         },
      ...         context=context).result().read().result()[()]
      >>> def get_schema(metadata):
      ...     context = ts.Context()
      ...     kvstore = {'driver': 'memory'}
      ...     ts.open({
      ...         'driver': 'json',
      ...         'kvstore': kvstore,
      ...         'path': 'info'
      ...     },
      ...             context=context).result().write(metadata).result()
      ...     return ts.open(
      ...         {
      ...             'driver': 'neuroglancer_precomputed',
      ...             'kvstore': kvstore
      ...         },
      ...         context=context).result().schema
      >>> metadata = get_metadata({
      ...     'multiscale_metadata': {
      ...         'num_channels': 2,
      ...         'data_type': 'uint8'
      ...     },
      ...     'scale_metadata': {
      ...         'resolution': [8, 8, 8],
      ...         'chunk_size': [100, 200, 300],
      ...         'sharding': None,
      ...         'size': [1000, 2000, 3000],
      ...         'voxel_offset': [20, 30, 40]
      ...     }
      ... })
      >>> metadata  # doctest:+JSON_OUTPUT
      {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint8",
        "num_channels": 2,
        "scales": [ {
            "chunk_sizes": [[100, 200, 300]],
            "encoding": "raw",
            "key": "8_8_8",
            "resolution": [8.0, 8.0, 8.0],
            "size": [1000, 2000, 3000],
            "voxel_offset": [20, 30, 40]
          }],
        "type": "image"
      }

   the corresponding :json:schema:`Schema` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [20, 30, 40, 0],
          "inner_order": [3, 2, 1, 0],
          "read_chunk": {"shape": [100, 200, 300, 2]},
          "write_chunk": {"shape": [100, 200, 300, 2]}
        },
        "codec": {"driver": "neuroglancer_precomputed", "encoding": "raw"},
        "dimension_units": [[8.0, "nm"], [8.0, "nm"], [8.0, "nm"], null],
        "domain": {
          "exclusive_max": [1020, 2030, 3040, 2],
          "inclusive_min": [20, 30, 40, 0],
          "labels": ["x", "y", "z", "channel"]
        },
        "dtype": "uint8",
        "rank": 4
      }

.. admonition:: Example with unsharded compressed segmentation encoding
   :class: example

   For the following `info <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#info-json-file-specification>`_ metadata file:

   .. doctest-output:: code-block json

      >>> metadata = get_metadata({
      ...     'multiscale_metadata': {
      ...         'num_channels': 2,
      ...         'data_type': 'uint64'
      ...     },
      ...     'scale_metadata': {
      ...         'resolution': [8, 8, 8],
      ...         'chunk_size': [100, 200, 300],
      ...         'sharding': None,
      ...         'size': [1000, 2000, 3000],
      ...         'voxel_offset': [20, 30, 40],
      ...         'encoding': 'compressed_segmentation',
      ...     }
      ... })
      >>> metadata  # doctest:+JSON_OUTPUT
      {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint64",
        "num_channels": 2,
        "scales": [ {
            "chunk_sizes": [[100, 200, 300]],
            "compressed_segmentation_block_size": [8, 8, 8],
            "encoding": "compressed_segmentation",
            "key": "8_8_8",
            "resolution": [8.0, 8.0, 8.0],
            "size": [1000, 2000, 3000],
            "voxel_offset": [20, 30, 40]
          }],
        "type": "segmentation"
      }

   the corresponding :json:schema:`Schema` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "codec_chunk": {"shape": [8, 8, 8, 1]},
          "grid_origin": [20, 30, 40, 0],
          "inner_order": [3, 2, 1, 0],
          "read_chunk": {"shape": [100, 200, 300, 2]},
          "write_chunk": {"shape": [100, 200, 300, 2]}
        },
        "codec": {"driver": "neuroglancer_precomputed", "encoding": "compressed_segmentation"},
        "dimension_units": [[8.0, "nm"], [8.0, "nm"], [8.0, "nm"], null],
        "domain": {
          "exclusive_max": [1020, 2030, 3040, 2],
          "inclusive_min": [20, 30, 40, 0],
          "labels": ["x", "y", "z", "channel"]
        },
        "dtype": "uint64",
        "rank": 4
      }

.. admonition:: Example with sharded raw encoding
   :class: example

   For the following `info <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#info-json-file-specification>`_ metadata file:

   .. doctest-output:: code-block json

      >>> metadata = get_metadata(
      ...     {
      ...         'multiscale_metadata': {
      ...             'num_channels': 2,
      ...             'data_type': 'uint8'
      ...         },
      ...         'scale_metadata': {
      ...             'resolution': [8, 8, 8],
      ...             'chunk_size': [64, 64, 64],
      ...             'size': [34432, 39552, 51508],
      ...             'voxel_offset': [20, 30, 40]
      ...         }
      ...     },
      ...     chunk_layout=ts.ChunkLayout(write_chunk_elements=8000000000),
      ... )
      >>> metadata  # doctest:+JSON_OUTPUT
      {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint8",
        "num_channels": 2,
        "scales": [ {
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
            "key": "8_8_8",
            "resolution": [8.0, 8.0, 8.0],
            "sharding": {
              "@type": "neuroglancer_uint64_sharded_v1",
              "data_encoding": "gzip",
              "hash": "identity",
              "minishard_bits": 6,
              "minishard_index_encoding": "gzip",
              "preshift_bits": 9,
              "shard_bits": 15
            },
            "size": [34432, 39552, 51508],
            "voxel_offset": [20, 30, 40]
          }],
        "type": "image"
      }

   the corresponding :json:schema:`Schema` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [20, 30, 40, 0],
          "inner_order": [3, 2, 1, 0],
          "read_chunk": {"shape": [64, 64, 64, 2]},
          "write_chunk": {"shape": [2048, 2048, 2048, 2]}
        },
        "codec": {
          "driver": "neuroglancer_precomputed",
          "encoding": "raw",
          "shard_data_encoding": "gzip"
        },
        "dimension_units": [[8.0, "nm"], [8.0, "nm"], [8.0, "nm"], null],
        "domain": {
          "exclusive_max": [34452, 39582, 51548, 2],
          "inclusive_min": [20, 30, 40, 0],
          "labels": ["x", "y", "z", "channel"]
        },
        "dtype": "uint8",
        "rank": 4
      }

Data type
~~~~~~~~~

The ``neuroglancer_precomputed`` data types map to TensorStore data types of the
same name:

- :json:schema:`~dtype.uint8`
- :json:schema:`~dtype.int8`
- :json:schema:`~dtype.uint16`
- :json:schema:`~dtype.int16`
- :json:schema:`~dtype.uint32`
- :json:schema:`~dtype.int32`
- :json:schema:`~dtype.uint64`
- :json:schema:`~dtype.float32`

Note that internally the ``neuroglancer_precomputed`` format always uses little
endian encoding.

Domain
~~~~~~

The :json:schema:`Schema.domain` has a rank of 4 and includes both the chunked
dimensions as well as the channel dimension.  The
:json:schema:`IndexDomain.shape` of the :json:schema:`Schema.domain` corresponds
to :json:schema:`driver/neuroglancer_precomputed.scale_metadata.size`;
:json:schema:`IndexDomain.inclusive_min` corresponds to
:json:schema:`driver/neuroglancer_precomputed.scale_metadata.voxel_offset`.  The
channel dimension always has an origin of 0.

The :ref:`dimension labels<dimension-labels>` are always :json:`["x", "y", "z",
"channel"]`.

The bounds are not resizable.

.. admonition:: Example
   :class: example

   For the following `info <https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md#info-json-file-specification>`_ metadata file:

   .. doctest-output:: code-block json

      >>> metadata = get_metadata({
      ...     'multiscale_metadata': {
      ...         'num_channels': 2,
      ...         'data_type': 'uint8'
      ...     },
      ...     'scale_metadata': {
      ...         'resolution': [8, 8, 8],
      ...         'chunk_size': [64, 64, 64],
      ...         'size': [1000, 2000, 3000],
      ...         'voxel_offset': [20, 30, 40]
      ...     }
      ... })
      >>> metadata  # doctest:+JSON_OUTPUT
      {
        "@type": "neuroglancer_multiscale_volume",
        "data_type": "uint8",
        "num_channels": 2,
        "scales": [ {
            "chunk_sizes": [[64, 64, 64]],
            "encoding": "raw",
            "key": "8_8_8",
            "resolution": [8.0, 8.0, 8.0],
            "size": [1000, 2000, 3000],
            "voxel_offset": [20, 30, 40]
          }],
        "type": "image"
      }

   the corresponding :json:schema:`IndexDomain` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).domain.to_json()  # doctest:+JSON_OUTPUT
      {
        "exclusive_max": [1020, 2030, 3040, 2],
        "inclusive_min": [20, 30, 40, 0],
        "labels": ["x", "y", "z", "channel"]
      }

Chunk layout
~~~~~~~~~~~~

The :json:schema:`ChunkLayout.read_chunk` shape, specifying the granularity at
which reads may be performed efficiently, corresponds to
:json:schema:`driver/neuroglancer_precomputed.scale_metadata.chunk_size`.  The
``channel`` dimension is always unchunked, i.e. the chunk size is equal to the
number of channels.

The :json:schema:`ChunkLayout.grid_origin` always matches the origin of the
:json:schema:`~Schema.domain`.

With respect to the dimension order of ``[x, y, z, channel]``, when using the
:json:`"raw"` :json:schema:`driver/neuroglancer_precomputed/Codec.encoding`, the
``neuroglancer_precomputed`` format always stores the data within chunks in
colexicographic order (i.e. Fortran order).

When using the unsharded format, the
shape is equal to the :json:schema:`ChunkLayout.read_chunk` shape.

When using the sharded format, if the sharding parameters are such that each
shard corresponds to a rectangular region, then the
:json:schema:`ChunkLayout.write_chunk` shape corresponds to the shape of each
shard.  Otherwise, the :json:schema:`~ChunkLayout.write_chunk` shape corresponds
to the shape of the entire volume, rounded up to a multiple of the
:json:schema:`~ChunkLayout.read_chunk` shape.

When using the :json:`"raw"` or :json:`"jpeg"`
:json:schema:`driver/neuroglancer_precomputed/Codec.encoding`, hard constraints
on the :json:schema:`ChunkLayout.codec_chunk` must not be specified.

When using the :json:`"compressed_segmentation"`
:json:schema:`driver/neuroglancer_precomputed/Codec.encoding`, the
:json:schema:`ChunkLayout.codec_chunk` shape corresponds to the
:json:schema:`~driver/neuroglancer_precomputed.scale_metadata.compressed_segmentation_block_size`.
Note that the codec chunk size along the channel dimension is always 1.

Selection of chunk layout when creating a new array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new array, the read chunk shape may be constrained explicitly
via :json:schema:`ChunkLayout/Grid.shape` or implicitly via
:json:schema:`ChunkLayout/Grid.aspect_ratio` and
:json:schema:`ChunkLayout/Grid.elements`.  If
:json:schema:`ChunkLayout/Grid.elements` is not specified for the
:json:schema:`~ChunkLayout.read_chunk`, the default is 1 million elements per
chunk.  A suitable base (read) chunk shape will be chosen based on these
constraints.

The write chunk shape may also be constrained separately, either explicitly via
:json:schema:`ChunkLayout/Grid.shape` or implicitly via
:json:schema:`ChunkLayout/Grid.elements`.  If no constraints on the write chunk
shape are specified, it will be set to match the read chunk shape and the
unsharded format will be used.  Otherwise, suitable sharding parameters will be
chosen automatically to satisfy the constraints.  If
:json:schema:`ChunkLayout/Grid.elements` is not specified for the write chunk,
the unsharded format is used.  Due to the limitations of the sharding format,
any :json:schema:`ChunkLayout/Grid.aspect_ratio` constraints on the write chunk
shape are ignored.

When using the :json:`"compressed_segmentation"`
:json:schema:`driver/neuroglancer_precomputed/Codec.encoding`, the compressed
segmentation block size is chosen automatically based on the
:json:schema:`ChunkLayout.codec_chunk` constraints.  Both explicit constraints
via :json:schema:`ChunkLayout/Grid.shape` and implicit constraints via
:json:schema:`ChunkLayout/Grid.aspect_ratio` and
:json:schema:`ChunkLayout/Grid.elements` are supported.  If
:json:schema:`ChunkLayout/Grid.elements` is not specified, the default is 512
elements per chunk.

.. admonition:: Example of unconstrained chunk layout
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'neuroglancer_precomputed',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     domain=ts.IndexDomain(
   ...         inclusive_min=[20, 30, 40, 0],
   ...         shape=[1000, 2000, 3000, 2],
   ...     )).result().chunk_layout
   ChunkLayout({
     'grid_origin': [20, 30, 40, 0],
     'inner_order': [3, 2, 1, 0],
     'read_chunk': {'shape': [81, 81, 81, 2]},
     'write_chunk': {'shape': [81, 81, 81, 2]},
   })

.. admonition:: Example of unconstrained chunk layout with compressed segmentation encoding
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'neuroglancer_precomputed',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint32,
   ...     codec=ts.CodecSpec({
   ...         'driver': 'neuroglancer_precomputed',
   ...         'encoding': 'compressed_segmentation',
   ...     }),
   ...     domain=ts.IndexDomain(
   ...         inclusive_min=[20, 30, 40, 0],
   ...         shape=[1000, 2000, 3000, 2],
   ...     )).result().chunk_layout
   ChunkLayout({
     'codec_chunk': {'shape': [8, 8, 8, 1]},
     'grid_origin': [20, 30, 40, 0],
     'inner_order': [3, 2, 1, 0],
     'read_chunk': {'shape': [81, 81, 81, 2]},
     'write_chunk': {'shape': [81, 81, 81, 2]},
   })

.. admonition:: Example of chunk layout with separate read and write chunk constraints
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'neuroglancer_precomputed',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     chunk_layout=ts.ChunkLayout(
   ...         chunk_aspect_ratio=[2, 1, 1, 0],
   ...         read_chunk_elements=2000000,
   ...         write_chunk_elements=1000000000,
   ...     ),
   ...     domain=ts.IndexDomain(
   ...         inclusive_min=[20, 30, 40, 0],
   ...         shape=[1000, 2000, 3000, 2],
   ...     )).result().chunk_layout
   ChunkLayout({
     'grid_origin': [20, 30, 40, 0],
     'inner_order': [3, 2, 1, 0],
     'read_chunk': {'shape': [159, 79, 79, 2]},
     'write_chunk': {'shape': [1113, 1264, 632, 2]},
   })

.. admonition:: Example of chunk layout with explicit chunk shapes
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'neuroglancer_precomputed',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     chunk_layout=ts.ChunkLayout(
   ...         read_chunk_shape=[64, 64, 64, 2],
   ...         write_chunk_shape=[512, 512, 512, 2],
   ...     ),
   ...     domain=ts.IndexDomain(
   ...         inclusive_min=[20, 30, 40, 0],
   ...         shape=[1000, 2000, 3000, 2],
   ...     )).result().chunk_layout
   ChunkLayout({
     'grid_origin': [20, 30, 40, 0],
     'inner_order': [3, 2, 1, 0],
     'read_chunk': {'shape': [64, 64, 64, 2]},
     'write_chunk': {'shape': [512, 512, 512, 2]},
   })

Codec
~~~~~

Within the :json:schema:`Schema.codec`, the compression parameters are
represented in the same way as in the
:json:schema:`~driver/neuroglancer_precomputed.scale_metadata`:

.. json:schema:: driver/neuroglancer_precomputed/Codec

It is an error to specify any other :json:schema:`Codec.driver`.

Fill value
~~~~~~~~~~

The ``neuroglancer_precomputed`` format does not support specifying a fill
value.  TensorStore always assumes a fill value of :json:`0`.

Dimension Units
~~~~~~~~~~~~~~~

The dimension units of the first three (``x``, ``y``, and ``z``) dimensions
always have a base unit of :json:`"nm"`; the multiplier corresponds to the
:json:schema:`~driver/neuroglancer_precomputed.scale_metadata.resolution`.  It
is an error to specify a base unit other than :json:`"nm"` for these dimensions.

The final (``channel``) dimension always has an unspecified base unit.  It is an
error to specify a unit for this dimension.

When creating a new scale, if neither :json:schema:`~Schema.dimension_units` nor
:json:schema:`~driver/neuroglancer_precomputed.scale_metadata.resolution` is
specified, a unit of :json:`"1nm"` is used by default.

When opening an existing scale, the scale to open may be selected based on the
specified :json:schema:`~Schema.dimension_units`.

Limitations
-----------

Resizing is not supported.

.. warning:: Writing to volumes in the sharded format is supported,
   but because updating a shard requires rewriting it entirely, write
   operations may be very inefficient unless special care is taken:

   1. It is most efficient to group writes by shard (i.e. according to the
      :json:schema:`ChunkLayout.write_chunk` shape).

   2. The temporary memory required to write a shard is 2 to 3 times the size of
      the shard.  It is therefore advised that the shards be kept as small as
      possible (while still avoiding an excess number of objects in the
      underlying :ref:`key-value store<key_value_store>`).
