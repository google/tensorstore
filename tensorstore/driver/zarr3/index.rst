.. _zarr3-driver:

``zarr3`` Driver
================

`Zarr v3<zarr-specs:v3/core/v3.0>` is a chunked array storage format.

The ``zarr3`` driver provides access to Zarr v3-format arrays backed by
any supported :ref:`key_value_store`.  It supports reading, writing,
creating new arrays, and resizing arrays.

.. json:schema:: driver/zarr3

.. json:schema:: driver/zarr3/Metadata


Codecs
------

Chunk data is encoded according to the
:json:schema:`~driver/zarr3/Metadata.codecs` specified in the metadata.

.. json:schema:: driver/zarr3/CodecChain

.. json:schema:: driver/zarr3/SingleCodec

.. _zarr3-array-to-array-codecs:

:literal:`Array -> array` codecs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. json:schema:: driver/zarr3/Codec/transpose

.. _zarr3-array-to-bytes-codecs:

:literal:`Array -> bytes` codecs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. json:schema:: driver/zarr3/Codec/bytes

.. json:schema:: driver/zarr3/Codec/sharding_indexed

.. _zarr3-bytes-to-bytes-codecs:

:literal:`Bytes -> bytes` codecs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compression
^^^^^^^^^^^

.. json:schema:: driver/zarr3/Codec/gzip

.. json:schema:: driver/zarr3/Codec/blosc

.. json:schema:: driver/zarr3/Codec/zstd

Checksum
^^^^^^^^

.. json:schema:: driver/zarr3/Codec/crc32c

Chunk key encodings
-------------------

The position of each chunk is encoded as a key according to the
:json:schema:`~driver/zarr3/Metadata.chunk_key_encoding` specified in the
metadata.

.. json:schema:: driver/zarr3/ChunkKeyEncoding

.. json:schema:: driver/zarr3/ChunkKeyEncoding.default

.. json:schema:: driver/zarr3/ChunkKeyEncoding.v2

Mapping to TensorStore Schema
-----------------------------

.. admonition:: Example without sharding
   :class: example

   For the following zarr :json:schema:`~driver/zarr3/Metadata`:

   .. doctest-output:: code-block json

      >>> def get_schema(metadata):
      ...     context = ts.Context()
      ...     kvstore = {'driver': 'memory'}
      ...     ts.open(
      ...         {
      ...             'driver': 'json',
      ...             'kvstore': kvstore,
      ...             'path': 'zarr.json'
      ...         },
      ...         context=context).result().write(metadata).result()
      ...     return ts.open({
      ...         'driver': 'zarr3',
      ...         'kvstore': kvstore,
      ...     },
      ...                    context=context).result().schema
      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1000, 2000, 3000],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 200, 300]}},
        "chunk_key_encoding": {"name": "default"},
        "data_type": "uint16",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 42
      }

   the corresponding :json:schema:`Schema` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [0, 0, 0],
          "inner_order": [0, 1, 2],
          "read_chunk": {"shape": [100, 200, 300]},
          "write_chunk": {"shape": [100, 200, 300]}
        },
        "codec": {
          "codecs": [{"configuration": {"endian": "little"}, "name": "bytes"}],
          "driver": "zarr3"
        },
        "domain": {"exclusive_max": [[1000], [2000], [3000]], "inclusive_min": [0, 0, 0]},
        "dtype": "uint16",
        "fill_value": 42,
        "rank": 3
      }

Data type
~~~~~~~~~

Zarr v3 data types correspond to the TensorStore data type of the same name.

.. json:schema:: driver/zarr3/DataType

Domain
~~~~~~

The :json:schema:`~IndexDomain.shape` of the :json:schema:`Schema.domain`
corresponds to :json:schema:`driver/zarr3/Metadata.shape`.

:ref:`Dimension labels<dimension-labels>` may be specified in the
:json:schema:`Schema.domain`, and correspond to
:json:schema:`driver/zarr3/Metadata.dimension_names`, but with the following differences:

- The `Zarr v3 specification
  <https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#dimension-names>`__
  distinguishes between an empty string (:json:`""`) and an unspecified
  dimension name (indicated by :json:`null`).  In either case, the corresponding
  TensorStore dimension label is the empty string.

- The Zarr v3 specification also permits the same non-empty name to be used for
  more than one dimension, but TensorStore requires that all non-empty dimension
  labels are unique.  If the Zarr metadata specifies dimension names that are
  not valid TensorStore dimension labels, the corresponding TensorStore domain
  simply leaves all dimensions unlabeled.

The upper bounds of the domain are resizable
(i.e. :ref:`implicit<implicit-bounds>`).

As Zarr v3 does not natively support a non-zero origin, the underlying domain always
has a zero origin (:json:schema:`IndexDomain.inclusive_min` is all zero), but it
may be translated by the :json:schema:`~driver/zarr3.transform`.

.. admonition:: Example
   :class: example

   For the following :json:schema:`~driver/zarr3/Metadata`:

   .. doctest-output:: code-block json

      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1000, 2000, 3000],
        "dimension_names": ["x", "y", "z"],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 200, 300]}},
        "chunk_key_encoding": {"name": "default"},
        "data_type": "uint16",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": 0
      }

   the corresponding :json:schema:`IndexDomain` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).domain.to_json()  # doctest:+JSON_OUTPUT
      {
        "exclusive_max": [[1000], [2000], [3000]],
        "inclusive_min": [0, 0, 0],
        "labels": ["x", "y", "z"]
      }

Chunk layout
~~~~~~~~~~~~

The :json:schema:`ChunkLayout.write_chunk` shape, specifying the granularity at
which writes may be performed efficiently, corresponds to the top-level
:json:schema:`~driver/zarr3/Metadata.chunk_grid.configuration.chunk_shape`.

The :json:schema:`ChunkLayout.grid_origin` is always the zero vector.

The :json:schema:`ChunkLayout.inner_order` depends on the
:json:schema:`driver/zarr3/Metadata.codecs` that are in use.  With just the
default :json:schema:`~driver/zarr3/Codec/bytes` codec, the inner order is
:python:`[0, 1, ..., n-1]` (C order); this order may be altered by the
:json:schema:`~driver/zarr3/Codec/transpose` codec.

When no :json:schema:`sharding codec<driver/zarr3/Codec/sharding_indexed>` is in
use, the :json:schema:`ChunkLayout.read_chunk` is equal to the
:json:schema:`ChunkLayout.write_chunk` shape.

When using a :json:schema:`sharding codec<driver/zarr3/Codec/sharding_indexed>`,
the :json:schema:`ChunkLayout.read_chunk` shape corresponds to the inner-most
:json:schema:`sub-chunk
shape<driver/zarr3/Codec/sharding_indexed.configuration.chunk_shape>`.

Selection of chunk layout when creating a new array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new array, the read and write chunk shapes may be constrained
explicitly via :json:schema:`ChunkLayout/Grid.shape` or implicitly via
:json:schema:`ChunkLayout/Grid.aspect_ratio` and
:json:schema:`ChunkLayout/Grid.elements`.  If
:json:schema:`ChunkLayout/Grid.elements` is not specified, the default is 1
million elements per chunk.  Suitable read and write chunk shapes are chosen
automatically based on these constraints, in combination with any constraints
implied by the specified :json:schema:`~driver/zarr3.metadata`.

If the chosen read chunk shape is not equal to the chosen write chunk shape, a
:json:schema:`sharding codec<driver/zarr3/Codec/sharding_indexed>` is inserted
into the codec chain automatically if not already specified.

If a :json:schema:`ChunkLayout.inner_order` constraint is specified, a
:json:schema:`~driver/zarr3/Codec/transpose` codec may be inserted automatically
just before the inner-most `array -> bytes codec<zarr3-array-to-bytes-codecs>`.

.. admonition:: Example of unconstrained chunk layout
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'zarr3',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     shape=[1000, 2000, 3000],
   ... ).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [101, 101, 101]},
     'write_chunk': {'shape': [101, 101, 101]},
   })

.. admonition:: Example of chunk layout with separate read and write chunk constraints
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'zarr3',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     chunk_layout=ts.ChunkLayout(
   ...         chunk_aspect_ratio=[2, 1, 1],
   ...         read_chunk_elements=2000000,
   ...         write_chunk_elements=1000000000,
   ...     ),
   ...     shape=[1000, 2000, 3000],
   ... ).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [200, 100, 100]},
     'write_chunk': {'shape': [1000, 1000, 1000]},
   })

.. admonition:: Example of chunk layout with explicit chunk shapes
   :class: example

   >>> ts.open(
   ...     {
   ...         'driver': 'zarr3',
   ...         'kvstore': {
   ...             'driver': 'memory'
   ...         }
   ...     },
   ...     create=True,
   ...     dtype=ts.uint16,
   ...     chunk_layout=ts.ChunkLayout(
   ...         read_chunk_shape=[64, 64, 64],
   ...         write_chunk_shape=[512, 512, 512],
   ...     ),
   ...     shape=[1000, 2000, 3000],
   ... ).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [64, 64, 64]},
     'write_chunk': {'shape': [512, 512, 512]},
   })

Codec
~~~~~

Within the :json:schema:`Schema.codec`, the chunk codec chain is represented in
the same way as in the :json:schema:`~driver/zarr3/Metadata`:

.. json:schema:: driver/zarr3/Codec

It is an error to specify any other :json:schema:`Codec.driver`.

Fill value
~~~~~~~~~~

The :json:schema:`Schema.fill_value` must be a scalar (rank 0).

As an optimization, chunks that are entirely equal to the fill value are not
stored.

Dimension units
~~~~~~~~~~~~~~~

The :json:schema:`Schema.dimension_units` property corresponds to the
:json:schema:`~driver/zarr3/Metadata.attributes.dimension_units` and
:json:schema:`~driver/n5.metadata.resolution` metadata property.  The base unit
is used directly; it is not converted in any way.
