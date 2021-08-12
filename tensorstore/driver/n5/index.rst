.. _n5-driver:

``n5`` Driver
=============

The ``n5`` driver provides access to `N5 <https://github.com/saalfeldlab/n5>`_
arrays backed by any supported :ref:`key_value_store`.  It supports reading,
writing, creating new datasets, and resizing datasets.

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

Mapping to TensorStore Schema
-----------------------------

.. admonition:: Example
   :class: example

   For the following N5 :json:schema:`~driver/n5.metadata`:

   .. doctest-output:: code-block json

      >>> def get_schema(metadata):
      ...     context = ts.Context()
      ...     kvstore = {'driver': 'memory'}
      ...     ts.open(
      ...         {
      ...             'driver': 'json',
      ...             'kvstore': kvstore,
      ...             'path': 'attributes.json'
      ...         },
      ...         context=context).result().write(metadata).result()
      ...     return ts.open({
      ...         'driver': 'n5',
      ...         'kvstore': kvstore
      ...     },
      ...                    context=context).result().schema
      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "dimensions": [1000, 2000, 3000],
        "blockSize": [100, 200, 300],
        "dataType": "uint16",
        "compression": {"type": "raw"}
      }

   the corresponding :json:schema:`Schema` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [0, 0, 0],
          "inner_order": [2, 1, 0],
          "read_chunk": {"shape": [100, 200, 300]},
          "write_chunk": {"shape": [100, 200, 300]}
        },
        "codec": {"compression": {"type": "raw"}, "driver": "n5"},
        "domain": {"exclusive_max": [[1000], [2000], [3000]], "inclusive_min": [0, 0, 0]},
        "dtype": "uint16",
        "rank": 3
      }

Data type
~~~~~~~~~

N5 data types map to TensorStore data types of the same name:

- :json:schema:`~dtype.uint8`
- :json:schema:`~dtype.int8`
- :json:schema:`~dtype.uint16`
- :json:schema:`~dtype.int16`
- :json:schema:`~dtype.uint32`
- :json:schema:`~dtype.int32`
- :json:schema:`~dtype.uint64`
- :json:schema:`~dtype.int64`
- :json:schema:`~dtype.float32`
- :json:schema:`~dtype.float64`

Note that internally the N5 format always uses big endian encoding.

Domain
~~~~~~

The :json:schema:`~IndexDomain.shape` of the :json:schema:`Schema.domain`
corresponds to :json:schema:`driver/n5.metadata.dimensions`.

:ref:`Dimension labels<dimension-labels>` may be specified in the
:json:schema:`Schema.domain`, and correspond to
:json:schema:`driver/n5.metadata.axes`.

The upper bounds of the domain are resizable
(i.e. :ref:`implicit<implicit-bounds>`).

As N5 does not natively support a non-zero origin, the underlying domain always
has a zero origin (:json:schema:`IndexDomain.inclusive_min` is all zero), but it
may be translated by the :json:schema:`~driver/n5.transform`.

.. admonition:: Example
   :class: example

   For the following N5 :json:schema:`~driver/n5.metadata`:

   .. doctest-output:: code-block json

      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "dimensions": [1000, 2000, 3000],
        "blockSize": [100, 200, 300],
        "dataType": "uint16",
        "compression": {"type": "raw"}
      }

   the corresponding :json:schema:`IndexDomain` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).domain.to_json()  # doctest:+JSON_OUTPUT
      {"exclusive_max": [[1000], [2000], [3000]], "inclusive_min": [0, 0, 0]}

Chunk layout
~~~~~~~~~~~~

The N5 format supports a single :json:schema:`driver/n5.metadata.blockSize`
property that corresponds to the :json:schema:`ChunkLayout/Grid.shape`
constraint.

.. admonition:: Example
   :class: example

   For the following N5 :json:schema:`~driver/n5.metadata`:

   .. doctest-output:: code-block json

      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "dimensions": [1000, 2000, 3000],
        "blockSize": [100, 200, 300],
        "dataType": "uint16",
        "compression": {"type": "raw"}
      }

   the corresponding :json:schema:`ChunkLayout` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata).chunk_layout.to_json()  # doctest:+JSON_OUTPUT
      {
        "grid_origin": [0, 0, 0],
        "inner_order": [2, 1, 0],
        "read_chunk": {"shape": [100, 200, 300]},
        "write_chunk": {"shape": [100, 200, 300]}
      }

The :json:schema:`ChunkLayout.grid_origin` is always all-zero.

As the N5 format supports only a single level of chunking, the
:json:schema:`ChunkLayout.read_chunk` and :json:schema:`ChunkLayout.write_chunk`
constraints are combined, and hard constraints on
:json:schema:`ChunkLayout.codec_chunk` must not be specified.

The N5 format always stores the data within chunks in colexicographic order
(i.e. Fortran order).

Selection of chunk layout when creating a new array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new array, the chunk shape may be constrained explicitly via
:json:schema:`ChunkLayout/Grid.shape` or implicitly via
:json:schema:`ChunkLayout/Grid.aspect_ratio` and
:json:schema:`ChunkLayout/Grid.elements`.  A suitable chunk
shape is chosen automatically based on these constraints.  If :json:schema:`ChunkLayout/Grid.elements` is
not specified, the default is 1 million elements per chunk:

.. admonition:: Example of unconstrained chunk layout
   :class: example

   >>> ts.open({
   ...     'driver': 'n5',
   ...     'kvstore': {
   ...         'driver': 'memory'
   ...     }
   ... },
   ...         create=True,
   ...         dtype=ts.uint16,
   ...         shape=[1000, 2000, 3000]).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [2, 1, 0],
     'read_chunk': {'shape': [102, 102, 102]},
     'write_chunk': {'shape': [102, 102, 102]},
   })

.. admonition:: Example of explicit chunk shape constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'n5',
   ...     'kvstore': {
   ...         'driver': 'memory'
   ...     }
   ... },
   ...         create=True,
   ...         dtype=ts.uint16,
   ...         shape=[1000, 2000, 3000],
   ...         chunk_layout=ts.ChunkLayout(
   ...             chunk_shape=[100, 200, 300])).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [2, 1, 0],
     'read_chunk': {'shape': [100, 200, 300]},
     'write_chunk': {'shape': [100, 200, 300]},
   })

.. admonition:: Example of chunk aspect ratio constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'n5',
   ...     'kvstore': {
   ...         'driver': 'memory'
   ...     }
   ... },
   ...         create=True,
   ...         dtype=ts.uint16,
   ...         shape=[1000, 2000, 3000],
   ...         chunk_layout=ts.ChunkLayout(
   ...             chunk_aspect_ratio=[1, 2, 2])).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [2, 1, 0],
     'read_chunk': {'shape': [64, 128, 128]},
     'write_chunk': {'shape': [64, 128, 128]},
   })

.. admonition:: Example of chunk aspect ratio and elements constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'n5',
   ...     'kvstore': {
   ...         'driver': 'memory'
   ...     }
   ... },
   ...         create=True,
   ...         dtype=ts.uint16,
   ...         shape=[1000, 2000, 3000],
   ...         chunk_layout=ts.ChunkLayout(
   ...             chunk_aspect_ratio=[1, 2, 2],
   ...             chunk_elements=2000000)).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [2, 1, 0],
     'read_chunk': {'shape': [79, 159, 159]},
     'write_chunk': {'shape': [79, 159, 159]},
   })

Codec
~~~~~

Within the :json:schema:`Schema.codec`, the compression parameters are
represented in the same way as in the :json:schema:`~driver/n5.metadata`:

.. json:schema:: driver/n5/Codec

It is an error to specify any other :json:schema:`Codec.driver`.

Fill value
~~~~~~~~~~

The N5 metadata format does not support specifying a fill value.  TensorStore
always assumes a fill value of :json:`0`.

Dimension units
~~~~~~~~~~~~~~~

The :json:schema:`Schema.dimension_units` correspond to the
:json:schema:`~driver/n5.metadata.units` and
:json:schema:`~driver/n5.metadata.resolution` metadata properties.  The base
unit is used directly; it is not converted in any way.

The N5 format requires that dimension units are specified either for all
dimensions, or for no dimensions; it is not possible to specify dimension units
for some dimensions while leaving the dimension units of the remaining
dimensions unspecified.  When creating a new dataset, if dimension units are
specified for at least one dimension, any dimensions for which the unit is
unspecified are assigned a dimensionless unit of :json:`1`.

Limitations
-----------

Datasets with varlength chunks are not supported.
