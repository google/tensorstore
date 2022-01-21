.. _zarr-driver:

``zarr`` Driver
===============

`Zarr <https://github.com/zarr-developers/zarr-python>`_ is a chunked
array storage format based on the `NumPy data type model
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_.

The ``zarr`` driver provides access to Zarr-format arrays backed by
any supported :ref:`key_value_store`.  It supports reading, writing,
creating new arrays, and resizing arrays.

Zarr supports arrays with `structured data types
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_
specifying multiple named fields that are packed together.
TensorStore fully supports such arrays, but each field must be opened
separately.

.. json:schema:: driver/zarr

Compressors
-----------

Chunk data is encoded according to the
:json:schema:`driver/zarr.metadata.compressor` specified in the metadata.

.. json:schema:: driver/zarr/Compressor

The following compressors are supported:

.. json:schema:: driver/zarr/Compressor/zlib
.. json:schema:: driver/zarr/Compressor/blosc
.. json:schema:: driver/zarr/Compressor/bz2

Mapping to TensorStore Schema
-----------------------------

.. admonition:: Example with scalar data type
   :class: example

   For the following zarr :json:schema:`~driver/zarr.metadata`:

   .. doctest-output:: code-block json

      >>> def get_schema(metadata, field=''):
      ...     context = ts.Context()
      ...     kvstore = {'driver': 'memory'}
      ...     ts.open(
      ...         {
      ...             'driver': 'json',
      ...             'kvstore': kvstore,
      ...             'path': '.zarray'
      ...         },
      ...         context=context).result().write(metadata).result()
      ...     return ts.open(
      ...         {
      ...             'driver': 'zarr',
      ...             'kvstore': kvstore,
      ...             'field': field,
      ...         },
      ...         context=context).result().schema
      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "zarr_format": 2,
        "shape": [1000, 2000, 3000],
        "chunks": [100, 200, 300],
        "dtype": "<u2",
        "compressor": null,
        "fill_value": 42,
        "order": "C",
        "filters": null
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
        "codec": {"compressor": null, "driver": "zarr", "filters": null},
        "domain": {"exclusive_max": [[1000], [2000], [3000]], "inclusive_min": [0, 0, 0]},
        "dtype": "uint16",
        "fill_value": 42,
        "rank": 3
      }

.. admonition:: Example with `structured data type <https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_
   :class: example

   For the following zarr :json:schema:`~driver/zarr.metadata`:

   .. doctest-output:: code-block json

      >>> metadata = json.loads(OUTPUT)  # doctest:+JSON_OUTPUT
      ... metadata
      {
        "zarr_format": 2,
        "shape": [1000, 2000, 3000],
        "chunks": [100, 200, 300],
        "dtype": [["x", "<u2", [2, 3]], ["y", "<f4", [5]]],
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1},
        "fill_value": "AQACAAMABAAFAAYAAAAgQQAAMEEAAEBBAABQQQAAYEE=",
        "order": "F",
        "filters": null
      }

   the corresponding :json:schema:`Schema` for the :json:`"x"`
   :json:schema:`~driver/zarr.field` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata, field='x').to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [0, 0, 0, 0, 0],
          "inner_order": [2, 1, 0, 3, 4],
          "read_chunk": {"shape": [100, 200, 300, 2, 3]},
          "write_chunk": {"shape": [100, 200, 300, 2, 3]}
        },
        "codec": {
          "compressor": {"blocksize": 0, "clevel": 5, "cname": "lz4", "id": "blosc", "shuffle": 1},
          "driver": "zarr",
          "filters": null
        },
        "domain": {
          "exclusive_max": [[1000], [2000], [3000], 2, 3],
          "inclusive_min": [0, 0, 0, 0, 0]
        },
        "dtype": "uint16",
        "fill_value": [[1, 2, 3], [4, 5, 6]],
        "rank": 5
      }

   and the corresponding :json:schema:`Schema` for the :json:`"y"`
   :json:schema:`~driver/zarr.field` is:

   .. doctest-output:: code-block json

      >>> get_schema(metadata, field='y').to_json()  # doctest:+JSON_OUTPUT
      {
        "chunk_layout": {
          "grid_origin": [0, 0, 0, 0],
          "inner_order": [2, 1, 0, 3],
          "read_chunk": {"shape": [100, 200, 300, 5]},
          "write_chunk": {"shape": [100, 200, 300, 5]}
        },
        "codec": {
          "compressor": {"blocksize": 0, "clevel": 5, "cname": "lz4", "id": "blosc", "shuffle": 1},
          "driver": "zarr",
          "filters": null
        },
        "domain": {"exclusive_max": [[1000], [2000], [3000], 5], "inclusive_min": [0, 0, 0, 0]},
        "dtype": "float32",
        "fill_value": [10.0, 11.0, 12.0, 13.0, 14.0],
        "rank": 4
      }

Data type
~~~~~~~~~

Zarr scalar data types map to TensorStore data types as follows:

.. table:: Supported data types
   :class: table-column-align-center

   +----------------------------------+-------------------------------------+
   | TensorStore data type            | Zarr data type                      |
   |                                  +--------------------+----------------+
   |                                  | Little endian      | Big endian     |
   +==================================+====================+================+
   | :json:schema:`~dtype.bool`       | :json:`"|b1"`                       |
   +----------------------------------+-------------------------------------+
   | :json:schema:`~dtype.uint8`      | :json:`"|u1"`                       |
   +----------------------------------+-------------------------------------+
   | :json:schema:`~dtype.int8`       | :json:`"|i1"`                       |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.uint16`     | :json:`"<u2"`      | :json:`">u2"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.int16`      | :json:`"<i2"`      | :json:`">i2"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.uint32`     | :json:`"<u2"`      | :json:`">u2"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.int32`      | :json:`"<i4"`      | :json:`">i4"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.uint64`     | :json:`"<u8"`      | :json:`">u8"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.int64`      | :json:`"<i8"`      | :json:`">i8"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.float16`    | :json:`"<f2"`      | :json:`">f2"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.bfloat16`   | :json:`"bfloat16"` |                |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.float32`    | :json:`"<f4"`      | :json:`">f4"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.float64`    | :json:`"<f8"`      | :json:`">f8"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.complex64`  | :json:`"<c8"`      | :json:`">c8"`  |
   +----------------------------------+--------------------+----------------+
   | :json:schema:`~dtype.complex128` | :json:`"<c16"`     | :json:`">c16"` |
   +----------------------------------+--------------------+----------------+

Zarr `structured data types
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_ are
supported, but are represented in TensorStore as scalar arrays with additional
dimensions.

When creating a new array, if a :json:schema:`driver/zarr.metadata.dtype` is not
specified explicitly, a scalar Zarr data type with the native endianness is
chosen based on the :json:schema:`Schema.dtype`.  To create an array with
non-native endianness or a `structured data type
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_, the
zarr :json:schema:`driver/zarr.metadata.dtype` must be specified explicitly.

.. note::

   TensorStore supports the non-standard `bfloat16
   <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ data type as
   an extension.  On little endian platforms, the official `Zarr Python library
   <https://github.com/zarr-developers/zarr-python>`_ is capable of reading
   arrays created with the ``bfloat16`` data type provided that a bfloat16 numpy
   data type has been registered.  The TensorStore Python library registers such
   a data type, as does TensorFlow and JAX.

.. warning::

   zarr datetime/timedelta data types are not currently supported.

Domain
~~~~~~

The :json:schema:`Schema.domain` includes both the chunked dimensions as well as
any *subarray* dimensions in the case of a `structured data type
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_.

.. admonition:: Example with scalar data type
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`"<u2"` and the
   :json:schema:`driver/zarr.metadata.shape` is :json:`[100, 200]`, then the
   :json:schema:`Schema.domain` is :json:`{"shape": [[100], [200]]}`.

.. admonition:: Example with structured data type
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`[["x", "<u2", [2,
   3]]]`, and the :json:schema:`driver/zarr.metadata.shape` is :json:`[100,
   200]`, then the :json:schema:`Schema.domain` is :json:`{"shape": [[100],
   [200], 2, 3]}`.

As zarr does not natively support a non-zero origin, the underlying domain
always has a zero origin (:json:schema:`IndexDomain.inclusive_min` is all zero),
but it may be translated by the :json:schema:`~TensorStore.transform`.

The upper bounds of the chunked dimensions are resizable
(i.e. `implicit<implicit-bounds>`), while the upper bounds of any subarray
dimensions are not resizable.

The zarr metadata format does not support persisting `dimension
labels<dimension-labels>`, but dimension labels may still be specified when
opening using a :json:schema:`~TensorStore.transform`.

Chunk layout
~~~~~~~~~~~~

The zarr format supports a single :json:schema:`driver/zarr.metadata.chunks`
property that corresponds to the :json:schema:`ChunkLayout/Grid.shape`
constraint.  As with the :json:schema:`Schema.domain`, the
:json:schema:`Schema.chunk_layout` includes both the chunked dimensions as well
as any subarray dimensions in the case of a `structured data type
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_.  The
chunk size for subarray dimensions is always the full extent.

.. admonition:: Example with scalar data type
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`"<u2"` and
   :json:schema:`driver/zarr.metadata.chunks` is :json:`[100, 200]`, then the
   :json:schema:`ChunkLayout/Grid.shape` is :json:`[100, 200]`.

.. admonition:: Example with structured data type
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`[["x", "<u2", [2,
   3]]]`, and :json:schema:`driver/zarr.metadata.chunks` is :json:`[100, 200]`,
   then the :json:schema:`ChunkLayout/Grid.shape` is :json:`[100, 200, 2, 3]`.

As the zarr format supports only a single level of chunking, the
:json:schema:`ChunkLayout.read_chunk` and :json:schema:`ChunkLayout.write_chunk`
constraints are combined, and hard constraints on
:json:schema:`ChunkLayout.codec_chunk` must not be specified.

The :json:schema:`ChunkLayout.grid_origin` is always all-zero.

The :json:schema:`ChunkLayout.inner_order` corresponds to
:json:schema:`driver/zarr.metadata.order`, but also includes the subarray
dimensions, which are always the inner-most dimensions.

.. admonition:: Example with scalar data type and C order
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`"<u2"`,
   :json:schema:`driver/zarr.metadata.order` is :json:`"C"`, and there are 3
   chunked dimensions, then the :json:schema:`ChunkLayout.inner_order` is
   :json:`[0, 1, 2]`.

.. admonition:: Example with scalar data type and Fortran order
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`"<u2"`,
   :json:schema:`driver/zarr.metadata.order` is :json:`"F"`, and there are 3
   chunked dimensions, then the :json:schema:`ChunkLayout.inner_order` is
   :json:`[2, 1, 0]`.

.. admonition:: Example with structured data type and C order
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`[["x", "<u2", [2,
   3]]]`, :json:schema:`driver/zarr.metadata.order` is :json:`"C"`, and there
   are 3 chunked dimensions, then the :json:schema:`ChunkLayout.inner_order` is
   :json:`[0, 1, 2, 3, 4]`.

.. admonition:: Example with structured data type and Fortran order
   :class: example

   If the :json:schema:`driver/zarr.metadata.dtype` is :json:`[["x", "<u2", [2,
   3]]]`, :json:schema:`driver/zarr.metadata.order` is :json:`"F"`, and there
   are 3 chunked dimensions, then the :json:schema:`ChunkLayout.inner_order` is
   :json:`[2, 1, 0, 3, 4]`.

Selection of chunk layout when creating a new array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a new array, the chunk sizes may be specified explicitly via
:json:schema:`ChunkLayout/Grid.shape` or implicitly via
:json:schema:`ChunkLayout/Grid.aspect_ratio` and
:json:schema:`ChunkLayout/Grid.elements`.  In the latter case, a suitable chunk
shape is chosen automatically.  If :json:schema:`ChunkLayout/Grid.elements` is
not specified, the default is 1 million elements per chunk:

.. admonition:: Example of unconstrained chunk layout
   :class: example

   >>> ts.open({
   ...     'driver': 'zarr',
   ...     'kvstore': {
   ...         'driver': 'memory'
   ...     }
   ... },
   ...         create=True,
   ...         dtype=ts.uint16,
   ...         shape=[1000, 2000, 3000]).result().chunk_layout
   ChunkLayout({
     'grid_origin': [0, 0, 0],
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [102, 102, 102]},
     'write_chunk': {'shape': [102, 102, 102]},
   })

.. admonition:: Example of explicit chunk shape constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'zarr',
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
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [100, 200, 300]},
     'write_chunk': {'shape': [100, 200, 300]},
   })

.. admonition:: Example of chunk aspect ratio constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'zarr',
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
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [64, 128, 128]},
     'write_chunk': {'shape': [64, 128, 128]},
   })

.. admonition:: Example of chunk aspect ratio and elements constraint
   :class: example

   >>> ts.open({
   ...     'driver': 'zarr',
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
     'inner_order': [0, 1, 2],
     'read_chunk': {'shape': [79, 159, 159]},
     'write_chunk': {'shape': [79, 159, 159]},
   })

Codec
~~~~~

Within the :json:schema:`Schema.codec`, the compression parameters are
represented in the same way as in the :json:schema:`~driver/zarr.metadata`:

.. json:schema:: driver/zarr/Codec

It is an error to specify any other :json:schema:`Codec.driver`.

Fill value
~~~~~~~~~~

For scalar zarr data types, the :json:schema:`Schema.fill_value` must be a
scalar (rank 0).  For `structured data types
<https://zarr.readthedocs.io/en/stable/spec/v2.html#data-type-encoding>`_, the
:json:schema:`Schema.fill_value` must be broadcastable to the subarray shape.

As an optimization, chunks that are entirely equal to the fill value are not
stored.

The zarr format allows the fill value to be unspecified, indicated by a
:json:schema:`driver/zarr.metadata.fill_value` of :json:`null`.  In that case,
TensorStore always uses a fill value of :json:`0`.  However, in this case
explicitly-written all-zero chunks are still stored.

Limitations
-----------

Filters are not supported.
