.. _schema:

Schema
======

The *schema* of a TensorStore specifies key properties of the format in a
uniform way that is independent of where and how the data is actually stored.
When creating a TensorStore, schema constraints and preferences may be
specified; the driver combines these constraints with any driver-specific
constraints/defaults to choose a suitable schema automatically.  When opening an
existing TensorStore, its schema is validated against any constraints that are
specified.

.. json:schema:: Schema

.. _chunk-layout:

Chunk layout
------------

For :ref:`chunked storage formats<chunked-drivers>`, the data storage layout can
be represented in a driver-independent way as a *chunk layout*.

A chunk layout specifies a hierarchical regular grid with up to three levels:

- The *write* level, the top-most level, specifies the grid to which writes
  should be aligned.  Writes of individual chunkss at this level may be
  performed without amplification.  For the :ref:`zarr-driver`, :ref:`n5-driver`
  and the :ref:`neuroglancer-precomputed-driver` using the unsharded format, the
  write level is also the only level; each write chunk corresponds to a single
  key in the underlying :ref:`key_value_store`.  For the
  :ref:`neuroglancer-precomputed-driver` using the sharded format, each write
  chunk corresponds to an entire shard.
- The *read* level evenly subdivides *write* chunks by an additional regular
  grid.  Reads of individual chunks at this level may be performed without
  amplification.  Every write chunk boundary must be aligned to a read chunk
  boundary.  If reads and writes may be performed at the same granularity, such
  as with the :ref:`zarr-driver`, :ref:`n5-driver`, and the
  :ref:`neuroglancer-precomputed-driver` using the unsharded format, there is no
  additional read grid; a read chunk is the same size as a write chunk.  For the
  :ref:`neuroglancer-precomputed-driver` using the sharded format, each read
  chunk corresponds to a base chunk as defined by the format.
- The *codec* level further subdivides the read level into codec chunks.  For
  formats that make use of it, the codec chunk shape may affect the compression
  rate.  For the :ref:`neuroglancer-precomputed-driver` when using the
  compressed segmentation encoding, the codec chunk shape specifies the
  compressed segmentation block shape.  The codec block shape does *not*
  necessarily evenly subdivide the read chunk shape.  (The precise offset of the
  codec chunk grid relative to the read chunk grid is not specified by the chunk
  layout.)

When creating a new TensorStore, constraints on the data storage layout can be
specified without specifying the precise layout explicitly.

.. json:schema:: ChunkLayout

.. json:schema:: ChunkLayout/Grid

.. _codec:

Codec
-----

.. json:schema:: Codec

Dimension units
---------------

.. json:schema:: Unit
