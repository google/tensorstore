.. _zarr-sharding-indexed-kvstore-driver:

``zarr_sharding_indexed`` Key-Value Store driver
======================================================

The ``zarr_sharding_indexed`` driver implements support for stored
representation used by the Zarr v3 ``sharding_indexed`` codec on top of a base
key-value store.

For a grid of rank ``n``, keys must be ``n * 4`` bytes long, specifying the grid
cell indices, where ``0 <= grid_cell_indices[i] < grid_shape[i]``, as ``n``
consecutive ``uint32be`` values.

.. json:schema:: kvstore/zarr_sharding_indexed

Example JSON specifications
---------------------------

.. code-block:: json

   {
     "driver": "zarr_sharding_indexed",
     "kvstore": "gs://my-bucket/path/to/sharded/data",
     "grid_shape": [32, 128],
     "index_codecs" [
       {"name": "bytes", "configuration": {"endian": "little"}},
       {"name": "crc32c"}
     ],
     "context": {
       "cache_pool": {"total_bytes_limit": 1000000000}
     }
   }

Limitations
-----------

It is strongly recommended to use a transaction when writing, and group writes.
Otherwise, there may be significant write amplification due to repeatedly
re-writing the entire shard.
