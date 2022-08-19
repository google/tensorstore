.. _neuroglancer-uint64-sharded-kvstore-driver:

``neuroglancer_uint64_sharded`` Key-Value Store driver
======================================================

The ``neuroglancer_uint64_sharded`` driver implements support for the
`Neuroglancer Precomputed sharded format
<https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed/sharded.md>`_
on top of a base key-value store.

Within the key-value store interface, which uses strings as keys, the uint64
keys are encoded as 8-byte big endian values.

.. json:schema:: kvstore/neuroglancer_uint64_sharded

.. json:schema:: kvstore/neuroglancer_uint64_sharded/ShardingSpec

Example JSON specifications
---------------------------

.. code-block:: json
   :caption: Example: Opening with identity hash and 1GB cache

   {
     "driver": "neuroglancer_uint64_sharded",
     "kvstore": "gs://my-bucket/path/to/sharded/data/",
     "metadata": {
       "@type": "neuroglancer_uint64_sharded_v1",
       "hash": "identity",
       "preshift_bits": 1,
       "minishard_bits": 3,
       "shard_bits": 3,
       "data_encoding": "raw",
       "minishard_index_encoding": "gzip",
     },
     "context": {
       "cache_pool": {"total_bytes_limit": 1000000000}
     }
   }


.. code-block:: json
   :caption: Example: Opening with murmurhash3_x86_128 hash and 1GB cache

   {
     "driver": "neuroglancer_uint64_sharded",
     "kvstore": "gs://my-bucket/path/to/sharded/data/",
     "metadata": {
       "@type": "neuroglancer_uint64_sharded_v1",
       "hash": "murmurhash3_x86_128",
       "preshift_bits": 0,
       "minishard_bits": 3,
       "shard_bits": 3,
       "data_encoding": "raw",
       "minishard_index_encoding": "gzip",
     },
     "context": {
       "cache_pool": {"total_bytes_limit": 1000000000}
     }
   }

Limitations
-----------

It is strongly recommended to use a transaction when writing, and group writes
by shard (one transaction per shard).  Otherwise, there may be significant write
amplification due to repeatedly re-writing the entire shard.
