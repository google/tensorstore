.. _key_value_store:

Key-Value Storage Layer
=======================

Most TensorStore drivers access the underlying storage for array data through a
*key-value store* layer that supports a variety of underlying storage systems,
such as the local filesystem and cloud storage systems, through different
`drivers<key-value-store-drivers>`.

.. json:schema:: KvStore

.. json:schema:: KvStoreUrl

.. _key-value-store-drivers:

Root drivers
------------

.. toctree::
   :maxdepth: 1

   file/index
   gcs/index
   http/index
   memory/index
   s3/index
   tsgrpc/index

Adapters
--------

.. toctree::
   :maxdepth: 1

   kvstack/index
   neuroglancer_uint64_sharded/index
   ocdbt/index
   zarr3_sharding_indexed/index
   zip/index

.. json:schema:: KvStoreAdapter
