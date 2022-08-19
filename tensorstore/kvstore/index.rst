.. _key_value_store:

Key-Value Storage Layer
=======================

Most TensorStore drivers access the underlying storage for array data through a
*key-value store* layer that supports a variety of underlying storage systems,
such as the local filesystem and cloud storage systems, through different
`drivers<key-value-store-drivers>`.

.. _key-value-store-drivers:

Drivers
-------

.. toctree::
   :maxdepth: 1
   :glob:

   */index

.. json:schema:: KvStore

.. json:schema:: KvStoreUrl
