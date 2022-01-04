.. _memory-kvstore-driver:

``memory`` Key-Value Store driver
=================================

The ``memory`` driver stores key-value pairs in a transient, in-memory table.
It is useful for manipulating data in memory and for testing.  It includes full
support for multi-key transactions.

.. json:schema:: kvstore/memory

.. json:schema:: Context.memory_key_value_store

.. json:schema:: KvStoreUrl/memory

