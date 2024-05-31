.. _kvstack-kvstore-driver:

``kvstack`` Key-Value Store driver
==================================

The ``kvstack`` key value store composes multipple underlying kvstores into
a composite kvstore view, where each base kvstore is mapped to a key ranges,
prefix, or exact key files.

- If the key ranges of the :json:schema:`~kvstore/kvstack.layers` overlap, the
  last layer takes precedence.

.. json:schema:: kvstore/kvstack

