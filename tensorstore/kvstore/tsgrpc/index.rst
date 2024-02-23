.. _tsgrpc_kvstore-kvstore-driver:

``tsgrpc_kvstore`` Key-Value Store driver
=========================================

The ``tsgrpc_kvstore`` driver connects to a gRPC server which implements
tensorstore_grpc.kvstore.KvStoreService.

.. json:schema:: kvstore/tsgrpc_kvstore

.. json:schema:: Context.data_copy_concurrency

Limitations
-----------

.. note::
   This is an experimental driver and should be considered a work-in-progress.
   There are known limitations around retries, large values, and authenticated
   connections.
