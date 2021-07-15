:hero: One-stop shop for high-performance array storage.

TensorStore
===========

.. toctree::
   :hidden:
   :caption: Python API

   python/tutorial
   python/indexing
   python/api/index

.. toctree::
   :hidden:
   :caption: Setup

   installation
   environment

.. toctree::
   :hidden:
   :caption: JSON

   spec
   index_space
   schema
   context

   driver/index
   kvstore/index

TensorStore is a library for efficiently reading and writing large multi-dimensional arrays.

Highlights
----------

- Provides a uniform API for reading and writing multiple array formats,
  including :doc:`zarr <driver/zarr/index>`, `N5 <driver/n5/index>`, and
  `Neuroglancer precomputed <driver/neuroglancer_precomputed/index>`.

- Natively supports multiple :doc:`storage drivers<kvstore/index>`, including
  `Google Cloud Storage <kvstore/gcs/index>`, `local and network filesystems
  <kvstore/file/index>`, `in-memory storage <kvstore/memory/index>`.
- Support for read/writeback caching and transactions, with strong atomicity,
  isolation, consistency, and durability (:wikipedia:`ACID <ACID>`) guarantees.
- Supports safe, efficient access from multiple processes and machines via optimistic concurrency.
- High-performance implementation in C++ automatically takes advantage of
  multiple cores for encoding/decoding and performs multiple concurrent I/O
  operations to saturate network bandwidth.
- Asynchronous API enables high-throughput access even to high-latency remote storage.
- Advanced, fully composable :doc:`indexing operations<index_space>` and virtual
  views.

Getting started
---------------

To get started using the Python API, start with the
:doc:`tutorial<python/tutorial>` and :doc:`indexing operation
guide<python/indexing>`, then refer to the detailed :doc:`Python API
reference<python/api/index>`.

For setup instructions, refer to the :doc:`installation` section.

For details for using a particular driver, refer to the
:doc:`driver<driver/index>` and :doc:`key-value storage<kvstore/index>`
reference.

Concepts
--------

The core abstraction, a **TensorStore** is an asynchronous view of a
multi-dimensional array.  Every TensorStore is backed by a
:doc:`driver<driver/index>`, which connects the high-level TensorStore interface
to an underlying data storage mechanism.

Opening or creating a TensorStore is done using a :doc:`JSON Spec<spec>`, which
is analogous to a URL/file path/database connection string.

TensorStore introduces a new indexing abstraction, the :ref:`index-transform`,
which underlies all indexing operations.  All indexing operations result in
virtual views and are fully composable.  Dimension labels are also supported,
and can be used in indexing operations through the `dimension
expression<python-dim-expressions>` mechanism.

Shared resources like in-memory caches and concurrency limits are configured
using the :doc:`context<context>` mechanism.

Properties of a TensorStore, like the domain, data type, chunk layout, fill
value, and encoding, can be queried and specified/constrained in a uniform way
using the :doc:`schema<schema>` mechanism.
