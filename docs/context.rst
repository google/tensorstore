.. _context:

Context framework
=================

Configuration options for TensorStore drivers are specified using a *context*
framework, which allows resources such as cache pools, concurrent execution
pools, and authentication credentials to be specified using JSON in a way that
allows sharing of resources by multiple TensorStore drivers.

.. json:schema:: Context

.. json:schema:: ContextResource

.. json:schema:: Context.cache_pool

.. json:schema:: Context.data_copy_concurrency

.. json:schema:: Context.file_io_concurrency
