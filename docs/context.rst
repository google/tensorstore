Context framework
=================

Configuration options for TensorStore drivers are specified using a *context*
framework, which allows resources such as cache pools, concurrent execution
pools, and authentication credentials to be specified using JSON in a way that
allows sharing of resources by multiple TensorStore drivers.

.. json-schema:: context_schema.yml
          
.. json-schema:: context_schema.yml#/definitions/resource

.. json-schema:: context_schema.yml#/definitions/cache_pool

.. json-schema:: context_schema.yml#/definitions/data_copy_concurrency

.. json-schema:: context_schema.yml#/definitions/file_io_concurrency
