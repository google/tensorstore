.. _neuroglancer-precomputed-driver:

``neuroglancer_precomputed`` Driver
===================================

The ``neuroglancer_precomputed`` driver provides access to `Neuroglancer Precomputed format <https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed>`_ volumes backed by any supported :ref:`key_value_store`.  It supports reading, writing, and creating new volumes.

Multiscale volumes are supported, but each scale must be opened individually.

.. json-schema:: schema.yml
   :title: JSON Schema

.. json-schema:: schema.yml#/definitions/sharding-spec

Example JSON specifications
---------------------------

.. code-block:: json
   :caption: Example: Opening the first (or only) scale of an existing multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume"
   }


.. code-block:: json
   :caption: Example: Opening an existing scale by index.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume",
     "scale_index": 1
   }


.. code-block:: json
   :caption: Example: Opening an existing scale by resolution.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume",
     "scale_metadata": {
       "resolution": [4, 4, 40]
     }
   }

.. code-block:: json
   :caption: Example: Opening an existing scale by key.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume",
     "scale_metadata": {
       "key": "4_4_40"
     }
   }


.. code-block:: json
   :caption: Example: Creating a new scale in an existing multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume",
     "scale_metadata": {
       "size": [40000, 50000, 10000],
       "encoding": "compressed_segmentation",
       "compressed_segmentation_block_size": [8, 8, 8],
       "chunk_size": [64, 64, 64],
       "resolution": [8, 8, 40]
     }
   }

.. code-block:: json
   :caption: Example: Creating a new multiscale volume.

   {
     "driver": "neuroglancer_precomputed",
     "kvstore": {"driver": "gcs", "bucket": "my-bucket"},
     "path": "path/to/volume",
     "multiscale_metadata": {
       "type": "segmentation",
       "data_type": "uint64",
       "num_channels": 1
     },
     "scale_metadata": {
       "size": [40000, 50000, 10000],
       "encoding": "compressed_segmentation",
       "compressed_segmentation_block_size": [8, 8, 8],
       "chunk_size": [64, 64, 64],
       "resolution": [8, 8, 40]
     }
   }

Limitations
-----------

Resizing is not supported.

.. warning:: Writing to volumes in the sharded format is supported,
   but because updating a shard requires rewriting it entirely, write
   operations may be very inefficient unless special care is taken:
   
   1. It is most efficient to group writes by shard.

   2. The temporary memory required to write a shard is 2 to 3 times
      the size of the shard.  It is therefore advised that the shards
      be kept as small as possible (while still avoiding an excess
      number of objects in the underlying
      :json-schema:`https://github.com/google/tensorstore/json-schema/key-value-store`).

