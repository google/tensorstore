.. _kvstore/gcs_grpc:

``gcs_grpc`` Key-Value Store driver
===================================

The ``gcs_grpc`` driver provides access to Google Cloud Storage using the
`Cloud Storage gRPC API
<https://cloud.google.com/storage/docs/grpc-overview>`_.  Keys directly
correspond to paths within a Google Cloud Storage bucket.

Compared to the :ref:`gcs<kvstore/gcs>` driver, which uses the JSON/HTTP API,
this driver uses gRPC, which generally offers lower latency and higher
throughput, and is required in order to write to ``RAPID`` storage class
buckets.

`Conditional writes
<https://cloud.google.com/storage/docs/generations-preconditions>`_ are used to
safely allow concurrent access from multiple machines.

.. json:schema:: kvstore/gcs_grpc

.. json:schema:: Context.gcs_user_project

.. json:schema:: Context.gcs_request_retries

.. json:schema:: Context.data_copy_concurrency

.. json:schema:: KvStoreUrl/gcs_grpc

.. _gcs_grpc-rapid-buckets:

RAPID storage class buckets
---------------------------

Google Cloud Storage ``RAPID`` storage class buckets (also known as *zonal*
buckets) only support *appendable* objects.  Because appendable objects become
immediately visible and readable before finalization, writing directly to the
destination key would violate single-key write atomicity.  To retain atomic
writes, TensorStore writes to a temporary object path within the same directory
and then atomically renames the temporary object to the final destination key
once the data is persisted and finalized.

.. warning::

   The :json:schema:`~kvstore/gcs_grpc.experimental_mode` option is experimental
   and may be modified or removed in a future release without backward
   compatibility guarantees.

Set :json:schema:`~kvstore/gcs_grpc.experimental_mode` to :json:`"zonal"` to
select this atomic zonal write path upfront:

.. code-block:: json

   {
     "driver": "gcs_grpc",
     "bucket": "my-rapid-bucket",
     "experimental_mode": "zonal"
   }

For zonal bucket workloads that do not require atomic writes or lack
``storage.objects.move`` and ``storage.objects.delete`` permissions, set
:json:schema:`~kvstore/gcs_grpc.experimental_mode` to :json:`"non_atomic"` to
write appendable objects directly to the destination key without a temporary
object or rename:

.. code-block:: json

   {
     "driver": "gcs_grpc",
     "bucket": "my-rapid-bucket",
     "experimental_mode": "non_atomic"
   }

If :json:schema:`~kvstore/gcs_grpc.experimental_mode` is set to
:json:`"standard"`, TensorStore forces the direct non-appendable write path as a
hard constraint without fallback; writes to buckets requiring appendable objects
will fail.

If :json:schema:`~kvstore/gcs_grpc.experimental_mode` is omitted, TensorStore
learns the bucket type automatically on the first write without requiring
bucket-level permissions: if a direct write fails because appendable objects
are required, the driver automatically switches to the atomic zonal write path
and remembers the mode for subsequent writes.  Likewise, if the direct write
succeeds, the driver remembers that the bucket is not zonal so that subsequent
writes do not re-probe.  An explicit mode setting is never overridden by
detection.

Behavior of the atomic zonal write path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- While a write is in progress, its temporary object (named
  ``<key>.__tmp_<hex>``) is visible to ``List`` and may be deleted by a
  concurrent ``DeleteRange`` covering its prefix.
- Cancelling a write does not interrupt a temporary upload or rename that is
  already in progress.  If the write is cancelled during the upload, the
  temporary object is deleted once the upload completes; a cancelled write
  whose rename has already started may still take effect.
- Uploads interrupted by a transient error are restarted from the beginning
  rather than resumed.
- ``RAPID`` buckets do not support `Requester Pays
  <https://cloud.google.com/storage/docs/requester-pays>`__; see
  :json:schema:`~kvstore/gcs_grpc.experimental_mode`.

Additional permissions
^^^^^^^^^^^^^^^^^^^^^^

The atomic zonal write path (:json:`"zonal"` or auto-detected) needs more than
the ``storage.objects.create`` permission required by the standard write path:

- ``storage.objects.move``, to rename the temporary object;
- ``storage.objects.delete``, both to overwrite an existing destination key and
  to remove a temporary object that was not renamed.

Workloads using :json:`"non_atomic"` write directly to the target object and
require only ``storage.objects.create``.  A credential scoped only to
``storage.objects.create`` and ``storage.objects.get`` works on regional
buckets and with :json:`"non_atomic"`, but fails on the atomic zonal write path.

Orphaned temporary objects
^^^^^^^^^^^^^^^^^^^^^^^^^^

If a process crashes between writing the temporary object and renaming it, or
if deleting a temporary object fails, the temporary object remains in the
bucket.  Because temporary objects use the dynamic naming pattern
``<key>.__tmp_<hex>``, Google Cloud Storage
`object lifecycle rules <https://cloud.google.com/storage/docs/lifecycle>`_
cannot match them using suffix filters.  Orphaned objects can be cleaned up
using ``DeleteRange`` across the enclosing range, out-of-band cleanup scripts,
or bucket-level TTL expiration rules on ephemeral buckets.

.. _gcs_grpc-authentication:

Authentication
--------------

See the :ref:`gcs driver authentication section<gcs-authentication>` for how to
configure Google Cloud credentials.

.. note::

   Unlike the HTTP-based :ref:`gcs<kvstore/gcs>` driver, which permits
   unauthenticated access to publicly readable buckets, the ``gcs_grpc``
   driver connecting to Google Cloud Storage endpoints requires valid Google
   Cloud credentials even for public buckets.  Insecure connections without
   credentials are only supported for local test endpoints (e.g. emulators).

Testing
-------

To test the ``gcs_grpc`` driver against a local fake server or test emulator,
such as `storage-testbench <https://github.com/googleapis/storage-testbench>`_,
specify the :json:schema:`~kvstore/gcs_grpc.endpoint` property:

.. code-block:: json

   {
     "driver": "gcs_grpc",
     "bucket": "my-test-bucket",
     "endpoint": "localhost:9000"
   }

When a non-Google endpoint (not ending in ``.googleapis.com``) is specified,
the driver automatically connects using insecure channel credentials.
