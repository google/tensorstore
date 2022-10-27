.. _gcs-kvstore-driver:

``gcs`` Key-Value Store driver
===============================

The ``gcs`` driver provides access to Google Cloud Storage.  Keys directly
correspond to paths within a Google Cloud Storage bucket.

`Conditional writes
<https://cloud.google.com/kvstore/docs/generations-preconditions>`_ are used to
safely allow concurrent access from multiple machines.

.. json:schema:: kvstore/gcs

.. json:schema:: Context.gcs_user_project

.. json:schema:: Context.gcs_request_concurrency

.. json:schema:: Context.gcs_request_retries

.. json:schema:: Context.experimental_gcs_rate_limiter

.. json:schema:: KvStoreUrl/gs

.. _gcs-authentication:

Authentication
--------------

To use the ``gcs`` driver, you can access buckets that allow public access
(i.e. access by ``allUsers``) without credentials.  In order to access
non-public buckets, you must specify service account credentials, which can be
done through one of the following methods:

1. Set the :envvar:`GOOGLE_APPLICATION_CREDENTIALS` environment variable to the
   local path to a `Google Cloud JSON credentials file
   <https://cloud.google.com/docs/authentication/getting-started>`_.

2. Set up `Google Cloud SDK application default credentials
   <https://cloud.google.com/sdk/gcloud/reference/auth/application-default/login>`_.
   `Install the Google Cloud SDK <https://cloud.google.com/sdk/docs>`_ and run:

   .. code-block:: shell

      gcloud auth application-default login

   This stores Google Cloud credentials in
   :file:`~/.config/gcloud/application_default_credentials.json` or
   :file:`$CLOUDSDK_CONFIG/application_default_credentials.json`.

   This is often the most convenient method to use on a development machine.

3. On Google Compute Engine (GCE), the default service account credentials are
   retrieved automatically from the metadata service if credentials are not
   otherwise specified.

TLS CA certificates
-------------------

TensorStore connects to the Google Cloud Storage API using HTTP and depends on
the system certificate authority (CA) store to secure connections.  In many
cases it will work by default without any additional configuration, but if you
receive an error like:

.. code-block:: none

   CURL error[77] Problem with the SSL CA cert (path? access rights?):
   error setting certificate verify locations:
     CAfile: /etc/ssl/certs/ca-certificates.crt
     CApath: none

refer to the :ref:`HTTP request-related environment
variables<http_environment_variables>` section for information on how to specify
the path to the system certificate store at runtime.

Testing
-------

To test the ``gcs`` driver with a fake Google Cloud Storage server, such as
`fake-gcs-server <https://github.com/fsouza/fake-gcs-server>`_, you can set the
:envvar:`TENSORSTORE_GCS_HTTP_URL` environment variable to
e.g. ``http://localhost:4443``.
