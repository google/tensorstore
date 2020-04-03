.. _gcs-kvstore-driver:

``gcs`` Key-Value Store driver
===============================

The ``gcs`` driver provides access to Google Cloud Storage.  Keys directly
correspond to paths within a Google Cloud Storage bucket.

`Conditional writes
<https://cloud.google.com/kvstore/docs/generations-preconditions>`_ are used to
safely allow concurrent access from multiple machines.

.. json-schema:: schema.yml

.. json-schema:: schema.yml#/definitions/gcs_user_project
                 
.. json-schema:: schema.yml#/definitions/gcs_request_concurrency

.. _gcs-authentication:

Authentication
--------------

To use the ``gcs`` driver, you must specify service account credentials, which
can be done through one of the following methods:

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
