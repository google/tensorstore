Environment variables
=====================

The following environment variables may be specified at runtime to configure the
behavior of TensorStore.

.. _http_environment_variables:

HTTP requests
-------------

TensorStore uses `libcurl <https://curl.haxx.se>`_ to make HTTP requests
(e.g. to the Google Cloud Storage API), and uses `TLS
<https://curl.haxx.se/docs/sslcerts.html>`_ to secure connections to remote
servers.

TLS CA certificates
^^^^^^^^^^^^^^^^^^^

On MS Windows and Mac OS, libcurl uses TLS facilities provided by the operating
system and no additional configuration is necessary.

On Linux and BSD, if TensorStore is built with a *bundled* version of libcurl,
as is the default, it expects to find the system certificate authority (CA)
bundle in PEM format at :file:`/etc/ssl/certs/ca-certificates.crt`, which is the
location used by most Linux distributions.  If the system CA bundle is available
at that path, no additional configuration is necessary.

If the system CA bundle is not available at that path, you can specify an
alternative certificate bundle path or certificate directory at *runtime* with
the :envvar:`TENSORSTORE_CA_BUNDLE` or :envvar:`TENSORSTORE_CA_PATH` environment
variables:

.. envvar:: TENSORSTORE_CA_BUNDLE

   Specifies the path to a local file containing one or more CA certificates
   concatenated into a single file in PEM format.  On many Linux distributions,
   the system certificate bundle is available at
   :file:`/etc/ssl/certs/ca-certificates.crt`.  Refer to the `libcurl
   documentation <https://curl.haxx.se/libcurl/c/CURLOPT_CAINFO.html>`__ for
   more details.

.. envvar:: TENSORSTORE_CA_PATH

   Specifies the path of a local directory containing one or more CA
   certificates in PEM format.  Each file in the directory must contain a single
   certificate, and the directory must be prepared using the OpenSSL
   :command:`c_rehash` command.  Refer to the `libcurl documentation
   <https://curl.haxx.se/libcurl/c/CURLOPT_CAPATH.html>`__ for more details.
   Note that this is not the most common format for the system CA certificate
   store.  In most cases, the system CA certificate store should instead be
   specified using :envvar:`TENSORSTORE_CA_BUNDLE`.

.. note::

   On Linux and BSD, TensoprStore may optionally be built to dynamically link to
   a system-provided version of libcurl by specifying
   :envvar:`TENSORSTORE_SYSTEM_LIBS=se_curl<TENSORSTORE_SYSTEM_LIBS>`.  In
   this case, the default CA bundle path of
   :file:`/etc/ssl/certs/ca-certificates.crt` does not apply; instead, the
   default depends on how the system-provided libcurl was built, and most likely
   no additional configuration will be necessary.

Proxy configuration
^^^^^^^^^^^^^^^^^^^

.. envvar:: all_proxy

   Specifies a proxy server to use for making any HTTP or HTTPS request.  Refer
   to the `libcurl documentation
   <https://curl.haxx.se/libcurl/c/CURLOPT_PROXY.html>`__ for more details.

.. envvar:: http_proxy

   Specifies a proxy server to use for making HTTP (not HTTPS) requests.  Takes
   precedence over :envvar:`all_proxy`.  Refer to the `libcurl documentation
   <https://curl.haxx.se/libcurl/c/CURLOPT_PROXY.html>`__ for more details.

.. envvar:: https_proxy

   Specifies a proxy server to use for making HTTPS requests.  Takes precedence
   over :envvar:`all_proxy`.  Refer to the `libcurl documentation
   <https://curl.haxx.se/libcurl/c/CURLOPT_PROXY.html>`__ for more details.

.. envvar:: no_proxy

   Specifies a comma-separated list of hostnames or ip addresses for which
   proxying is disabled.  Refer to the `libcurl documentation
   <https://curl.haxx.se/libcurl/c/CURLOPT_NOPROXY.html>`__ for more details.

Debugging
^^^^^^^^^

.. envvar:: TENSORSTORE_CURL_VERBOSE

   If set to any value, verbose debugging information will be printed to stderr
   for all HTTP requests.

.. envvar:: SSLKEYLOGFILE

   Specifies the path to a local file where information necessary to decrypt
   TensorStore's TLS traffic will be saved in a format compatible with
   Wireshark.  Refer to the `libcurl documentation
   <https://ec.haxx.se/usingcurl/usingcurl-tls/tls-sslkeylogfile>`__ for more
   details.

Google Cloud Credentials
------------------------

.. envvar:: GOOGLE_APPLICATION_CREDENTIALS

   Specifies the local path to a `Google Cloud JSON credentials file
   <https://cloud.google.com/docs/authentication/getting-started>`_.  Refer to
   the :ref:`Google Cloud Storage Authentication<gcs-authentication>` section
   for details.

Google Cloud Storage
--------------------

.. envvar:: TENSORSTORE_GCS_HTTP_URL

   Specifies to connect to an alternative server in place of
   ``https://storage.googleapis.com``.  Note that the normal Google oauth2
   credentials *are* included in requests, and therefore only trusted servers
   should be used.

.. envvar:: TENSORSTORE_GCS_REQUEST_CONCURRENCY

   Specifies the concurrency level used by the shared Context
   :json:schema:`Context.gcs_request_concurrency` resource. Defaults to 32.

