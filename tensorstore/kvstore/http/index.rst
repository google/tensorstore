.. _http-kvstore-driver:

``http`` Key-Value Store driver
===============================

The ``http`` driver allows arbitrary HTTP servers to be used as read-only
key-value stores.  Keys directly correspond to HTTP paths.

.. json:schema:: kvstore/http

.. json:schema:: Context.http_request_concurrency

.. json:schema:: Context.http_request_retries

.. json:schema:: KvStoreUrl/http

Cache behavior
--------------

When used with an in-memory cache, the staleness of responses is bounded using
the `Cache-Control
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control>`__
request header the `Date
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date>`__ response
header.

If the server supports the `ETag
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag>`__ response
header, the `If-Match
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match>`__ and
`If-None-Match
<https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match>`__ request
headers are used to revalidate cached responses.

TLS CA certificates
-------------------

For ``https://`` URLs, TensorStore depends on the system certificate authority
(CA) store to secure connections.  In many cases it will work by default without
any additional configuration, but if you receive an error like:

.. code-block:: none

   CURL error[77] Problem with the SSL CA cert (path? access rights?):
   error setting certificate verify locations:
     CAfile: /etc/ssl/certs/ca-certificates.crt
     CApath: none

refer to the :ref:`HTTP request-related environment
variables<http_environment_variables>` section for information on how to specify
the path to the system certificate store at runtime.
