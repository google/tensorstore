.. _s3-kvstore-driver:

``s3`` Key-Value Store driver
===============================

The ``s3`` driver provides access to Amazon S3 and S3-compatible object stores.
Keys directly correspond to paths within an S3 bucket.

.. json:schema:: kvstore/s3

.. json:schema:: Context.s3_request_concurrency

.. json:schema:: Context.s3_request_retries

.. json:schema:: Context.experimental_s3_rate_limiter

.. json:schema:: Context.aws_credentials

.. json:schema:: KvStoreUrl/s3


.. _s3-authentication:

Authentication
--------------

To use the ``s3`` driver, you can access buckets that allow public access
without credentials.  Otherwise amazon credentials are required:

1. Credentials may be obtained from the environment. Set the
   :envvar:`AWS_ACCESS_KEY_ID` environment variable, optionally along with
   the :envvar:`AWS_SECRET_ACCESS_KEY` environment variable and the
   :envvar:`AWS_SESSION_TOKEN` environment variable as they would be
   used by the `aws cli <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>`_.

2. Credentials may be obtained from the default user credentials file, when
   found at :file:`~/.aws/credentials`, or the file specified by the
   environment variable :envvar:`AWS_SHARED_CREDENTIALS_FILE`, along with
   a profile from the schema, or as indicated by the :envvar:`AWS_PROFILE`
   environment variables.

3. Credentials may be retrieved from the EC2 Instance Metadata Service (IMDS)
   when it is available.

.. envvar:: AWS_ACCESS_KEY_ID

   Specifies an AWS access key associated with an IAM account.
   See <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>

.. envvar:: AWS_SECRET_ACCESS_KEY

   Specifies the secret key associated with the access key.
   This is essentially the "password" for the access key.
   See <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>

.. envvar:: AWS_SESSION_TOKEN

   Specifies the session token value that is required if you are using temporary
   security credentials that you retrieved directly from AWS STS operations.
   See <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>

.. envvar:: AWS_SHARED_CREDENTIALS_FILE

   Specifies the location of the file that the AWS CLI uses to store access keys.
   The default path is :file:`~/.aws/credentials`.
   See <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>

.. envvar:: AWS_PROFILE

  Specifies the name of the AWS CLI profile with the credentials and options to
  use. This can be the name of a profile stored in a credentials or config file,
  or the value ``default`` to use the default profile.
  
  If defined, this environment variable overrides the behavior of using the
  profile named ``[default]`` in the credentials file.
  See <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html>

.. envvar:: AWS_EC2_METADATA_SERVICE_ENDPOINT

  Overrides the default EC2 Instance Metadata Service (IMDS) endpoint of 
  ``http://169.254.169.254``. This must be a valid uri, and should respond to
  the AWS IMDS api endpoints.
  See <https://docs.aws.amazon.com/sdkref/latest/guide/feature-imds-credentials.html>

.. envvar:: TENSORSTORE_S3_REQUEST_CONCURRENCY

   Specifies the concurrency level used by the shared Context
   :json:schema:`Context.s3_request_concurrency` resource. Defaults to 32.

