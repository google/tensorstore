#!/usr/bin/env python3
# Copyright 2021 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates a bazelrc file with the CI remote cache configuration.

The Google Cloud service account key used for writing to the remote cache is
specified in the environment variable `BAZEL_CACHE_SERVICE_ACCOUNT_KEY`.  Note
that this environment variable contains the actual JSON key, not a path to it.

If this environment variable is unspecified or the value is empty, a read-only
remote cache configuration is used.

Otherwise, the key is written to a temporary file, and the path to that
temporary file is recorded in the output bazelrc, and a read-write cache
configuration is used.
"""

import argparse
import os
import tempfile


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('cache_key', type=str)
  ap.add_argument('--bazelrc', type=str,
                  default=os.path.expanduser('~/.bazelrc'))
  args = ap.parse_args()

  cache_key = args.cache_key

  creds = os.getenv('BAZEL_CACHE_SERVICE_ACCOUNT_KEY')

  if creds:
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                     delete=False) as f:
      f.write(creds)
      creds_path = f.name
    print('Using writable remote cache')
  else:
    print('Using read-only remote cache')

  bucket = 'tensorstore-github-actions-bazel-cache'

  with open(args.bazelrc, 'w') as f:
    f.write('common --remote_cache=' +
            'https://storage.googleapis.com/{bucket}/{cache_key}\n'.format(
                bucket=bucket, cache_key=cache_key))
    f.write('common --remote_upload_local_results=%s\n' %
            ('true' if creds else 'false'))
    if creds:
      f.write('common --google_credentials=%s\n' %
              creds_path.replace('\\', '/'))


if __name__ == '__main__':
  main()
