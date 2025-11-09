# Copyright 2020 The TensorStore Authors
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
"""Downloads the intersphinx inventories referenced by `conf.py`.

This allows the documentation build to be hermetic.
"""

import os
import requests

if __name__ == '__main__':
  docs_dir = os.path.dirname(os.path.realpath(__file__))
  conf_path = os.path.join(docs_dir, 'conf.py')
  conf_module = {}
  with open(conf_path, 'r') as f:
    exec(f.read(), conf_module)
  intersphinx_mapping = conf_module['intersphinx_mapping']
  for _, (url, (local_path, _)) in intersphinx_mapping.items():
    if not url.endswith('/'):
      url += '/'
    full_url = url + 'objects.inv'
    full_path = os.path.join(docs_dir, local_path)
    resp = requests.get(full_url)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, 'wb') as f:
      f.write(resp.content)
    print('Fetched %s -> %s' % (full_url, full_path))
