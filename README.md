# TensorStore

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/tensorstore)](https://pypi.org/project/tensorstore)
[![Build](https://github.com/google/tensorstore/workflows/Build/badge.svg)](https://github.com/google/tensorstore/actions?query=workflow%3ABuild)
[![Docs](https://github.com/google/tensorstore/workflows/Docs/badge.svg)](https://google.github.io/tensorstore)


TensorStore is an open-source C++ and Python software library designed for
storage and manipulation of large multi-dimensional arrays that:

* Provides advanced, fully composable indexing operations and virtual views.

* Provides a uniform API for reading and writing multiple array formats, including
  [zarr](https://zarr.dev/) and [N5](https://github.com/saalfeldlab/n5).

* Natively supports multiple storage systems, such as local and network
  filesystems, [Google Cloud Storage](https://cloud.google.com/storage),
  HTTP servers, and in-memory storage.

* Offers an asynchronous API to enable high-throughput access even to
  high-latency remote storage.

* Supports read/writeback caching and transactions, with strong atomicity,
  isolation, consistency, and durability (ACID) guarantees.

* Supports safe, efficient access from multiple processes and machines via
  optimistic concurrency.

Documentation and installation instructions are at
<https://google.github.io/tensorstore>.


## Getting Started

To get started using the TensorStore Python API, you can install the tensorstore
PyPI package using:

```
pip install tensorstore
```

Refer to the [tutorials](https://google.github.io/tensorstore/python/tutorial.html)
and [API documentation](https://google.github.io/tensorstore/python/api/index.html),
or the announcement on the [Google Research Blog](https://ai.googleblog.com/2022/09/tensorstore-for-high-performance.html)
for more details.


This is not an officially supported Google product.

# License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
