// Copyright 2024 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/batch.h"

// Other headers
#include <optional>
#include <utility>

#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/batch.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

Batch ValidateOptionalBatch(std::optional<Batch> batch) {
  if (!batch) return no_batch;
  if (!*batch) throw py::value_error("batch was already submitted");
  return std::move(*batch);
}

namespace {
using BatchCls = py::class_<Batch>;

auto MakeBatchClass(py::module m) {
  return BatchCls(m, "Batch", R"(

Batches are used to group together read operations for potentially improved
efficiency.

Operations associated with a batch will potentially be deferred until all
references to the batch are released.

The batch behavior of any particular operation ultimately depends on the
underlying driver implementation, but in many cases batching operations can
reduce the number of separate I/O requests performed.

Example usage as a context manager (recommended):

    >>> store = ts.open(
    ...     {
    ...         'driver': 'zarr3',
    ...         'kvstore': {
    ...             'driver': 'file',
    ...             'path': 'tmp/dataset/'
    ...         },
    ...     },
    ...     shape=[5, 6],
    ...     chunk_layout=ts.ChunkLayout(read_chunk_shape=[2, 3],
    ...                                 write_chunk_shape=[6, 6]),
    ...     dtype=ts.uint16,
    ...     create=True,
    ...     delete_existing=True).result()
    >>> store[...] = np.arange(5 * 6, dtype=np.uint16).reshape([5, 6])
    >>> with ts.Batch() as batch:
    ...     read_future1 = store[:3].read(batch=batch)
    ...     read_future2 = store[3:].read(batch=batch)
    >>> await read_future1
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17]], dtype=uint16)
    >>> await read_future2
    array([[18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29]], dtype=uint16)

.. warning::

   Any operation performed as part of a batch may be deferred until the batch is
   submitted.  Blocking on (or awaiting) the completion of such an operation
   while retaining a reference to the batch will likely lead to deadlock.

Equivalent example using explicit call to :py:meth:`.submit`:

    >>> batch = ts.Batch()
    >>> read_future1 = store[:3].read(batch=batch)
    >>> read_future2 = store[3:].read(batch=batch)
    >>> batch.submit()
    >>> await read_future1
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17]], dtype=uint16)
    >>> await read_future2
    array([[18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29]], dtype=uint16)

Equivalent example relying on implicit submit by the destructor when the last reference is released:

    >>> batch = ts.Batch()
    >>> read_future1 = store[:3].read(batch=batch)
    >>> read_future2 = store[3:].read(batch=batch)
    >>> del batch
    >>> await read_future1
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17]], dtype=uint16)
    >>> await read_future2
    array([[18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29]], dtype=uint16)

.. warning::

   Relying on this implicit submit behavior is not recommended and may result in
   the submit being delayed indefinitely, due to Python implicitly retaining a
   reference to the object, or due to a cyclic reference.

Group:
  Core

Constructors
============

Operations
==========

)");
}

void DefineBatchAttributes(BatchCls& cls) {
  using Self = Batch;
  cls.def(py::init([]() { return Batch::New(); }), R"(
Creates a new batch.
)");
  cls.def(
      "submit", [](Self& self) { self.Release(); },
      R"(
Submits the batch.

After calling this method, attempting to start any new operation will this batch
will result in an error.

Raises:
  ValueError: If :py:meth:`.submit` has already been called.

Group:
  Operations
)");
  cls.def("__enter__", [](const Self& self) { return &self; });
  cls.def("__exit__", [](Self& self, py::object exc_type, py::object exc_value,
                         py::object traceback) { self.Release(); });
}

void RegisterBatchBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeBatchClass(m)]() mutable { DefineBatchAttributes(cls); });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterBatchBindings, /*priority=*/-450);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
