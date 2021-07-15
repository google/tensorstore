// Copyright 2020 The TensorStore Authors
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

#include "python/tensorstore/tensorstore_class.h"

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/transaction.h"
#include "python/tensorstore/write_futures.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/array.h"
#include "tensorstore/cast.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/json_pprint_python.h"
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

WriteFutures IssueCopyOrWrite(
    TensorStore<> self,
    std::variant<TensorStore<>, ArrayArgumentPlaceholder> source) {
  if (auto* store = std::get_if<TensorStore<>>(&source)) {
    return tensorstore::Copy(*store, self);
  } else {
    auto& source_obj = std::get_if<ArrayArgumentPlaceholder>(&source)->obj;
    SharedArray<const void> source_array;
    ConvertToArray<const void, dynamic_rank, /*nothrow=*/false>(
        source_obj, &source_array, self.dtype(), 0, self.rank());
    return tensorstore::Write(std::move(source_array), self);
  }
}

namespace open_setters {

struct SetRead : public spec_setters::SetModeBase<ReadWriteMode::read> {
  static constexpr const char* name = "read";
  static constexpr const char* doc = R"(
Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
)";
};

struct SetWrite : public spec_setters::SetModeBase<ReadWriteMode::write> {
  static constexpr const char* name = "write";
  static constexpr const char* doc = R"(
Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
)";
};

using spec_setters::SetCreate;
using spec_setters::SetDeleteExisting;
using spec_setters::SetOpen;

struct SetContext {
  using type = internal_context::ContextImplPtr;
  static constexpr const char* name = "context";
  static constexpr const char* doc = R"(

Shared resource context.  Defaults to a new (unshared) context with default
options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
such as cache pools, between multiple open TensorStores, you must specify a
context.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(WrapImpl(std::move(value)));
  }
};

struct SetTransaction {
  using type = internal::TransactionState::CommitPtr;
  static constexpr const char* name = "transaction";
  static constexpr const char* doc = R"(

Transaction to use for opening/creating, and for subsequent operations.  By
default, the open is non-transactional.

.. note::

   To perform transactional operations using a TensorStore that was previously
   opened without a transaction, use :py:obj:`TensorStore.with_transaction`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(
        internal::TransactionState::ToTransaction(std::move(value)));
  }
};

}  // namespace open_setters

constexpr auto ForwardOpenSetters = [](auto callback, auto... other_param) {
  using namespace open_setters;
  WithSchemaKeywordArguments(callback, other_param..., SetRead{}, SetWrite{},
                             SetOpen{}, SetCreate{}, SetDeleteExisting{},
                             SetContext{}, SetTransaction{});
};

}  // namespace

void RegisterTensorStoreBindings(pybind11::module m) {
  py::class_<TensorStore<>, std::shared_ptr<TensorStore<>>> cls_tensorstore(
      m, "TensorStore", R"(
Asynchronous multi-dimensional array handle.

Examples:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         },
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[1000, 20000],
    ...     create=True)
    >>> dataset
    TensorStore({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [1000, 1049],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [1000, 20000],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[1000], [20000]],
        'input_inclusive_min': [0, 0],
      },
    })
    >>> await dataset[5:10, 6:8].write(42)
    >>> await dataset[0:10, 0:10].read()
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0]], dtype=uint32)

Group:
  Core
)");
  cls_tensorstore.def_property_readonly(
      "rank", [](const TensorStore<>& self) { return self.rank(); },
      R"(Number of dimensions in the domain.

This is equivalent to :python:`self.domain.rank`.

Example:

    >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> dataset.rank
    2

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "ndim", [=](const TensorStore<>& self) { return self.rank(); },
      R"(
Alias for :py:obj:`.rank`.

Example:

    >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> dataset.ndim
    2

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "domain",
      [](const TensorStore<>& self) -> IndexDomain<> { return self.domain(); },
      R"(
Domain of the array.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     domain=ts.IndexDomain(shape=[70, 80], labels=['x', 'y']),
    ...     create=True)
    >>> dataset.domain
    { "x": [0, 70*), "y": [0, 80*) }

The bounds of the domain reflect any transformations that have been applied:

    >>> dataset[30:50].domain
    { "x": [30, 50), "y": [0, 80*) }

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "dtype", [](const TensorStore<>& self) { return self.dtype(); }, R"(
Data type of the array.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> dataset.dtype
    dtype("uint32")

Group:
  Data type

)");

  cls_tensorstore.def(
      "spec", [](const TensorStore<>& self) { return self.spec(); }, R"(
Spec that may be used to re-open or re-create the array.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> dataset.spec()
    Spec({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [70, 80],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [70, 80],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[70], [80]],
        'input_inclusive_min': [0, 0],
      },
    })

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "mode",
      [](const TensorStore<>& self) {
        std::string mode;
        if (!!(self.read_write_mode() & ReadWriteMode::read)) mode += "r";
        if (!!(self.read_write_mode() & ReadWriteMode::write)) mode += "w";
        return mode;
      },
      R"(
Read/write mode.

Returns:

  :python:`'r'`
    read only

  :python:`'w'`
    write only

  :python:`'rw'`
    read-write

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "readable",
      [](const TensorStore<>& self) {
        return !!(self.read_write_mode() & ReadWriteMode::read);
      },
      R"(
Indicates if reading is supported.

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "writable",
      [](const TensorStore<>& self) {
        return !!(self.read_write_mode() & ReadWriteMode::write);
      },
      R"(
Indicates if writing is supported.

Group:
  Accessors

)");

  cls_tensorstore.def("__repr__", [](const TensorStore<>& self) -> std::string {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self.spec() |
            [](const auto& spec) {
              return spec.ToJson(IncludeDefaults{false});
            },
        "TensorStore(", ")");
  });

  cls_tensorstore.def(
      "read",
      [](const TensorStore<>& self,
         std::optional<ContiguousLayoutOrder> order) {
        py::gil_scoped_release gil_release;
        return tensorstore::Read<zero_origin>(self, {order.value_or(c_order)});
      },
      R"(
Reads the data within the current domain.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> await dataset[5:10, 8:12].read()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=uint32)

.. tip::

   Depending on the cache behavior of the driver, the read may be satisfied by
   the cache and not require any I/O.

When *not* using a :py:obj:`.transaction`, the
read result only reflects committed data; the result never includes uncommitted
writes.

When using a transaction, the read result reflects all writes completed (but not
yet committed) to the transaction.

Args:
  order: Contiguous layout order of the returned array:

    :python:`'C'`
      Specifies C order, i.e. lexicographic/row-major order.

    :python:`'F'`
      Specifies Fortran order, i.e. colexicographic/column-major order.

Returns:
  A future representing the asynchronous read result.

.. tip::

   Synchronous reads (blocking the current thread) may be performed by calling
   :py:obj:`Future.result` on the returned future:

   >>> dataset[5:10, 8:12].read().result()
   array([[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]], dtype=uint32)

See also:

  - :py:obj:`.__array__`

Group:
  I/O

)",
      py::arg("order") = "C");

  cls_tensorstore.def(
      "write",
      [](const TensorStore<>& self,
         std::variant<TensorStore<>, ArrayArgumentPlaceholder> source) {
        return IssueCopyOrWrite(std::move(self), std::move(source));
      },
      R"(
Writes to the current domain.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> await dataset[5:10, 6:8].write(42)
    >>> await dataset[0:10, 0:10].read()
    array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0]], dtype=uint32)
    >>> await dataset[5:10, 6:8].write([1, 2])
    >>> await dataset[5:10, 6:8].read()
    array([[1, 2],
           [1, 2],
           [1, 2],
           [1, 2],
           [1, 2]], dtype=uint32)

Args:

  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self.domain` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

Returns:

  Future representing the asynchronous result of the write operation.

Logically there are two steps to the write operation:

1. reading/copying from the :python:`source`, and

2. waiting for the write to be committed, such that it will be reflected in
   subsequent reads.

The completion of these two steps can be tracked separately using the returned
:py:obj:`WriteFutures.copy` and :py:obj:`WriteFutures.commit` futures,
respectively:

Waiting on the returned `WriteFutures` object itself waits for
the entire write operation to complete, and is equivalent to waiting on the
:py:obj:`WriteFutures.commit` future.  The returned
:py:obj:`WriteFutures.copy` future becomes ready once the data has been fully
read from :python:`source`.  After this point, :python:`source` may be safely
modified without affecting the write operation.

.. warning::

   You must either synchronously or asynchronously wait on the returned future
   in order to ensure the write actually completes.  If all references to the
   future are dropped without waiting on it, the write may be cancelled.

Group:
  I/O

Non-transactional semantics
---------------------------

When *not* using a :py:obj:`Transaction`, the returned `WriteFutures.commit`
future becomes ready only once the data has been durably committed by the
underlying storage layer.  The precise durability guarantees depend on the
driver, but for example:

- when using the :ref:`file-kvstore-driver`, the data is only considered
  committed once the ``fsync`` system call completes, which should normally
  guarantee that it will survive a system crash;

- when using the :ref:`gcs-kvstore-driver`, the data is only considered
  committed once the write is acknowledged and durability is guaranteed by
  Google Cloud Storage.

Because committing a write often has significant latency, it is advantageous to
issue multiple writes concurrently and then wait on all of them jointly:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> await asyncio.wait([dataset[i * 5].write(i) for i in range(10)])

This can also be accomplished with synchronous blocking:

    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[70, 80],
    ...                   create=True).result()
    >>> futures = [dataset[i * 5].write(i) for i in range(10)]
    >>> for f in futures:
    ...     f.result()

Note:

  When issuing writes asynchronously, keep in mind that uncommitted writes are
  never reflected in non-transactional reads.

For most drivers, data is written in fixed-size
:ref:`write chunks<chunk-layout>` arranged in a regular grid.  When concurrently
issuing multiple writes that are not perfectly aligned to disjoint write chunks,
specifying a :json:schema:`Context.cache_pool` enables writeback caching, which
can improve efficiency by coalescing multiple writes to the same chunk.

Alternatively, for more explicit control over writeback behavior, you can use a
:py:obj:`Transaction`.

Transactional semantics
-----------------------

Transactions provide explicit control over writeback, and allow uncommitted
writes to be read:

    >>> txn = ts.Transaction()
    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> await dataset.with_transaction(txn)[5:10, 6:8].write([1, 2])
    >>> # Transactional read reflects uncommitted write
    >>> await dataset.with_transaction(txn)[5:10, 6:8].read()
    array([[1, 2],
           [1, 2],
           [1, 2],
           [1, 2],
           [1, 2]], dtype=uint32)
    >>> # Non-transactional read does not reflect uncommitted write
    >>> await dataset[5:10, 6:8].read()
    array([[0, 0],
           [0, 0],
           [0, 0],
           [0, 0],
           [0, 0]], dtype=uint32)
    >>> await txn.commit_async()
    >>> # Now, non-transactional read reflects committed write
    >>> await dataset[5:10, 6:8].read()
    array([[1, 2],
           [1, 2],
           [1, 2],
           [1, 2],
           [1, 2]], dtype=uint32)

.. warning::

   When using a :py:obj:`Transaction`, the returned `WriteFutures.commit` future
   does *not* indicate that the data is durably committed by the underlying
   storage layer.  Instead, it merely indicates that the write will be reflected
   in any subsequent reads *using the same transaction*.  The write is only
   durably committed once the *transaction* is committed successfully.

)",
      py::arg("source"));

  cls_tensorstore.def(
      "__array__",
      [](const TensorStore<>& self, std::optional<py::dtype> dtype,
         std::optional<py::object> context) {
        py::gil_scoped_release gil_release;
        return ValueOrThrow(tensorstore::Read<zero_origin>(self).result());
      },
      R"(
Automatic conversion to `numpy.ndarray` for interoperability with NumPy.

*Synchronously* reads from the current domain and returns the result as an array.
Equivalent to :python:`self.read().result()`.

Examples:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> dataset[10:20, 5:10] + np.array(5)
    array([[5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5],
           [5, 5, 5, 5, 5]], dtype=uint32)

.. warning::

   This reads the entire domain into memory and blocks the current thread while
   reading.  For large arrays, it may be better to partition the domain into
   blocks and process each block separately.

See also:

   - :py:obj:`.read`

Group:
  I/O

)",
      py::arg("dtype") = std::nullopt, py::arg("context") = std::nullopt);

  cls_tensorstore.def(
      "resolve",
      [](const TensorStore<>& self, bool fix_resizable_bounds) {
        py::gil_scoped_release gil_release;
        ResolveBoundsOptions options = {};
        if (fix_resizable_bounds) {
          options.mode = options.mode | tensorstore::fix_resizable_bounds;
        }
        return tensorstore::ResolveBounds(self, options);
      },
      R"(
Obtains updated bounds, subject to the cache policy.

Group:
  I/O
)",
      py::arg("fix_resizable_bounds") = false);

  cls_tensorstore.def(
      "astype",
      [](const TensorStore<>& self, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(self, target_dtype.value));
      },
      R"(
Returns a read/write view as the specified data type.

Example:

  >>> store = ts.array([1, 2, 3], dtype=ts.uint32)
  >>> store.astype(ts.string)
  TensorStore({
    'base': {'array': [1, 2, 3], 'driver': 'array', 'dtype': 'uint32'},
    'driver': 'cast',
    'dtype': 'string',
    'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
  })

Group:
  Data type
)",
      py::arg("dtype"));

  cls_tensorstore.def_property_readonly(
      "chunk_layout",
      [](const TensorStore<>& self) { return self.chunk_layout(); },
      R"(
:ref:`Chunk layout<chunk-layout>` of the TensorStore.

Example:

  >>> store = await ts.open(
  ...     {
  ...         'driver': 'zarr',
  ...         'kvstore': {
  ...             'driver': 'memory'
  ...         }
  ...     },
  ...     shape=[1000, 2000, 3000],
  ...     dtype=ts.float32,
  ...     create=True)
  >>> store.chunk_layout
  ChunkLayout({
    'grid_origin': [0, 0, 0],
    'inner_order': [0, 1, 2],
    'read_chunk': {'shape': [102, 102, 102]},
    'write_chunk': {'shape': [102, 102, 102]},
  })

Group:
  Accessors

)");

  cls_tensorstore.def_property_readonly(
      "codec",
      [](const TensorStore<>& self)
          -> std::optional<internal::IntrusivePtr<const CodecSpec>> {
        auto codec = ValueOrThrow(self.codec());
        if (!codec.valid()) return std::nullopt;
        return codec;
      },
      R"(
Data codec spec.

This may be used to create a new TensorStore with the same codec.

Equal to :py:obj:`None` if the codec is unknown or not applicable.

Example:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     create=True,
    ...     shape=[100],
    ...     dtype=ts.uint32)
    >>> store.codec
    CodecSpec({
      'compressor': {
        'blocksize': 0,
        'clevel': 5,
        'cname': 'lz4',
        'id': 'blosc',
        'shuffle': -1,
      },
      'driver': 'zarr',
      'filters': None,
    })

)");

  cls_tensorstore.def_property_readonly(
      "fill_value",
      [](const TensorStore<>& self) -> std::optional<SharedArray<const void>> {
        auto fill_value = ValueOrThrow(self.fill_value());
        if (!fill_value.valid()) return std::nullopt;
        return fill_value;
      },
      R"(
Fill value for positions not yet written.

Equal to :py:obj:`None` if the fill value is unknown or not applicable.

The fill value has data type equal to :python:`self.dtype` and a shape that is
:ref:`broadcast-compatible<index-domain-alignment>` with :python:`self.shape`.

Example:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     create=True,
    ...     shape=[100],
    ...     dtype=ts.uint32,
    ...     fill_value=42)
    >>> store.fill_value
    array(42, dtype=uint32)

)");

  cls_tensorstore.def_property_readonly(
      "schema",
      [](const TensorStore<>& self) -> Schema {
        return ValueOrThrow(self.schema());
      },
      R"(
:ref:`Schema<schema>` of this TensorStore.

This schema may be used to create a new TensorStore with the same schema, but
possibly using a different driver, storage location, etc.

Example:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     create=True,
    ...     shape=[100],
    ...     dtype=ts.uint32,
    ...     fill_value=42)
    >>> store.schema
    Schema({
      'chunk_layout': {
        'grid_origin': [0],
        'inner_order': [0],
        'read_chunk': {'shape': [100]},
        'write_chunk': {'shape': [100]},
      },
      'codec': {
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'driver': 'zarr',
        'filters': None,
      },
      'domain': {'exclusive_max': [[100]], 'inclusive_min': [0]},
      'dtype': 'uint32',
      'fill_value': 42,
      'rank': 1,
    })

.. note:

   Each access to this property results in a new copy of the schema.  Modifying
   that copying by calling `Schema.update` does not affect this TensorStore.

)");

  cls_tensorstore.def(py::pickle(
      [](const TensorStore<>& self) -> py::tuple {
        auto builder = internal::ContextSpecBuilder::Make();
        auto spec = ValueOrThrow(self.spec(builder));
        auto pickled_context =
            internal_python::PickleContextSpecBuilder(std::move(builder));
        auto json_spec = ValueOrThrow(spec.ToJson());
        return py::make_tuple(py::cast(json_spec), std::move(pickled_context),
                              static_cast<int>(self.read_write_mode()));
      },
      [](py::tuple t) -> tensorstore::TensorStore<> {
        auto json_spec = py::cast<::nlohmann::json>(t[0]);
        auto context =
            WrapImpl(internal_python::UnpickleContextSpecBuilder(t[1]));
        auto read_write_mode = static_cast<ReadWriteMode>(
            py::cast<int>(t[2]) & static_cast<int>(ReadWriteMode::read_write));
        if (read_write_mode == ReadWriteMode::dynamic) {
          throw py::value_error(
              "Invalid ReadWriteMode encountered unpickling TensorStore");
        }
        py::gil_scoped_release gil_release;
        return ValueOrThrow(tensorstore::Open(std::move(json_spec),
                                              std::move(context),
                                              read_write_mode)
                                .result());
      }));

  cls_tensorstore.attr("__iter__") = py::none();

  cls_tensorstore.def_property_readonly(
      "transaction",
      [](const TensorStore<>& self) {
        return internal::TransactionState::ToCommitPtr(self.transaction());
      },
      R"(
Associated transaction used for read/write operations.

Group:
  Transactions

)");

  cls_tensorstore.def(
      "with_transaction",
      [](const TensorStore<>& self,
         internal::TransactionState::CommitPtr transaction) {
        return ValueOrThrow(self | internal::TransactionState::ToTransaction(
                                       std::move(transaction)));
      },
      R"(Returns a transaction-bound view of this TensorStore.

The returned view may be used to perform transactional read/write operations.

Group:
  Transactions

)",
      py::arg("transaction"));

  DefineIndexTransformOperations(
      &cls_tensorstore,
      {
          /*numpy_indexing=*/{
              /*kDefault*/ {/*get*/ R"(
Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

This operation does not actually read any data; it merely returns a virtual view
that reflects the result of the indexing operation.  To read data, call
:py:obj:`.read` on the returned view.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> view = dataset[[5, 10, 20], 6:10]
    >>> view
    TensorStore({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [70, 80],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [70, 80],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [3, 10],
        'input_inclusive_min': [0, 6],
        'output': [{'index_array': [[5], [10], [20]]}, {'input_dimension': 1}],
      },
    })

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.oindex`
   - :py:obj:`TensorStore.vindex`

Group:
  Indexing

Overload:
  indices
)",
                            /*set*/ R"(
Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

This allows Python subscript assignment syntax to be used as a shorthand for
:python:`self[indices].write(source).result()`.

Example:

    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[70, 80],
    ...                   create=True).result()
    >>> dataset[5:10, 6:8] = [1, 2]
    >>> dataset[4:10, 5:9].read().result()
    array([[0, 0, 0, 0],
           [0, 1, 2, 0],
           [0, 1, 2, 0],
           [0, 1, 2, 0],
           [0, 1, 2, 0],
           [0, 1, 2, 0]], dtype=uint32)

Args:
  indices: NumPy-style indexing terms.
  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self[indices].domain` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

Transactional writes are also supported:

    >>> txn = ts.Transaction()
    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[70, 80],
    ...                   create=True).result()
    >>> dataset.with_transaction(txn)[5:10, 6:8] = [1, 2]
    >>> txn.commit_sync()

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

Overload:
  indices

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.write`
   - :py:obj:`TensorStore.oindex.__setitem__`
   - :py:obj:`TensorStore.vindex.__setitem__`

)"},
              /*kOindex*/
              {/*get*/ R"(
Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
integer or boolean array indexing terms are applied orthogonally:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> view = dataset.oindex[[5, 10, 20], [7, 8, 10]]
    >>> view
    TensorStore({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [70, 80],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [70, 80],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [3, 3],
        'input_inclusive_min': [0, 0],
        'output': [
          {'index_array': [[5], [10], [20]]},
          {'index_array': [[7, 8, 10]]},
        ],
      },
    })

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.__getitem__(indices)`
   - :py:obj:`TensorStore.vindex`

Group:
  Indexing

)",
               /*set*/ R"(
Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

This is similar to :py:obj:`.__setitem__(indices)`, but differs in that any integer or
boolean array indexing terms are applied orthogonally:

    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[70, 80],
    ...                   create=True).result()
    >>> dataset.oindex[[5, 6, 8], [2, 5]] = [1, 2]
    >>> dataset[5:10, 0:6].read().result()
    array([[0, 0, 1, 0, 0, 2],
           [0, 0, 1, 0, 0, 2],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 2],
           [0, 0, 0, 0, 0, 0]], dtype=uint32)

Args:
  indices: NumPy-style indexing terms.
  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self.oindex[indices].domain` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.write`
   - :py:obj:`TensorStore.__setitem__(indices)`
   - :py:obj:`TensorStore.vindex.__setitem__`

)"},
              /*kVindex*/
              {/*get*/ R"(
Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
:python:`indices` specifies any array indexing terms, the broadcasted array
dimensions are unconditionally added as the first dimensions of the result
domain:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[60, 70, 80],
    ...     create=True)
    >>> view = dataset.vindex[:, [5, 10, 20], [7, 8, 10]]
    >>> view
    TensorStore({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [60, 70, 80],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [60, 70, 80],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [3, [60]],
        'input_inclusive_min': [0, 0],
        'output': [
          {'input_dimension': 1},
          {'index_array': [[5], [10], [20]]},
          {'index_array': [[7], [8], [10]]},
        ],
      },
    })

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.__getitem__(indices)`
   - :py:obj:`TensorStore.oindex`

Group:
  Indexing

)",
               /*set*/ R"(
Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

This is similar to :py:obj:`.__setitem__(indices)`, but differs in that if
:python:`indices` specifies any array indexing terms, the broadcasted array
dimensions are unconditionally added as the first dimensions of the
domain to be aligned to :python:`source`:

    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[2, 70, 80],
    ...                   create=True).result()
    >>> dataset.vindex[:, [5, 6, 8], [2, 5, 6]] = [[1, 2], [3, 4], [5, 6]]
    >>> dataset[:, 5:10, 0:6].read().result()
    array([[[0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]],
    <BLANKLINE>
           [[0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]]], dtype=uint32)

Args:
  indices: NumPy-style indexing terms.
  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self.vindex[indices].domain` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

See also:

   - :ref:`python-numpy-style-indexing`
   - :py:obj:`TensorStore.write`
   - :py:obj:`TensorStore.__setitem__(indices)`
   - :py:obj:`TensorStore.oindex.__setitem__`

)"},
          },
          /*index_transform*/
          {/*get*/ R"(
Computes a virtual view using an explicit :ref:`index transform<index-transform>`.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[70, 80],
    ...     create=True)
    >>> transform = ts.IndexTransform(
    ...     input_shape=[3],
    ...     output=[
    ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
    ...         ts.OutputIndexMap(index_array=[5, 4, 3])
    ...     ])
    >>> dataset[transform]
    TensorStore({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [70, 80],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<u4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [70, 80],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [3],
        'input_inclusive_min': [0],
        'output': [{'index_array': [1, 2, 3]}, {'index_array': [5, 4, 3]}],
      },
    })
    >>> await dataset[transform].write([1, 2, 3])
    >>> await dataset[1:6, 1:6].read()
    array([[0, 0, 0, 0, 1],
           [0, 0, 0, 2, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint32)

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.

Returns:

  View of rank :python:`transform.input_rank` and domain
  :python:`self.domain[transform]`.

This is the most general form of indexing, to which all other indexing methods
reduce:

- :python:`self[expr]` is equivalent to
  :python:`self[ts.IndexTransform(self.domain)[expr]]`

- :python:`self.oindex[expr]` is equivalent to
  :python:`self[ts.IndexTransform(self.domain).oindex[expr]]`

- :python:`self.vindex[expr]` is equivalent to
  :python:`self[ts.IndexTransform(self.domain).vindex[expr]]`

In most cases it is more convenient to use one of those other indexing forms
instead.

Group:
  Indexing

Overload:
  transform

See also:

   - :ref:`index-transform`
   - :py:obj:`TensorStore.__getitem__(indices)`
   - :py:obj:`TensorStore.__getitem__(domain)`
   - :py:obj:`TensorStore.__getitem__(expr)`
   - :py:obj:`TensorStore.oindex`
   - :py:obj:`TensorStore.vindex`

)",
           /*set*/ R"(
Synchronously writes using an explicit :ref:`index transform<index-transform>`.

This allows Python subscript assignment syntax to be used as a shorthand for
:python:`self[transform].write(source).result()`.

Example:

    >>> dataset = ts.open({
    ...     'driver': 'zarr',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   shape=[70, 80],
    ...                   create=True).result()
    >>> transform = ts.IndexTransform(
    ...     input_shape=[3],
    ...     output=[
    ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
    ...         ts.OutputIndexMap(index_array=[5, 4, 3])
    ...     ])
    >>> dataset[transform] = [1, 2, 3]
    >>> dataset[1:6, 1:6].read().result()
    array([[0, 0, 0, 0, 1],
           [0, 0, 0, 2, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint32)

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.
  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self.domain[transform]` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

Overload:
  transform

See also:

   - :ref:`index-transform`
   - :py:obj:`TensorStore.write`
   - :py:obj:`TensorStore.__setitem__(indices)`
   - :py:obj:`TensorStore.__setitem__(domain)`
   - :py:obj:`TensorStore.__setitem__(expr)`
   - :py:obj:`TensorStore.vindex.__setitem__`
   - :py:obj:`TensorStore.oindex.__setitem__`

)"},
          /*index_domain*/
          {/*get*/ R"(
Computes a virtual view using an explicit :ref:`index domain<index-domain>`.

The domain of the resultant view is computed as in
:py:obj:`IndexDomain.__getitem__(domain)`.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     domain=ts.IndexDomain(shape=[60, 70, 80], labels=['x', 'y', 'z']),
    ...     create=True)
    >>> domain = ts.IndexDomain(labels=['x', 'z'],
    ...                         inclusive_min=[5, 6],
    ...                         exclusive_max=[8, 9])
    >>> dataset[domain]
    TensorStore({
      'driver': 'n5',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'axes': ['x', 'y', 'z'],
        'blockSize': [60, 70, 80],
        'compression': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'shuffle': 1,
          'type': 'blosc',
        },
        'dataType': 'uint32',
        'dimensions': [60, 70, 80],
      },
      'transform': {
        'input_exclusive_max': [8, [70], 9],
        'input_inclusive_min': [5, 0, 6],
        'input_labels': ['x', 'y', 'z'],
      },
    })

Args:

  domain: Index domain, must have dimension labels that can be
    :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

Returns:

  Virtual view with domain equal to :python:`self.domain[domain]`.

Group:
  Indexing

Overload:
  domain

See also:

   - :ref:`index-domain`
   - :ref:`index-domain-alignment`
   - :py:obj:`IndexTransform.__getitem__(domain)`

)",
           /*set*/ R"(
Synchronously writes using an explicit :ref:`index domain<index-domain>`.

Example:

    >>> dataset = ts.open({
    ...     'driver': 'n5',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   domain=ts.IndexDomain(shape=[60, 70, 80],
    ...                                         labels=['x', 'y', 'z']),
    ...                   create=True).result()
    >>> domain = ts.IndexDomain(labels=['x', 'z'],
    ...                         inclusive_min=[5, 6],
    ...                         exclusive_max=[8, 9])
    >>> dataset[domain] = 42
    >>> dataset[5:10, 0, 5:10].read().result()
    array([[ 0, 42, 42, 42,  0],
           [ 0, 42, 42, 42,  0],
           [ 0, 42, 42, 42,  0],
           [ 0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0]], dtype=uint32)

Args:

  transform: Index transform, :python:`transform.output_rank` must equal
    :python:`self.rank`.
  source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
    :python:`self.domain[transform]` and with a data type convertible to
    :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
    `array_like`, including a scalar.

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

Overload:
  domain

See also:

   - :ref:`index-domain`
   - :ref:`index-domain-alignment`
   - :py:obj:`TensorStore.write`

)"},
          /*dim_expression*/
          {/*get*/ R"(
Computes a virtual view using a :ref:`dimension expression<python-dim-expressions>`.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     domain=ts.IndexDomain(shape=[60, 70, 80], labels=['x', 'y', 'z']),
    ...     create=True)
    >>> dataset[ts.d['x', 'z'][5:10, 6:9]]
    TensorStore({
      'driver': 'n5',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'axes': ['x', 'y', 'z'],
        'blockSize': [60, 70, 80],
        'compression': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'shuffle': 1,
          'type': 'blosc',
        },
        'dataType': 'uint32',
        'dimensions': [60, 70, 80],
      },
      'transform': {
        'input_exclusive_max': [10, [70], 9],
        'input_inclusive_min': [5, 0, 6],
        'input_labels': ['x', 'y', 'z'],
      },
    })

Returns:

  Virtual view with the dimension expression applied.

Group:
  Indexing

Overload:
  expr

See also:

   - :ref:`python-dim-expressions`
   - :py:obj:`IndexTransform.__getitem__(expr)`

)",
           /*set*/ R"(
Synchronously writes using a :ref:`dimension expression<python-dim-expressions>`.

Example:

    >>> dataset = ts.open({
    ...     'driver': 'n5',
    ...     'kvstore': {
    ...         'driver': 'memory'
    ...     }
    ... },
    ...                   dtype=ts.uint32,
    ...                   domain=ts.IndexDomain(shape=[60, 70, 80],
    ...                                         labels=['x', 'y', 'z']),
    ...                   create=True).result()
    >>> dataset[ts.d['x', 'z'][5:10, 6:9]] = [1, 2, 3]
    >>> dataset[5:10, 0, 5:10].read().result()
    array([[0, 1, 2, 3, 0],
           [0, 1, 2, 3, 0],
           [0, 1, 2, 3, 0],
           [0, 1, 2, 3, 0],
           [0, 1, 2, 3, 0]], dtype=uint32)

.. warning::

   When *not* using a transaction, the subscript assignment syntax always blocks
   synchronously on the completion of the write operation.  When performing
   multiple, fine-grained writes, it is recommended to either use a transaction
   or use the asynchronous :py:obj:`TensorStore.write` interface directly.

Group:
  I/O

Overload:
  expr

See also:

   - :ref:`python-dim-expressions`
   - :py:obj:`TensorStore.write`
   - :py:obj:`TensorStore.__getitem__(expr)`

)"},
      },
      [](std::shared_ptr<TensorStore<>> self) {
        return internal::TensorStoreAccess::handle(*self).transform;
      },
      [](std::shared_ptr<TensorStore<>> self, IndexTransform<> new_transform) {
        auto handle = internal::TensorStoreAccess::handle(*self);
        handle.transform = std::move(new_transform);
        return internal::TensorStoreAccess::Construct<TensorStore<>>(
            std::move(handle));
      },
      [](TensorStore<> self,
         std::variant<TensorStore<>, ArrayArgumentPlaceholder> source) {
        ValueOrThrow(internal_python::InterruptibleWait(
            IssueCopyOrWrite(std::move(self), std::move(source))
                .commit_future));
      });

  m.def(
      "cast",
      [](const TensorStore<>& store, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(store, target_dtype.value));
      },
      R"(
Returns a read/write view as the specified data type.

Example:

    >>> array = ts.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=ts.float32)
    >>> view = ts.cast(array, ts.uint32)
    >>> view
    TensorStore({
      'base': {
        'array': [1.5, 2.5, 3.5, 4.5, 5.5],
        'driver': 'array',
        'dtype': 'float32',
      },
      'driver': 'cast',
      'dtype': 'uint32',
      'transform': {'input_exclusive_max': [5], 'input_inclusive_min': [0]},
    })
    >>> await view.read()
    array([1, 2, 3, 4, 5], dtype=uint32)

Group:
  Views
)",
      py::arg("store"), py::arg("dtype"));

  m.def(
      "array",
      [](ArrayArgumentPlaceholder array, std::optional<DataTypeLike> dtype,
         internal_context::ContextImplPtr context) -> TensorStore<> {
        if (!context) {
          context = internal_context::Access::impl(Context::Default());
        }
        SharedArray<void> converted_array;
        if (dtype.has_value()) {
          ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                         /*NoThrow=*/false, /*AllowCopy=*/true>(
              array.obj, &converted_array, dtype ? dtype->value : DataType());
        } else {
          ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                         /*NoThrow=*/false, /*AllowCopy=*/false>(
              array.obj, &converted_array);
        }
        return ValueOrThrow(FromArray(WrapImpl(std::move(context)),
                                      std::move(converted_array)));
      },
      R"(
Returns a TensorStore that reads/writes from an in-memory array.

Args:
  array: Source array.
  dtype: Data type to which :python:`array` will be converted.
  context: Context to use.

Group:
  Views
)",
      py::arg("array"), py::arg("dtype") = std::nullopt,
      py::arg("context") = nullptr);

  ForwardOpenSetters([&](auto... param_def) {
    std::string doc = R"(
Opens or creates a :py:class:`TensorStore` from a :py:class:`Spec`.

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     create=True,
    ...     dtype=ts.int32,
    ...     shape=[1000, 2000, 3000],
    ...     chunk_layout=ts.ChunkLayout(inner_order=[2, 1, 0]),
    ... )
    >>> store
    TensorStore({
      'driver': 'zarr',
      'dtype': 'int32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [102, 102, 102],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<i4',
        'fill_value': None,
        'filters': None,
        'order': 'F',
        'shape': [1000, 2000, 3000],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[1000], [2000], [3000]],
        'input_inclusive_min': [0, 0, 0],
      },
    })

Args:
  spec: TensorStore Spec to open.  May also be specified as :json:schema:`JSON<TensorStore>`.

)";
    AppendKeywordArgumentDocs(doc, param_def...);
    doc += R"(

Examples
========

Opening an existing TensorStore
-------------------------------

To open an existing TensorStore, you can use a *minimal* :py:class:`.Spec` that
specifies required driver-specific options, like the storage location.
Information that can be determined automatically from the existing metadata,
like the data type, domain, and chunk layout, may be omitted:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'neuroglancer_precomputed',
    ...         'kvstore': {
    ...             'driver': 'gcs',
    ...             'bucket': 'neuroglancer-janelia-flyem-hemibrain',
    ...         },
    ...         'path': 'v1.2/segmentation',
    ...     },
    ...     read=True)
    >>> store
    TensorStore({
      'driver': 'neuroglancer_precomputed',
      'dtype': 'uint64',
      'kvstore': {
        'bucket': 'neuroglancer-janelia-flyem-hemibrain',
        'driver': 'gcs',
      },
      'multiscale_metadata': {'num_channels': 1, 'type': 'segmentation'},
      'path': 'v1.2/segmentation',
      'scale_index': 0,
      'scale_metadata': {
        'chunk_size': [64, 64, 64],
        'compressed_segmentation_block_size': [8, 8, 8],
        'encoding': 'compressed_segmentation',
        'key': '8.0x8.0x8.0',
        'resolution': [8.0, 8.0, 8.0],
        'sharding': {
          '@type': 'neuroglancer_uint64_sharded_v1',
          'data_encoding': 'gzip',
          'hash': 'identity',
          'minishard_bits': 6,
          'minishard_index_encoding': 'gzip',
          'preshift_bits': 9,
          'shard_bits': 15,
        },
        'size': [34432, 39552, 41408],
        'voxel_offset': [0, 0, 0],
      },
      'transform': {
        'input_exclusive_max': [34432, 39552, 41408, 1],
        'input_inclusive_min': [0, 0, 0, 0],
        'input_labels': ['x', 'y', 'z', 'channel'],
      },
    })

Creating a new TensorStore
--------------------------

To create a new TensorStore, you must specify required driver-specific options,
like the storage location, as well as :py:class:`Schema` constraints like the
data type and domain.  Suitable defaults are chosen automatically for schema
properties that are left unconstrained:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         },
    ...     },
    ...     create=True,
    ...     dtype=ts.float32,
    ...     shape=[1000, 2000, 3000],
    ...     fill_value=42)
    >>> store
    TensorStore({
      'driver': 'zarr',
      'dtype': 'float32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [102, 102, 102],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<f4',
        'fill_value': 42.0,
        'filters': None,
        'order': 'C',
        'shape': [1000, 2000, 3000],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[1000], [2000], [3000]],
        'input_inclusive_min': [0, 0, 0],
      },
    })

Partial constraints may be specified on the chunk layout, and the driver will
determine a matching chunk layout automatically:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         },
    ...     },
    ...     create=True,
    ...     dtype=ts.float32,
    ...     shape=[1000, 2000, 3000],
    ...     chunk_layout=ts.ChunkLayout(
    ...         chunk_shape=[10, None, None],
    ...         chunk_aspect_ratio=[None, 2, 1],
    ...         chunk_elements=10000000,
    ...     ),
    ... )
    >>> store
    TensorStore({
      'driver': 'zarr',
      'dtype': 'float32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [10, 1414, 707],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '<f4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [1000, 2000, 3000],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[1000], [2000], [3000]],
        'input_inclusive_min': [0, 0, 0],
      },
    })

The schema constraints allow key storage characteristics to be specified
independent of the driver/format:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         },
    ...     },
    ...     create=True,
    ...     dtype=ts.float32,
    ...     shape=[1000, 2000, 3000],
    ...     chunk_layout=ts.ChunkLayout(
    ...         chunk_shape=[10, None, None],
    ...         chunk_aspect_ratio=[None, 2, 1],
    ...         chunk_elements=10000000,
    ...     ),
    ... )
    >>> store
    TensorStore({
      'driver': 'n5',
      'dtype': 'float32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'blockSize': [10, 1414, 707],
        'compression': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'shuffle': 1,
          'type': 'blosc',
        },
        'dataType': 'float32',
        'dimensions': [1000, 2000, 3000],
      },
      'transform': {
        'input_exclusive_max': [[1000], [2000], [3000]],
        'input_inclusive_min': [0, 0, 0],
      },
    })

Driver-specific constraints can be used in combination with, or instead of,
schema constraints:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         },
    ...         'metadata': {
    ...             'dtype': '>f4'
    ...         },
    ...     },
    ...     create=True,
    ...     shape=[1000, 2000, 3000])
    >>> store
    TensorStore({
      'driver': 'zarr',
      'dtype': 'float32',
      'kvstore': {'driver': 'memory'},
      'metadata': {
        'chunks': [102, 102, 102],
        'compressor': {
          'blocksize': 0,
          'clevel': 5,
          'cname': 'lz4',
          'id': 'blosc',
          'shuffle': -1,
        },
        'dimension_separator': '.',
        'dtype': '>f4',
        'fill_value': None,
        'filters': None,
        'order': 'C',
        'shape': [1000, 2000, 3000],
        'zarr_format': 2,
      },
      'transform': {
        'input_exclusive_max': [[1000], [2000], [3000]],
        'input_inclusive_min': [0, 0, 0],
      },
    })

Group:
  Core
)";
    m.def(
        "open",
        [](const SpecLike& spec,
           KeywordArgument<decltype(param_def)>... kwarg) {
          TransactionalOpenOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          return tensorstore::Open(std::move(spec.value), std::move(options));
        },
        doc.c_str(), py::arg("spec"), py::kw_only(),
        MakeKeywordArgumentPyArg(param_def)...);
  });
}

}  // namespace internal_python
}  // namespace tensorstore
