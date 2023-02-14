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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/homogeneous_tuple.h"
#include "python/tensorstore/index.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/kvstore.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/transaction.h"
#include "python/tensorstore/write_futures.h"
#include "tensorstore/array.h"
#include "tensorstore/cast.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {

WriteFutures IssueCopyOrWrite(
    const TensorStore<>& self,
    std::variant<PythonTensorStoreObject*, ArrayArgumentPlaceholder> source) {
  if (auto* store = std::get_if<PythonTensorStoreObject*>(&source)) {
    return tensorstore::Copy((**store).value, self);
  } else {
    auto& source_obj = std::get_if<ArrayArgumentPlaceholder>(&source)->value;
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

using spec_setters::SetAssumeMetadata;
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

   To perform transactional operations using a :py:obj:`TensorStore` that was
   previously opened without a transaction, use
   :py:obj:`TensorStore.with_transaction`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    return self.Set(
        internal::TransactionState::ToTransaction(std::move(value)));
  }
};

}  // namespace open_setters

constexpr auto ForwardOpenSetters = [](auto callback, auto... other_param) {
  WithSchemaKeywordArguments(
      callback, other_param..., open_setters::SetRead{},
      open_setters::SetWrite{}, open_setters::SetOpen{},
      open_setters::SetCreate{}, open_setters::SetDeleteExisting{},
      open_setters::SetAssumeMetadata{}, open_setters::SetContext{},
      open_setters::SetTransaction{}, spec_setters::SetKvstore{});
};

constexpr auto ForwardSpecRequestSetters = [](auto callback,
                                              auto... other_param) {
  callback(other_param..., spec_setters::SetOpen{}, spec_setters::SetCreate{},
           spec_setters::SetDeleteExisting{}, spec_setters::SetAssumeMetadata{},
           spec_setters::SetMinimalSpec{}, spec_setters::SetRetainContext{},
           spec_setters::SetUnbindContext{});
};

using TensorStoreCls = py::class_<PythonTensorStoreObject>;

TensorStoreCls MakeTensorStoreClass(py::module m) {
  auto cls = PythonTensorStoreObject::Define(R"(
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
  DisallowInstantiationFromPython(cls);
  m.attr("TensorStore") = cls;
  return cls;
}

void DefineTensorStoreAttributes(TensorStoreCls& cls) {
  using Self = PythonTensorStoreObject;
  cls.def_property_readonly(
      "rank", [](Self& self) { return self.value.rank(); },
      R"(Number of dimensions in the domain.

This is equivalent to :python:`self.domain.rank`.

Example:

    >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> dataset.rank
    2

Group:
  Accessors

)");

  cls.def_property_readonly(
      "ndim", [](Self& self) { return self.value.rank(); },
      R"(
Alias for :py:obj:`.rank`.

Example:

    >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
    >>> dataset.ndim
    2

Group:
  Accessors

)");

  cls.def_property_readonly(
      "domain", [](Self& self) -> IndexDomain<> { return self.value.domain(); },
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

  cls.def_property_readonly(
      "dtype", [](Self& self) { return self.value.dtype(); }, R"(
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

  ForwardSpecRequestSetters([&](auto... param_def) {
    std::string doc = R"(
Spec that may be used to re-open or re-create the TensorStore.

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
    >>> dataset.spec(minimal_spec=True)
    Spec({
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [[70], [80]],
        'input_inclusive_min': [0, 0],
      },
    })
    >>> dataset.spec(minimal_spec=True, unbind_context=True)
    Spec({
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
      'driver': 'zarr',
      'dtype': 'uint32',
      'kvstore': {'driver': 'memory'},
      'transform': {
        'input_exclusive_max': [[70], [80]],
        'input_inclusive_min': [0, 0],
      },
    })

If neither :py:param:`.retain_context` nor :py:param:`.unbind_context` is
specified, the returned :py:obj:`~tensorstore.Spec` does not include any context
resources, equivalent to specifying
:py:param:`tensorstore.Spec.update.strip_context`.

Args:

)";
    AppendKeywordArgumentDocs(doc, param_def...);

    doc += R"(

Group:
  Accessors

)";

    cls.def(
        "spec",
        [](Self& self, KeywordArgument<decltype(param_def)>... kwarg) -> Spec {
          SpecRequestOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          return ValueOrThrow(self.value.spec(std::move(options)));
        },
        doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
  });

  cls.def_property_readonly(
      "mode",
      [](Self& self) {
        std::string mode;
        if (!!(self.value.read_write_mode() & ReadWriteMode::read)) mode += "r";
        if (!!(self.value.read_write_mode() & ReadWriteMode::write))
          mode += "w";
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

  cls.def_property_readonly(
      "readable",
      [](Self& self) {
        return !!(self.value.read_write_mode() & ReadWriteMode::read);
      },
      R"(
Indicates if reading is supported.

Group:
  Accessors

)");

  cls.def_property_readonly(
      "writable",
      [](Self& self) {
        return !!(self.value.read_write_mode() & ReadWriteMode::write);
      },
      R"(
Indicates if writing is supported.

Group:
  Accessors

)");

  cls.def("__repr__", [](Self& self) -> std::string {
    return internal_python::PrettyPrintJsonAsPythonRepr(
        self.value.spec(tensorstore::unbind_context) |
            [](const auto& spec) { return spec.ToJson(); },
        "TensorStore(", ")");
  });

  cls.def(
      "read",
      [](Self& self, ContiguousLayoutOrder order)
          -> PythonFutureWrapper<SharedArray<void>> {
        return PythonFutureWrapper<SharedArray<void>>(
            tensorstore::Read<zero_origin>(self.value, {order}),
            self.reference_manager());
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

  cls.def(
      "write",
      [](Self& self,
         std::variant<PythonTensorStoreObject*, ArrayArgumentPlaceholder>
             source) {
        return PythonWriteFutures(
            IssueCopyOrWrite(self.value, std::move(source)),
            self.reference_manager());
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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

  cls.def(
      "resize",
      [](Self& self,
         std::optional<SequenceParameter<OptionallyImplicitIndex>>
             inclusive_min,
         std::optional<SequenceParameter<OptionallyImplicitIndex>>
             exclusive_max,
         bool resize_metadata_only, bool resize_tied_bounds, bool expand_only,
         bool shrink_only) {
        if (!inclusive_min) {
          inclusive_min =
              std::vector<OptionallyImplicitIndex>(self.value.rank());
        }
        if (!exclusive_max) {
          exclusive_max =
              std::vector<OptionallyImplicitIndex>(self.value.rank());
        }
        tensorstore::ResizeOptions options = {};
        if (resize_metadata_only) {
          options.mode = options.mode | tensorstore::resize_metadata_only;
        }
        if (resize_tied_bounds) {
          options.mode = options.mode | tensorstore::resize_tied_bounds;
        }
        if (expand_only) {
          options.mode = options.mode | tensorstore::expand_only;
        }
        if (shrink_only) {
          options.mode = options.mode | tensorstore::shrink_only;
        }
        return PythonFutureWrapper<TensorStore<>>(
            tensorstore::Resize(
                self.value,
                std::vector<Index>(inclusive_min.value().begin(),
                                   inclusive_min.value().end()),
                std::vector<Index>(exclusive_max.value().begin(),
                                   exclusive_max.value().end()),
                options),
            self.reference_manager());
      },
      R"(
Resizes the current domain, persistently modifying the stored representation.

Depending on the :py:param`resize_metadata_only`, if the bounds are shrunk,
existing elements outside of the new bounds may be deleted. If the bounds are
expanded, elements outside the existing bounds will initially contain either the
fill value, or existing out-of-bounds data remaining after a prior resize
operation.

Example:

    >>> dataset = await ts.open(
    ...     {
    ...         'driver': 'zarr',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     dtype=ts.uint32,
    ...     shape=[3, 3],
    ...     create=True)
    >>> await dataset.write(np.arange(9, dtype=np.uint32).reshape((3, 3)))
    >>> dataset = await dataset.resize(exclusive_max=(3, 2))
    >>> await dataset.read()
    array([[0, 1],
           [3, 4],
           [6, 7]], dtype=uint32)

Args:

  inclusive_min: Sequence of length :python:`self.rank()` specifying the new
    inclusive min bounds.  A bound of :python:`None` indicates no change.
  exclusive_max: Sequence of length :python:`self.rank()` specifying the new
    exclusive max bounds.  A bound of :python:`None` indicates no change.
  resize_metadata_only: Requests that, if applicable, the resize operation
    affect only the metadata but not delete data chunks that are outside of the
    new bounds.
  resize_tied_bounds: Requests that the resize be permitted even if other
    bounds tied to the specified bounds must also be resized.  This option
    should be used with caution.
  expand_only: Fail if any bounds would be reduced.
  shrink_only: Fail if any bounds would be increased.

Returns:

  Future that resolves to a copy of :python:`self` with the updated bounds, once
  the resize operation completes.

Group:
  I/O

)",
      py::arg("inclusive_min") = std::nullopt,
      py::arg("exclusive_max") = std::nullopt,
      py::arg("resize_metadata_only") = false,
      py::arg("resize_tied_bounds") = false, py::arg("expand_only") = false,
      py::arg("shrink_only") = false);

  cls.def(
      "__array__",
      [](Self& self, std::optional<py::dtype> dtype,
         std::optional<py::object> context) {
        return ValueOrThrow(internal_python::InterruptibleWait(
            tensorstore::Read<zero_origin>(self.value)));
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

  cls.def(
      "resolve",
      [](Self& self,
         bool fix_resizable_bounds) -> PythonFutureWrapper<TensorStore<>> {
        ResolveBoundsOptions options = {};
        if (fix_resizable_bounds) {
          options.mode = options.mode | tensorstore::fix_resizable_bounds;
        }
        return PythonFutureWrapper<TensorStore<>>(
            tensorstore::ResolveBounds(self.value, options),
            self.reference_manager());
      },
      R"(
Obtains updated bounds, subject to the cache policy.

Group:
  I/O
)",
      py::arg("fix_resizable_bounds") = false);

  cls.def(
      "astype",
      [](Self& self, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(self.value, target_dtype.value));
      },
      R"(
Returns a read/write view as the specified data type.

Example:

  >>> store = ts.array([1, 2, 3], dtype=ts.uint32)
  >>> store.astype(ts.string)
  TensorStore({
    'base': {'array': [1, 2, 3], 'driver': 'array', 'dtype': 'uint32'},
    'context': {'data_copy_concurrency': {}},
    'driver': 'cast',
    'dtype': 'string',
    'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
  })

Group:
  Data type
)",
      py::arg("dtype"));

  cls.def_property_readonly(
      "chunk_layout", [](Self& self) { return self.value.chunk_layout(); },
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

  cls.def_property_readonly(
      "codec",
      [](Self& self)
          -> std::optional<
              internal::IntrusivePtr<const internal::CodecDriverSpec>> {
        auto codec = ValueOrThrow(self.value.codec());
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

Group:
  Accessors
)");

  cls.def_property_readonly(
      "fill_value",
      [](Self& self) -> std::optional<SharedArray<const void>> {
        auto fill_value = ValueOrThrow(self.value.fill_value());
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

Group:
  Accessors
)");

  cls.def_property_readonly(
      "dimension_units",
      [](Self& self) -> HomogeneousTuple<std::optional<Unit>> {
        return internal_python::SpanToHomogeneousTuple<std::optional<Unit>>(
            ValueOrThrow(self.value.dimension_units()));
      },
      R"(
Physical units of each dimension of the domain.

The *physical unit* for a dimension is the physical quantity corresponding to a
single index increment along each dimension.

A value of :python:`None` indicates that the unit is unknown.  A dimension-less
quantity is indicated by a unit of :python:`ts.Unit(1, "")`.

Example:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory'
    ...         }
    ...     },
    ...     create=True,
    ...     shape=[100, 200],
    ...     dtype=ts.uint32,
    ...     dimension_units=['5nm', '8nm'])
    >>> store.dimension_units
    (Unit(5, "nm"), Unit(8, "nm"))

Group:
  Accessors
)");

  cls.def_property_readonly(
      "kvstore",
      [](Self& self) -> std::optional<KvStore> {
        auto kvstore = self.value.kvstore();
        if (!kvstore.valid()) return std::nullopt;
        return kvstore;
      },
      R"(
Associated key-value store used as the underlying storage.

Equal to :python:`None` if the driver does not use a key-value store.

Example:

    >>> store = await ts.open(
    ...     {
    ...         'driver': 'n5',
    ...         'kvstore': {
    ...             'driver': 'memory',
    ...             'path': 'abc/',
    ...         }
    ...     },
    ...     create=True,
    ...     shape=[100, 200],
    ...     dtype=ts.uint32,
    ... )
    >>> store.kvstore
    KvStore({'context': {'memory_key_value_store': {}}, 'driver': 'memory', 'path': 'abc/'})

If a :py:obj:`.transaction` is bound to this :py:obj:`.TensorStore`, the same
transaction will also be bound to the returned :py:obj:`KvStore`.

Group:
  Accessors
)");

  cls.def_property_readonly(
      "schema",
      [](Self& self) -> Schema { return ValueOrThrow(self.value.schema()); },
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

Group:
  Accessors

)");

  EnableGarbageCollectedObjectPicklingFromSerialization(
      cls, internal::TensorStoreNonNullSerializer<>{});

  cls.attr("__iter__") = py::none();

  cls.def_property_readonly(
      "transaction",
      [](Self& self) -> std::optional<internal::TransactionState::CommitPtr> {
        if (self.value.transaction() == no_transaction) return std::nullopt;
        return internal::TransactionState::ToCommitPtr(
            self.value.transaction());
      },
      R"(
Associated transaction used for read/write operations.

Group:
  Transactions

)");

  cls.def(
      "with_transaction",
      [](Self& self,
         std::optional<internal::TransactionState::CommitPtr> transaction) {
        if (!transaction) transaction.emplace();
        return ValueOrThrow(
            self.value |
            internal::TransactionState::ToTransaction(std::move(*transaction)));
      },
      R"(Returns a transaction-bound view of this TensorStore.

The returned view may be used to perform transactional read/write operations.

Group:
  Transactions

)",
      py::arg("transaction"));

  DefineIndexTransformOperations(
      &cls,
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
      [](Self& self) {
        return internal::TensorStoreAccess::handle(self.value).transform;
      },
      [](Self& self, IndexTransform<> new_transform) -> PythonTensorStore {
        auto handle = internal::TensorStoreAccess::handle(self.value);
        handle.transform = std::move(new_transform);
        return internal::TensorStoreAccess::Construct<TensorStore<>>(
            std::move(handle));
      },
      [](Self& self,
         std::variant<PythonTensorStoreObject*, ArrayArgumentPlaceholder>
             source) {
        ValueOrThrow(internal_python::InterruptibleWait(
            IssueCopyOrWrite(self.value, std::move(source)).commit_future));
      });
}

void DefineTensorStoreFunctions(py::module m) {
  m.def(
      "cast",
      [](PythonTensorStoreObject& store, DataTypeLike target_dtype) {
        return ValueOrThrow(tensorstore::Cast(store.value, target_dtype.value));
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
      'context': {'data_copy_concurrency': {}},
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
         internal_context::ContextImplPtr context) {
        if (!context) {
          context = internal_context::Access::impl(Context::Default());
        }
        SharedArray<void> converted_array;
        if (dtype.has_value()) {
          ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                         /*NoThrow=*/false, /*AllowCopy=*/true>(
              array.value, &converted_array, dtype ? dtype->value : DataType());
        } else {
          ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                         /*NoThrow=*/false, /*AllowCopy=*/false>(
              array.value, &converted_array);
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
    ...             'path': 'v1.2/segmentation/',
    ...         },
    ...     },
    ...     read=True)
    >>> store
    TensorStore({
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'gcs_request_concurrency': {},
        'gcs_request_retries': {},
        'gcs_user_project': {},
      },
      'driver': 'neuroglancer_precomputed',
      'dtype': 'uint64',
      'kvstore': {
        'bucket': 'neuroglancer-janelia-flyem-hemibrain',
        'driver': 'gcs',
        'path': 'v1.2/segmentation/',
      },
      'multiscale_metadata': {'num_channels': 1, 'type': 'segmentation'},
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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
      'context': {
        'cache_pool': {},
        'data_copy_concurrency': {},
        'memory_key_value_store': {},
      },
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

.. _python-open-assume-metadata:

Using :py:param:`.assume_metadata` for improved concurrent open efficiency
--------------------------------------------------------------------------

Normally, when opening or creating a chunked format like
:ref:`zarr<zarr-driver>`, TensorStore first attempts to read the existing
metadata (and confirms that it matches any specified constraints), or (if
creating is allowed) creates a new metadata file based on any specified
constraints.

When the same TensorStore stored on a distributed filesystem or cloud storage is
opened concurrently from many machines, the simultaneous requests to read and
write the metadata file by every machine can create contention and result in
high latency on some distributed filesystems.

The :py:param:`.assume_metadata` open mode allows redundant reading and writing
of the metadata file to be avoided, but requires careful use to avoid data
corruption.

.. admonition:: Example of skipping reading the metadata when opening an existing array
   :class: example

   >>> context = ts.Context()
   >>> # First create the array normally
   >>> store = await ts.open({
   ...     "driver": "zarr",
   ...     "kvstore": "memory://"
   ... },
   ...                       context=context,
   ...                       dtype=ts.float32,
   ...                       shape=[5],
   ...                       create=True)
   >>> # Note that the .zarray metadata has been written.
   >>> await store.kvstore.list()
   [b'.zarray']
   >>> await store.write([1, 2, 3, 4, 5])
   >>> spec = store.spec()
   >>> spec
   Spec({
     'driver': 'zarr',
     'dtype': 'float32',
     'kvstore': {'driver': 'memory'},
     'metadata': {
       'chunks': [5],
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
       'shape': [5],
       'zarr_format': 2,
     },
     'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
   })
   >>> # Re-open later without re-reading metadata
   >>> store2 = await ts.open(spec,
   ...                        context=context,
   ...                        open=True,
   ...                        assume_metadata=True)
   >>> # Read data using the unverified metadata from `spec`
   >>> await store2.read()

.. admonition:: Example of skipping writing the metadata when creating a new array
   :class: example

   >>> context = ts.Context()
   >>> spec = ts.Spec(json={"driver": "zarr", "kvstore": "memory://"})
   >>> spec.update(dtype=ts.float32, shape=[5])
   >>> # Open the array without writing the metadata.  If using a distributed
   >>> # filesystem, this can safely be executed on multiple machines concurrently,
   >>> # provided that the `spec` is identical and the metadata is either fully
   >>> # constrained, or exactly the same TensorStore version is used to ensure the
   >>> # same defaults are applied.
   >>> store = await ts.open(spec,
   ...                       context=context,
   ...                       open=True,
   ...                       create=True,
   ...                       assume_metadata=True)
   >>> await store.write([1, 2, 3, 4, 5])
   >>> # Note that the data chunk has been written but not the .zarray metadata
   >>> await store.kvstore.list()
   [b'0']
   >>> # From a single machine, actually write the metadata to ensure the array
   >>> # can be re-opened knowing the metadata.  This can be done in parallel with
   >>> # any other writing.
   >>> await ts.open(spec, context=context, open=True, create=True)
   >>> # Metadata has now been written.
   >>> await store.kvstore.list()
   [b'.zarray', b'0']

Group:
  Core
)";
    m.def(
        "open",
        [](SpecLike spec, KeywordArgument<decltype(param_def)>... kwarg)
            -> PythonFutureWrapper<TensorStore<>> {
          TransactionalOpenOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          return PythonFutureWrapper<TensorStore<>>(
              tensorstore::Open(std::move(spec.spec), std::move(options)),
              std::move(spec.reference_manager));
        },
        doc.c_str(), py::arg("spec"), py::kw_only(),
        MakeKeywordArgumentPyArg(param_def)...);
  });
}

void RegisterTensorStoreBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeTensorStoreClass(m), m]() mutable {
    DefineTensorStoreAttributes(cls);
    DefineTensorStoreFunctions(m);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterTensorStoreBindings, /*priority=*/-1000);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
