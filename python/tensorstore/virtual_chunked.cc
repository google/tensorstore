// Copyright 2021 The TensorStore Authors
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
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "absl/base/optimization.h"
#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"  // IWYU pragma: keep
#include "python/tensorstore/spec.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/tensorstore_class.h"
#include "python/tensorstore/time.h"
#include "python/tensorstore/transaction.h"  // IWYU pragma: keep
#include "python/tensorstore/type_name_override.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/virtual_chunked.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {
using VirtualChunkedReadParametersCls =
    py::class_<virtual_chunked::ReadParameters>;

auto MakeVirtualChunkedReadParametersClass(py::module m) {
  return VirtualChunkedReadParametersCls(m, "VirtualChunkedReadParameters",
                                         R"(
Options passed to read callbacks used with :py:obj:`.virtual_chunked`.

Group:
  Virtual views
)");
}

void DefineVirtualChunkedReadParametersAttributes(
    VirtualChunkedReadParametersCls& cls) {
  using Self = virtual_chunked::ReadParameters;
  cls.def_property_readonly(
      "if_not_equal",
      [](const Self& self) { return py::bytes(self.if_not_equal_.value); }, R"(
Cached generation, read request can be skipped if no newer data is available.
)");

  cls.def_property_readonly(
      "staleness_bound",
      [](const Self& self) {
        return internal_python::ToPythonTimestamp(self.staleness_bound());
      },
      R"(
Read may be fulfilled with cached data no older than the specified bound.
)");
}

using VirtualChunkedWriteParametersCls =
    py::class_<virtual_chunked::WriteParameters>;

auto MakeVirtualChunkedWriteParametersClass(py::module m) {
  return VirtualChunkedWriteParametersCls(m, "VirtualChunkedWriteParameters",
                                          R"(
Options passed to write callbacks used with :py:obj:`.virtual_chunked`.

Group:
  Virtual views
)");
}

void DefineVirtualChunkedWriteParametersAttributes(
    VirtualChunkedWriteParametersCls& cls) {
  using Self = virtual_chunked::WriteParameters;
  cls.def_property_readonly(
      "if_equal",
      [](const Self& self) { return py::bytes(self.if_equal_.value); }, R"(
If non-empty, writeback should be conditioned on the existing data matching the specified generation.
)");
}

namespace virtual_chunked_setters {

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

constexpr auto ForwardVirtualChunkedSetters = [](auto callback,
                                                 auto... other_param) {
  callback(other_param..., schema_setters::SetRank{},
           schema_setters::SetDtype{}, schema_setters::SetDomain{},
           schema_setters::SetShape{}, schema_setters::SetChunkLayout{},
           schema_setters::SetDimensionUnits{}, schema_setters::SetSchema{},
           SetContext{}, SetTransaction{});
};

}  // namespace virtual_chunked_setters

TimestampedStorageGeneration NormalizeOptionalTimestampedStorageGeneration(
    const std::optional<TimestampedStorageGeneration>& stamp) {
  if (stamp) return *stamp;
  return TimestampedStorageGeneration(
      StorageGeneration::FromValues(std::string_view()),
      absl::InfiniteFuture());
}

/// Adapts a Python `read_function` or `write_function` into a serializable
/// `ReadFunction` or `WriteFunction` as expected by the C++
/// `tensorstore::VirtualChunked` interface.
///
/// This is a base class of `ReadFunctionAdapter` and `WriteFunctionAdapter`,
/// which must also define an `id` member for use by `SerializableFunction`.
///
/// \tparam Read `true` indicates a `read_function`, `false` indicates a
///     `write_function`.
template <bool Read>
struct FunctionAdapterBase {
  using Parameters = std::conditional_t<Read, virtual_chunked::ReadParameters,
                                        virtual_chunked::WriteParameters>;
  using Element = std::conditional_t<Read, void, const void>;

  struct State {
    SerializableAbstractEventLoop loop;
    PythonWeakRef python_function;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.loop, x.python_function);
    };
  };

  GilSafeHolder<State> state;
  IndexDomain<> orig_domain;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.state, x.orig_domain);
  };

  Future<TimestampedStorageGeneration> operator()(
      Array<Element, dynamic_rank, offset_origin> offset_array,
      Parameters params) const {
    ExitSafeGilScopedAcquire gil;
    if (!gil.acquired()) return PythonExitingError();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto zero_origin_array,
        (ArrayOriginCast<zero_origin, container>(offset_array)));
    IndexDomainBuilder domain_builder(offset_array.rank());
    domain_builder.bounds(offset_array.domain());
    // Check `orig_domain.rank()` in case it was deserialized from a corrupt
    // representation.
    if (orig_domain.valid() && orig_domain.rank() == offset_array.rank()) {
      domain_builder.labels(orig_domain.labels());
    }
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain, domain_builder.Finalize());
    py::object future_like;
    py::object python_array;
    if (CallAndSetErrorIndicator([&] {
          python_array = GetNumpyArray(UnownedToShared(zero_origin_array));
          auto python_domain = py::cast(domain);
          auto python_params = py::cast(params);
          future_like =
              py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
                  py::reinterpret_borrow<py::object>(
                      state->python_function.get_value_or_throw())
                      .ptr(),
                  python_domain.ptr(), python_array.ptr(), python_params.ptr(),
                  nullptr));
        })) {
      return GetStatusFromPythonException();
    }
    auto optional_timestamp_future = internal_python::ConvertToFuture<
        std::optional<TimestampedStorageGeneration>>(
        future_like, state->loop.obj.get_value_or_none());
    if (!Read || internal_python::CanDataTypeShareMemoryWithNumpy(
                     zero_origin_array.dtype())) {
      // The `python_function` does not modify `python_array`, or directly
      // modifies the content of `zero_origin_array`.
      return MapFutureValue(
          InlineExecutor{},
          [](std::optional<TimestampedStorageGeneration> stamp) {
            return NormalizeOptionalTimestampedStorageGeneration(stamp);
          },
          std::move(optional_timestamp_future));
    }
    if constexpr (Read) {
      // Need to copy `python_array` content back to
      // `zero_origin_array`.
      return MapFutureValue(
          params.executor(),
          [python_array = GilSafeHolder<py::object>(std::move(python_array)),
           zero_origin_array = std::move(zero_origin_array)](
              std::optional<TimestampedStorageGeneration> stamp)
              -> Result<TimestampedStorageGeneration> {
            ExitSafeGilScopedAcquire gil;
            if (!gil.acquired()) return PythonExitingError();
            if (CallAndSetErrorIndicator([&] {
                  CopyFromNumpyArray(*python_array, zero_origin_array);
                })) {
              return GetStatusFromPythonException();
            }
            return NormalizeOptionalTimestampedStorageGeneration(stamp);
          },
          std::move(optional_timestamp_future));
    }
    ABSL_UNREACHABLE();  // COV_NF_LINE
  }
};

struct ReadFunctionAdapter : public FunctionAdapterBase<true> {
  // Use an id that starts with a number to ensure it does not conflict with any
  // user-defined `SerializableFunction` id.
  constexpr static const char id[] = "0python:tensorstore.virtual_chunked.read";
};

struct WriteFunctionAdapter : public FunctionAdapterBase<false> {
  constexpr static const char id[] =
      "0python:tensorstore.virtual_chunked.write";
};

void RegisterVirtualChunkedBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeVirtualChunkedReadParametersClass(m)]() mutable {
    DefineVirtualChunkedReadParametersAttributes(cls);
  });
  defer([cls = MakeVirtualChunkedWriteParametersClass(m)]() mutable {
    DefineVirtualChunkedWriteParametersAttributes(cls);
  });

  defer([m]() mutable {
    virtual_chunked_setters::ForwardVirtualChunkedSetters(
        [&](auto... param_def) {
          std::string doc = R"(
Creates a :py:obj:`.TensorStore` where the content is read/written chunk-wise by an arbitrary function.

Example (read-only):

    >>> a = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32)
    >>> async def do_read(domain: ts.IndexDomain, array: np.ndarray,
    ...                   read_params: ts.VirtualChunkedReadParameters):
    ...     print(f'Computing content for: {domain}')
    ...     array[...] = (await a[domain].read()) + 100
    >>> t = ts.virtual_chunked(do_read, dtype=a.dtype, domain=a.domain)
    >>> await t.read()
    Computing content for: { [0, 2), [0, 3) }
    array([[101, 102, 103],
           [104, 105, 106]], dtype=uint32)

Example (read/write):

    >>> array = np.zeros(shape=[4, 5], dtype=np.uint32)
    >>> array[1] = 50
    >>> def do_read(domain, chunk, read_context):
    ...     chunk[...] = array[domain.index_exp]
    >>> def do_write(domain, chunk, write_context):
    ...     array[domain.index_exp] = chunk
    >>> t = ts.virtual_chunked(
    ...     do_read,
    ...     do_write,
    ...     dtype=array.dtype,
    ...     shape=array.shape,
    ...     chunk_layout=ts.ChunkLayout(read_chunk_shape=(2, 3)))
    >>> await t.read()
    array([[ 0,  0,  0,  0,  0],
           [50, 50, 50, 50, 50],
           [ 0,  0,  0,  0,  0],
           [ 0,  0,  0,  0,  0]], dtype=uint32)
    >>> t[1:3, 1:3] = 42
    >>> array
    array([[ 0,  0,  0,  0,  0],
           [50, 42, 42, 50, 50],
           [ 0, 42, 42,  0,  0],
           [ 0,  0,  0,  0,  0]], dtype=uint32)

Args:

  read_function: Callback that handles chunk read requests.  Must be specified
    to create a virtual view that supports reads.  To create a write-only view,
    leave this unspecified (as :py:obj:`None`).

    This function should assign to the array the content for the specified
    :py:obj:`~tensorstore.IndexDomain`.

    The returned :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration`
    identifies the version of the content, for caching purposes.  If versioning
    is not applicable, :py:obj:`None` may be returned to indicate a value that
    may be cached indefinitely.

    If it returns a :ref:`coroutine<python:async>`, the coroutine will be
    executed using the event loop indicated by :py:param:`.loop`.

  write_function: Callback that handles chunk write requests.  Must be specified
    to create a virtual view that supports writes.  To create a read-only view,
    leave this unspecified (as :py:obj:`None`).

    This function store the content of the array for the specified
    :py:obj:`~tensorstore.IndexDomain`.

    The returned :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration`
    identifies the stored version of the content, for caching purposes.  If
    versioning is not applicable, :py:obj:`None` may be returned to indicate a
    value that may be cached indefinitely.

    If it returns a :ref:`coroutine<python:async>`, the coroutine will be
    executed using the event loop indicated by :py:param:`.loop`.

  loop: Event loop on which to execute :py:param:`.read_function` and/or
    :py:param:`.write_function` if they are
    :ref:`async functions<python:async def>`.  If not specified (or
    :py:obj:`None` is specified), defaults to the loop returned by
    :py:obj:`asyncio.get_running_loop` (in the context of the call to
    :py:obj:`.virtual_chunked`).  If :py:param:`.loop` is not specified and
    there is no running event loop, it is an error for
    :py:param:`.read_function` or :py:param:`.write_function` to return a
    coroutine.

)";
          AppendKeywordArgumentDocs(doc, param_def...);
          doc += R"(


Warning:

  Neither :py:param:`.read_function` nor :py:param:`.write_function` should
  block synchronously while waiting for another TensorStore operation; blocking
  on another operation that uses the same
  :json:schema:`Context.data_copy_concurrency` resource may result in deadlock.
  Instead, it is better to specify a :ref:`coroutine function<python:async def>`
  for :py:param:`.read_function` and :py:param:`.write_function` and use
  :ref:`await<python:await>` to wait for the result of other TensorStore
  operations.

Group:
  Virtual views

Caching
-------

By default, the computed content of chunks is not cached, and will be
recomputed on every read.  To enable caching:

- Specify a :py:obj:`~tensorstore.Context` that contains a
  :json:schema:`~Context.cache_pool` with a non-zero size limit, e.g.:
  :json:`{"cache_pool": {"total_bytes_limit": 100000000}}` for 100MB.

- Additionally, if the data is not immutable, the :py:param:`read_function`
  should return a unique generation and a timestamp that is not
  :python:`float('inf')`.  When a cached chunk is re-read, the
  :py:param:`read_function` will be called with
  :py:obj:`~tensorstore.VirtualChunkedReadParameters.if_not_equal` specified.
  If the generation specified by
  :py:obj:`~tensorstore.VirtualChunkedReadParameters.if_not_equal` is still
  current, the :py:param:`read_function` may leave the output array unmodified
  and return a :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration` with
  an appropriate
  :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration.time` but
  :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration.generation` left
  unspecified.

Pickle support
--------------

The returned :py:obj:`.TensorStore` supports pickling if, and only if, the
:py:param:`.read_function` and :py:param:`.write_function` support pickling.

.. note::

   The :py:mod:`pickle` module only supports global functions defined in named
   modules.  For broader function support, you may wish to use
   `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__.

.. warning::

   The specified :py:param:`.loop` is not preserved when the returned
   :py:obj:`.TensorStore` is pickled, since it is a property of the current
   thread.  Instead, when unpickled, the resultant :py:obj:`.TensorStore` will
   use the running event loop (as returned by
   :py:obj:`asyncio.get_running_loop`) of the thread used for unpickling, if
   there is one.

Transaction support
-------------------

Transactional reads and writes are supported on virtual_chunked views.  A
transactional write simply serves to buffer the write in memory until it is
committed.  Transactional reads will observe prior writes made using the same
transaction.  However, when the transaction commit is initiated, the
:py:param:`.write_function` is called in exactly the same way as for a
non-transactional write, and if more than one chunk is affected, the commit will
be non-atomic.  If the transaction is atomic, it is an error to write to more
than one chunk in the same transaction.

You are also free to use transactional operations, e.g. operations on a
:py:class:`.KvStore` or another :py:class:`.TensorStore`, within the
:py:param:`.read_function` or :py:param:`.write_function`.

- For read-write views, you should not attempt to use the same transaction
  within the :py:param:`.read_function` or :py:param:`.write_function` that is
  also used for read or write operations on the virtual view directly, because
  both :py:param:`.write_function` and :py:param:`.read_function` may be called
  after the commit starts, and any attempt to perform new operations using the
  same transaction once it is already being committed will fail; instead, any
  transactional operations performed within the :py:param:`.read_function` or
  :py:param:`.write_function` should use a different transaction.

- For read-only views, it is possible to use the same transaction within the
  :py:param:`.read_function` as is also used for read operations on the virtual
  view directly, though this may not be particularly useful.

Specifying a transaction directly when creating the virtual chunked view is no
different than binding the transaction to an existing virtual chunked view.

)";

          using VirtualChunkedReadFunction =
              Callable<FutureLike<std::optional<TimestampedStorageGeneration>>,
                       IndexDomain<>, SharedArray<void>,
                       virtual_chunked::ReadParameters>;

          using VirtualChunkedWriteFunction =
              Callable<FutureLike<std::optional<TimestampedStorageGeneration>>,
                       IndexDomain<>, SharedArray<const void>,
                       virtual_chunked::WriteParameters>;

          m.def(
              "virtual_chunked",
              [](std::optional<VirtualChunkedReadFunction> read_function,
                 std::optional<VirtualChunkedWriteFunction> write_function,
                 std::optional<AbstractEventLoopParameter> loop,
                 KeywordArgument<decltype(param_def)>... kwarg)
                  -> PythonTensorStore {
                virtual_chunked::OpenOptions options;
                ApplyKeywordArguments<decltype(param_def)...>(options,
                                                              kwarg...);
                if (!loop) {
                  loop.emplace().value =
                      internal_python::GetCurrentThreadAsyncioEventLoop();
                }
                const auto get_read_function = [&] {
                  ReadFunctionAdapter read_function_adapter;
                  read_function_adapter.orig_domain = options.domain();
                  read_function_adapter.state->python_function =
                      py::reinterpret_borrow<py::object>(read_function->value);
                  read_function_adapter.state->loop.obj = loop->value;
                  return read_function_adapter;
                };
                const auto get_write_function = [&] {
                  WriteFunctionAdapter write_function_adapter;
                  write_function_adapter.orig_domain = options.domain();
                  write_function_adapter.state->python_function =
                      py::reinterpret_borrow<py::object>(write_function->value);
                  write_function_adapter.state->loop.obj = loop->value;
                  return write_function_adapter;
                };
                if (read_function && !write_function) {
                  return TensorStore<>(ValueOrThrow(
                      VirtualChunked(get_read_function(), std::move(options))));
                } else if (read_function && write_function) {
                  return TensorStore<>(ValueOrThrow(
                      VirtualChunked(get_read_function(), get_write_function(),
                                     std::move(options))));
                } else if (!read_function && write_function) {
                  return TensorStore<>(ValueOrThrow(VirtualChunkedWriteOnly(
                      get_write_function(), std::move(options))));

                } else {
                  throw py::value_error(
                      "Must specify `read_function`, `write_function`, or "
                      "both");
                }
              },
              doc.c_str(), py::arg("read_function") = std::nullopt,
              py::arg("write_function") = std::nullopt, py::kw_only(),
              py::arg("loop") = std::nullopt,
              MakeKeywordArgumentPyArg(param_def)...);
        });
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterVirtualChunkedBindings, /*priority=*/-300);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore
