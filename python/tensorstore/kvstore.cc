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

#include "python/tensorstore/context.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/keyword_arguments.h"
#include "python/tensorstore/kvstore.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "python/tensorstore/time.h"
#include "python/tensorstore/transaction.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json/pprint_python.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal_python {
namespace {

namespace py = ::pybind11;

py::bytes CordToPython(const absl::Cord& value) {
  return py::bytes(std::string(value));
}

std::optional<absl::Cord> OptionalCordFromPython(
    std::optional<std::string_view> value) {
  if (!value) return std::nullopt;
  return absl::Cord(std::move(*value));
}

namespace kvstore_spec_setters {

template <ContextBindingMode Mode>
struct SetContextBindingModeBase {
  using type = bool;
  template <typename Self>
  static absl::Status Apply(Self& self, bool value) {
    if (!value) return absl::OkStatus();
    self.Set(Mode);
    return absl::OkStatus();
  }
};

struct SetRetainContext
    : SetContextBindingModeBase<ContextBindingMode::retain> {
  static constexpr const char* name = "retain_context";
  static constexpr const char* doc = R"(

Retain all bound context resources (e.g. specific concurrency pools, specific
cache pools).

The resultant :py:obj:`~tensorstore.KvStore.Spec` may be used to re-open the
:py:obj:`~tensorstore.KvStore` using the identical context resources.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetUnbindContext
    : SetContextBindingModeBase<ContextBindingMode::unbind> {
  static constexpr const char* name = "unbind_context";
  static constexpr const char* doc = R"(

Convert any bound context resources to context resource specs that fully capture
the graph of shared context resources and interdependencies.

Re-binding/re-opening the resultant spec will result in a new graph of new
context resources that is isomorphic to the original graph of context resources.
The resultant spec will not refer to any external context resources;
consequently, binding it to any specific context will have the same effect as
binding it to a default context.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetStripContext : SetContextBindingModeBase<ContextBindingMode::strip> {
  static constexpr const char* name = "strip_context";
  static constexpr const char* doc = R"(

Replace any bound context resources and unbound context resource specs by
default context resource specs.

If the resultant :py:obj:`~tensorstore.KvStore.Spec` is re-opened with, or
re-bound to, a new context, it will use the default context resources specified
by that context.

Specifying a value of :python:`False` has no effect.

)";
};

struct SetContext {
  using type = internal_context::ContextImplPtr;
  static constexpr const char* name = "context";
  static constexpr const char* doc = R"(

Bind any context resource specs using the specified shared resource context.

Any already-bound context resources remain unchanged.  Additionally, any context
resources specified by a nested :json:schema:`KvStore.context` spec will be
created as specified, but won't be overridden by :py:param:`.context`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    self.Set(WrapImpl(std::move(value)));
    return absl::OkStatus();
  }
};

struct SetTransaction {
  using type = internal::TransactionState::CommitPtr;
  static constexpr const char* name = "transaction";
  static constexpr const char* doc = R"(

Transaction to use for read/write operations.  By default, operations are
non-transactional.

.. note::

   To perform transactional operations using a :py:obj:`KvStore` that was
   previously opened without a transaction, use
   :py:obj:`KvStore.with_transaction`.

)";
  template <typename Self>
  static absl::Status Apply(Self& self, type value) {
    self.Set(internal::TransactionState::ToTransaction(std::move(value)));
    return absl::OkStatus();
  }
};

constexpr auto ForwardSpecRequestOptions = [](auto callback,
                                              auto... other_param) {
  callback(other_param..., SetRetainContext{}, SetUnbindContext{});
};

constexpr auto ForwardSpecConvertOptions = [](auto callback,
                                              auto... other_param) {
  callback(other_param..., SetUnbindContext{}, SetStripContext{}, SetContext{});
};

constexpr auto ForwardSpecOpenOptions = [](auto callback, auto... other_param) {
  callback(other_param..., SetContext{}, SetTransaction{});
};

}  // namespace kvstore_spec_setters

using KvStoreCls = py::class_<PythonKvStoreObject>;
using KvStoreSpecCls = py::class_<PythonKvStoreSpecObject>;
using KeyRangeCls = py::class_<KeyRange>;
using TimestampedStorageGenerationCls =
    py::class_<TimestampedStorageGeneration>;
using ReadResultCls = py::class_<kvstore::ReadResult>;

KvStoreCls MakeKvStoreClass(py::module m) {
  auto cls = PythonKvStoreObject::Define(R"(
Key-value store that maps an ordered set of byte string keys to byte string values.

This is used as the storage interface for most of the
:ref:`TensorStore drivers<tensorstore-drivers>`.

The actual storage mechanism is determined by the
:ref:`driver<key-value-store-drivers>`.

Example:

    >>> store = await ts.KvStore.open({'driver': 'memory'})
    >>> await store.write(b'a', b'value')
    KvStore.TimestampedStorageGeneration(...)
    >>> await store.read(b'a')
    KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
    >>> await store.read(b'b')
    KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))
    >>> await store.list()
    [b'a']

By default, operations are non-transactional, but transactional operations are
also supported:

    >>> txn = ts.Transaction()
    >>> store.with_transaction(txn)[b'a']
    b'value'
    >>> store.with_transaction(txn)[b'a'] = b'new value'
    >>> store.with_transaction(txn)[b'a']
    b'new value'
    >>> store[b'a']
    b'value'
    >>> txn.commit_sync()
    >>> store[b'a']
    b'new value'

Group:
  Core

Classes
-------

Constructors
------------

Accessors
---------

I/O
---

Synchronous I/O
---------------

)");
  DisallowInstantiationFromPython(cls);
  m.attr("KvStore") = cls;
  return cls;
}

void DefineKvStoreAttributes(KvStoreCls& cls) {
  using Self = PythonKvStoreObject;

  cls.def_property(
      "path", [](Self& self) -> std::string_view { return self.value.path; },
      [](Self& self, std::string path) { self.value.path = std::move(path); },
      R"(
Path prefix within the base key-value store.

Example:

    >>> store = await ts.KvStore.open({
    ...     'driver': 'gcs',
    ...     'bucket': 'my-bucket',
    ...     'path': 'path/to/object'
    ... })
    >>> store.spec()
    KvStore.Spec({'bucket': 'my-bucket', 'driver': 'gcs', 'path': 'path/to/object'})
    >>> store.path
    'path/to/object'

Group:
  Accessors
)");

  cls.def_property_readonly(
      "url",
      [](Self& self) -> std::string {
        return ValueOrThrow(self.value.ToUrl());
      },
      R"(
:json:schema:`URL representation<KvStoreUrl>` of the key-value store specification.

Example:

    >>> store = await ts.KvStore.open({
    ...     'driver': 'gcs',
    ...     'bucket': 'my-bucket',
    ...     'path': 'path/to/object'
    ... })
    >>> store.url
    'gs://my-bucket/path/to/object'

Group:
  Accessors
)");

  cls.def_property(
      "transaction",
      [](Self& self) -> std::optional<internal::TransactionState::CommitPtr> {
        if (self.value.transaction == no_transaction) return std::nullopt;
        return internal::TransactionState::ToCommitPtr(self.value.transaction);
      },
      [](Self& self,
         std::optional<internal::TransactionState::CommitPtr> transaction) {
        if (!transaction) {
          self.value.transaction = no_transaction;
        } else {
          self.value.transaction = internal::TransactionState::ToTransaction(
              std::move(*transaction));
        }
      },
      R"(
Transaction bound to this key-value store.

Group:
  Transactions
)");

  cls.def(
      "with_transaction",
      [](Self& self,
         std::optional<internal::TransactionState::CommitPtr> transaction)
          -> KvStore {
        if (!transaction) transaction.emplace();
        return ValueOrThrow(
            self.value |
            internal::TransactionState::ToTransaction(std::move(*transaction)));
      },
      R"(Returns a transaction-bound view of this key-value store.

The returned view may be used to perform transactional read/write operations.

Example:

    >>> store = await ts.KvStore.open({'driver': 'memory'})
    >>> txn = ts.Transaction()
    >>> await store.with_transaction(txn).write(b'a', b'value')
    >>> (await store.with_transaction(txn).read(b'a')).value
    b'value'
    >>> await txn.commit_async()

Group:
  Transactions

)",
      py::arg("transaction"));

  cls.def(
      "list",
      [](Self& self, std::optional<KeyRange> range,
         size_t strip_prefix_length) {
        kvstore::ListOptions options;
        if (range) options.range = std::move(*range);
        options.strip_prefix_length = strip_prefix_length;
        return MapFutureValue(
            InlineExecutor{},
            [](auto items) { return BytesVector{std::move(items)}; },
            kvstore::ListFuture(self.value, std::move(options)));
      },
      py::arg("range") = std::nullopt, py::arg("strip_prefix_length") = 0,
      R"(
Lists the keys in the key-value store.

Example:

    >>> store = ts.KvStore.open({'driver': 'memory'}).result()
    >>> store[b'a'] = b'value'
    >>> store[b'b'] = b'value'
    >>> store.list().result()
    [b'a', b'b']

Args:

  range: If specified, restricts to the specified key range.

  strip_prefix_length: Strips the specified number of bytes from the start of
    the returned keys.

Returns:

  Future that resolves to the list of matching keys, in an unspecified order.

Raises:

  ValueError: If a :py:obj:`.transaction` is specified.

Warning:

  This returns all keys within :py:param:`range` as a single :py:obj:`list`.  If
  there are a large number of matching keys, this can consume a large amount of
  memory.

Group:
  I/O

)");

  cls.def(
      "read",
      [](Self& self, std::string_view key,
         std::optional<std::string> if_not_equal,
         std::optional<double> staleness_bound) {
        kvstore::ReadOptions options;
        if (if_not_equal) {
          options.if_not_equal.value = std::move(*if_not_equal);
        }
        if (staleness_bound) {
          options.staleness_bound = FromPythonTimestamp(*staleness_bound);
        }
        return kvstore::Read(self.value, key, std::move(options));
      },
      py::arg("key"), py::kw_only(), py::arg("if_not_equal") = std::nullopt,
      py::arg("staleness_bound") = std::nullopt, R"(
Reads the value of a single key.

A missing key is not treated as an error; instead, a :py:obj:`.ReadResult` with
:py:obj:`.ReadResult.state` set to :python:`'missing'` is returned.

Note:

  The behavior in the case of a missing key differs from that of
  :py:obj:`.__getitem__`, which raises :py:obj:`KeyError` to indicate a missing
  key.

Example:

    >>> store = await ts.KvStore.open({'driver': 'memory'})
    >>> await store.write(b'a', b'value')
    KvStore.TimestampedStorageGeneration(...)
    >>> await store.read(b'a')
    KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
    >>> store[b'a']
    b'value'
    >>> await store.read(b'b')
    KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))
    >>> store[b'b']
    Traceback (most recent call last):
        ...
    KeyError...

    >>> store[b'a'] = b'value'
    >>> store[b'b'] = b'value'
    >>> store.list().result()

If a :py:obj:`.transaction` is bound, the read reflects any writes made within
the transaction, and the commit of the transaction will fail if the value
associated with :py:param:`key` changes after the read due to external writes,
i.e. consistent reads are guaranteed.

Args:

  key: The key to read.  This is appended (without any separator) to the
    existing :py:obj:`.path`, if any.

  if_not_equal: If specified, the read is aborted if the generation associated
    with :py:param:`key` matches :py:param:`if_not_equal`.  An aborted read due
    to this condition is indicated by a :py:obj:`.ReadResult.state` of
    :python:`'unspecified'`.  This may be useful for validating a cached value
    cache validation at a higher level.

  staleness_bound: Specifies a time in (fractional) seconds since the Unix
    epoch.  If specified, data that is cached internally by the kvstore
    implementation may be used without validation if not older than the
    :py:param:`staleness_bound`.  Cached data older than
    :py:param:`staleness_bound` must be validated before being returned.  A
    value of :python:`float('inf')` indicates that the result must be current as
    of the time the :py:obj:`.read` request was made, i.e. it is equivalent to
    specifying a value of :python:`time.time()`.  A value of
    :python:`float('-inf')` indicates that cached data may be returned without
    validation irrespective of its age.

Returns:
  Future that resolves when the read operation completes.

See also:

  - :py:obj:`.write`
  - :py:obj:`.__getitem__`
  - :py:obj:`.__setitem__`
  - :py:obj:`.__delitem__`

Group:
  I/O
)");

  cls.attr("__iter__") = py::none();

  cls.def(
      "__getitem__",
      [](Self& self, std::string_view key) {
        auto result =
            ValueOrThrow(InterruptibleWait(kvstore::Read(self.value, key)));
        if (result.state == kvstore::ReadResult::kMissing) {
          throw py::key_error();
        }
        return CordToPython(result.value);
      },
      py::arg("key"), R"(
Synchronously reads the value of a single key.

Example:

    >>> store = ts.KvStore.open({'driver': 'memory'}).result()
    >>> store[b'a'] = b'value'
    >>> store[b'a']
    b'value'
    >>> store[b'b']
    Traceback (most recent call last):
        ...
    KeyError...

Args:

  key: The key to read.  This is appended (without any separator) to the
    existing :py:obj:`.path`, if any.

Returns:

  The value associated with :py:param:`key` on success.

Raises:

  KeyError: If :py:param:`key` is not found.
  Exception: If an I/O error occurs.

Note:

  The current thread is blocked until the read completes, but computations in
  other threads may continue.

See also:

  - :py:obj:`.read`
  - :py:obj:`.__setitem__`
  - :py:obj:`.__delitem__`

Group:
  Synchronous I/O
)");

  cls.def(
      "write",
      [](Self& self, std::string_view key,
         std::optional<std::string_view> value,
         std::optional<std::string> if_equal) {
        kvstore::WriteOptions options;
        if (if_equal) {
          options.if_equal = StorageGeneration{std::move(*if_equal)};
        }
        return kvstore::Write(self.value, key, OptionalCordFromPython(value),
                              std::move(options));
      },
      py::arg("key"), py::arg("value"), py::kw_only(),
      py::arg("if_equal") = std::nullopt, R"(
Writes or deletes a single key.

Example:

    >>> store = await ts.KvStore.open({'driver': 'memory'})
    >>> await store.write(b'a', b'value')
    KvStore.TimestampedStorageGeneration(...)
    >>> await store.read(b'a')
    KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
    >>> await store.write(b'a', None)
    KvStore.TimestampedStorageGeneration(...)
    >>> await store.read(b'a')
    KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))

Args:

  key: Key to write/delete.  This is appended (without any separator) to the
    existing :py:obj:`.path`, if any.

  value: Value to store, or :py:obj:`None` to delete.

  if_equal: If specified, indicates a conditional write operation.  The write is
    performed only if the existing generation associated with :py:param:`key`
    matches :py:param:`if_equal`.

Returns:

  - If no :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
    resolves to the new storage generation for :py:param:`key` once the write
    operation completes and durability is guaranteed (to the extent supported by
    the :ref:`driver<key-value-store-drivers>`).

  - If a :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
    resolves to an empty storage generation once the write operation is recorded
    in the transaction.  The write operation is not actually performed until the
    transaction is committed.

See also:

  - :py:obj:`.__setitem__`
  - :py:obj:`.__delitem__`

Group:
  I/O
)");

  cls.def(
      "__setitem__",
      [](Self& self, std::string_view key,
         std::optional<std::string_view> value) {
        ValueOrThrow(InterruptibleWait(
            kvstore::Write(self.value, key, OptionalCordFromPython(value))));
      },
      py::arg("key"), py::arg("value"), R"(
Synchronously writes the value of a single key.

Example:

    >>> store = ts.KvStore.open({'driver': 'memory'}).result()
    >>> store[b'a'] = b'value'
    >>> store[b'a']
    b'value'
    >>> store[b'b']
    Traceback (most recent call last):
        ...
    KeyError...
    >>> store[b'a'] = None
    >>> store[b'a']
    Traceback (most recent call last):
        ...
    KeyError...

Args:


  key: Key to write/delete.  This is appended (without any separator) to the
    existing :py:obj:`.path`, if any.

  value: Value to store, or :py:obj:`None` to delete.

Raises:

  Exception: If an I/O error occurs.

Note:

  - If no :py:obj:`.transaction` is specified, the current thread is blocked
    until the write completes and durability is guaranteed (to the extent
    supported by the :ref:`driver<key-value-store-drivers>`).

  - If a :py:obj:`.transaction` is specified, the current thread is blocked
    until the write is recorded in the transaction.  The actual write operation
    is not performed until the transaction is committed.

  Computations in other threads may continue even while the current thread is
  blocked.

See also:

  - :py:obj:`.write`
  - :py:obj:`.__getitem__`
  - :py:obj:`.__delitem__`

Group:
  Synchronous I/O
)");

  cls.def(
      "__delitem__",
      [](Self& self, std::string_view key) {
        ValueOrThrow(
            InterruptibleWait(kvstore::Write(self.value, key, std::nullopt)));
      },
      py::arg("key"), R"(
Synchronously deletes a single key.

Example:

    >>> store = ts.KvStore.open({'driver': 'memory'}).result()
    >>> store[b'a'] = b'value'
    >>> store[b'a']
    b'value'
    >>> del store[b'a']
    >>> store[b'a']
    Traceback (most recent call last):
        ...
    KeyError...

Args:

  key: Key to delete.  This is appended (without any separator) to the existing
    :py:obj:`.path`, if any.

Raises:

  Exception: If an I/O error occurs.

Note:

  - If no :py:obj:`.transaction` is specified, the current thread is blocked
    until the delete completes and durability is guaranteed (to the extent
    supported by the :ref:`driver<key-value-store-drivers>`).

  - If a :py:obj:`.transaction` is specified, the current thread is blocked
    until the delete is recorded in the transaction.  The actual delete
    operation is not performed until the transaction is committed.

  Computations in other threads may continue even while the current thread is
  blocked.

See also:

  - :py:obj:`.write`
  - :py:obj:`.__getitem__`
  - :py:obj:`.__setitem__`

Group:
  Synchronous I/O
)");

  cls.def(
      "delete_range",
      [](Self& self, KeyRange range) {
        return kvstore::DeleteRange(self.value, std::move(range));
      },
      py::arg("range"), R"(
Deletes a key range.

Example:

    >>> store = await ts.KvStore.open({'driver': 'memory'})
    >>> await store.write(b'a', b'value')
    >>> await store.write(b'b', b'value')
    >>> await store.write(b'c', b'value')
    >>> await store.list()
    [b'a', b'b', b'c']
    >>> await store.delete_range(ts.KvStore.KeyRange(b'aa', b'cc'))
    >>> await store.list()
    [b'a']

Args:

  range: Key range to delete.  This is relative to the existing :py:obj:`.path`,
    if any.

Returns:

  - If no :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
    becomes ready when the delete operation has completed and durability is
    guaranteed (to the extent supported by the
    :ref:`driver<key-value-store-drivers>`).

  - If a :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
    becomes ready when the delete operation is recorded in the transaction.  The
    delete operation is not actually performed until the transaction is
    committed.

Group:
  I/O
)");

  kvstore_spec_setters::ForwardSpecRequestOptions([&](auto... param_def) {
    std::string doc = R"(
Spec that may be used to re-open or re-create the key-value store.

Example:

    >>> kvstore = await ts.KvStore.open({'driver': 'memory', 'path': 'abc/'})
    >>> kvstore.spec()
    KvStore.Spec({'driver': 'memory', 'path': 'abc/'})
    >>> kvstore.spec(unbind_context=True)
    KvStore.Spec({'context': {'memory_key_value_store': {}}, 'driver': 'memory', 'path': 'abc/'})
    >>> kvstore.spec(retain_context=True)
    KvStore.Spec({
      'context': {'memory_key_value_store': {}},
      'driver': 'memory',
      'memory_key_value_store': ['memory_key_value_store'],
      'path': 'abc/',
    })

Args:

)";
    AppendKeywordArgumentDocs(doc, param_def...);

    doc += R"(

Group:
  Accessors

)";

    cls.def(
        "spec",
        [](Self& self, KeywordArgument<decltype(param_def)>... kwarg) {
          kvstore::SpecRequestOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          return self.value.spec(std::move(options));
        },
        doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
  });

  kvstore_spec_setters::ForwardSpecOpenOptions([&](auto... param_def) {
    std::string doc = R"(
Opens a key-value store.

Example of opening from a :json:schema:`JSON KvStore spec<KvStore>`:

    >>> kvstore = await ts.KvStore.open({'driver': 'memory', 'path': 'abc/'})
    >>> await kvstore.write(b'x', b'y')
    KvStore.TimestampedStorageGeneration(b'...', ...)
    >>> await kvstore.read(b'x')
    KvStore.ReadResult(state='value', value=b'y', stamp=KvStore.TimestampedStorageGeneration(b'...', ...))

Example of opening from a :json:schema:`URL<KvStoreUrl>`:

    >>> kvstore = await ts.KvStore.open('memory://abc/')
    >>> kvstore.spec()
    KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

Example of opening from an existing :py:obj:`KvStore.Spec`:

    >>> spec = ts.KvStore.Spec({'driver': 'memory', 'path': 'abc/'})
    >>> kvstore = await ts.KvStore.open(spec)
    >>> kvstore.spec()
    KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

Args:

  spec: Key-value store spec to open.  May also be specified as
    :json:schema:`JSON<KvStore>` or a :json:schema:`URL<KvStoreUrl>`.

)";
    AppendKeywordArgumentDocs(doc, param_def...);
    doc += R"(

Group:
  Constructors

)";

    cls.def_static(
        "open",
        [](std::variant<PythonKvStoreSpecObject*, ::nlohmann::json> spec_like,
           KeywordArgument<decltype(param_def)>... kwarg) {
          kvstore::OpenOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          kvstore::Spec spec;
          PythonObjectReferenceManager reference_manager;
          if (auto* p = std::get_if<PythonKvStoreSpecObject*>(&spec_like)) {
            spec = (**p).value;
            reference_manager = (**p).reference_manager();
          } else if (auto* p = std::get_if<::nlohmann::json>(&spec_like)) {
            spec = ValueOrThrow(kvstore::Spec::FromJson(std::move(*p)));
          }
          return PythonFutureWrapper<KvStore>(
              kvstore::Open(std::move(spec), std::move(options)),
              std::move(reference_manager));
        },
        doc.c_str(), py::arg("spec"), py::kw_only(),
        MakeKeywordArgumentPyArg(param_def)...);
  });

  cls.def(
      "__repr__",
      [](Self& self) {
        return internal_python::PrettyPrintJsonAsPythonRepr(
            self.value.spec(tensorstore::unbind_context) |
                [](const auto& spec) { return spec.ToJson(); },
            "KvStore(", ")");
      },
      R"(
Returns a string representation based on the  :json:schema:`JSON representation<KvStore>`.

Example:

    >>> kvstore = await ts.KvStore.open({
    ...     'driver': 'file',
    ...     'path': 'tmp/data/'
    ... })
    >>> kvstore
    KvStore({'context': {'file_io_concurrency': {}}, 'driver': 'file', 'path': 'tmp/data/'})

)");

  cls.def(
      "copy", [](Self& self) { return self.value; }, R"(
Returns a copy of the key-value store.

Example:

  >>> a = await ts.KvStore.open({'driver': 'file', 'path': 'tmp/data/'})
  >>> b = a.copy()
  >>> a.path = 'tmp/data/abc/'
  >>> a
  KvStore({
    'context': {'file_io_concurrency': {}},
    'driver': 'file',
    'path': 'tmp/data/abc/',
  })
  >>> b
  KvStore({'context': {'file_io_concurrency': {}}, 'driver': 'file', 'path': 'tmp/data/'})

Group:
  Accessors
)");

  cls.def("__copy__", [](Self& self) { return self.value; });

  cls.def(
      "__deepcopy__", [](Self& self, py::dict memo) { return self.value; },
      py::arg("memo"));

  EnableGarbageCollectedObjectPicklingFromSerialization(cls);
}

KvStoreSpecCls MakeKvStoreSpecClass(KvStoreCls& kvstore_cls) {
  auto cls = PythonKvStoreSpecObject::Define(R"(
Parsed representation of a :json:schema:`JSON key-value store<KvStore>` specification.
)");
  kvstore_cls.attr("Spec") = cls;
  // These properties don't get set correctly by default for nested classes, and
  // are needed for pickling to work correctly.
  cls.attr("__module__") = "tensorstore";
  cls.attr("__qualname__") = "KvStore.Spec";
  return cls;
}

void DefineKvStoreSpecAttributes(KvStoreSpecCls& cls) {
  using Self = PythonKvStoreSpecObject;

  kvstore_spec_setters::ForwardSpecConvertOptions([&](auto... param_def) {
    std::string doc = R"(
Modifies a spec.

Example:

    >>> spec = ts.KvStore.Spec({
    ...     'driver': 'memory',
    ...     'path': 'abc/',
    ...     'memory_key_value_store': 'memory_key_value_store#a'
    ... })
    >>> spec.update(context=ts.Context({'memory_key_value_store#a': {}}))
    >>> spec
    KvStore.Spec({
      'context': {'memory_key_value_store#a': {}},
      'driver': 'memory',
      'memory_key_value_store': ['memory_key_value_store#a'],
      'path': 'abc/',
    })
    >>> spec.update(unbind_context=True)
    >>> spec
    KvStore.Spec({
      'context': {'memory_key_value_store#a': {}},
      'driver': 'memory',
      'memory_key_value_store': 'memory_key_value_store#a',
      'path': 'abc/',
    })
    >>> spec.update(strip_context=True)
    >>> spec
    KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

Args:

)";
    AppendKeywordArgumentDocs(doc, param_def...);

    doc += R"(

Group:
  Mutators

)";

    cls.def(
        "update",
        [](Self& self, KeywordArgument<decltype(param_def)>... kwarg) {
          kvstore::SpecConvertOptions options;
          ApplyKeywordArguments<decltype(param_def)...>(options, kwarg...);
          ThrowStatusException(self.value.Set(std::move(options)));
        },
        doc.c_str(), py::kw_only(), MakeKeywordArgumentPyArg(param_def)...);
  });

  cls.def_property(
      "path", [](Self& self) -> std::string_view { return self.value.path; },
      [](Self& self, std::string_view path) { self.value.path = path; },
      R"(
Path prefix within the base key-value store.

Example:

    >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
    >>> spec.path
    'tmp/data'

Group:
  Accessors
)");

  cls.def_property_readonly(
      "url",
      [](Self& self) -> std::string {
        return ValueOrThrow(self.value.ToUrl());
      },
      R"(
:json:schema:`URL representation<KvStoreUrl>` of the key-value store specification.

Example:

    >>> spec = ts.KvStore.Spec({
    ...     'driver': 'gcs',
    ...     'bucket': 'my-bucket',
    ...     'path': 'path/to/object'
    ... })
    >>> spec.url
    'gs://my-bucket/path/to/object'

Group:
  Accessors
)");

  cls.def(
      "__add__",
      [](Self& self, std::string_view suffix) {
        auto new_spec = self.value;
        new_spec.AppendSuffix(suffix);
        return new_spec;
      },
      py::arg("suffix"), R"(
Returns a key-value store with the suffix appended to the path.

The suffix is appended directly to :py:obj:`.path` without any separator.  To
ensure there is a :python:`'/'` separator, use :py:obj:`.__truediv__` instead.

Example:

    >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
    >>> spec + '/abc'
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})
    >>> spec + 'abc'
    KvStore.Spec({'driver': 'file', 'path': 'tmp/dataabc'})

Group:
  Operators
)");

  cls.def(
      "__truediv__",
      [](Self& self, std::string_view component) {
        auto new_spec = self.value;
        new_spec.AppendPathComponent(component);
        return new_spec;
      },
      py::arg("component"), R"(
Returns a key-value store with an additional path component joined to the path.

Example:

    >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
    >>> spec / 'abc'
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})
    >>> spec / '/abc'
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})

Group:
  Operators
)");

  cls.def(
      "__new__",
      [](py::handle cls, ::nlohmann::json json_spec) {
        return ValueOrThrow(kvstore::Spec::FromJson(std::move(json_spec)));
      },
      py::arg("json"), R"(
Constructs from the :json:schema:`JSON representation<KvStore>` or a :json:schema:`URL<KvStoreUrl>`.

Example of constructing from the :json:schema:`JSON representation<KvStore>`:

    >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
    >>> spec
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

Example of constructing from a :json:schema:`URL<KvStoreUrl>`:

    >>> spec = ts.KvStore.Spec('file://tmp/data/')
    >>> spec
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

)");

  cls.def(
      "to_json",
      [](Self& self, bool include_defaults) {
        return ValueOrThrow(
            self.value.ToJson({IncludeDefaults{include_defaults}}));
      },
      R"(
Converts to the :json:schema:`JSON representation<KvStore>`.

Example:

  >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/dataset/'})
  >>> spec /= 'abc/'
  >>> spec.to_json()
  {'driver': 'file', 'path': 'tmp/dataset/abc/'}
  >>> spec.to_json(include_defaults=True)
  {'context': {},
   'driver': 'file',
   'file_io_concurrency': 'file_io_concurrency',
   'path': 'tmp/dataset/abc/'}

Group:
  Accessors
)",
      py::arg("include_defaults") = false);

  cls.def(
      "copy", [](Self& self) { return self.value; }, R"(
Returns a copy of the key-value store spec.

Example:

  >>> a = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
  >>> b = a.copy()
  >>> a.path = 'tmp/data/abc/'
  >>> a
  KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc/'})
  >>> b
  KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

Group:
  Accessors
)");

  cls.def("__copy__", [](Self& self) { return self.value; });

  cls.def(
      "__deepcopy__", [](Self& self, py::dict memo) { return self.value; },
      py::arg("memo"));

  cls.def(
      "__repr__",
      [](Self& self) {
        JsonSerializationOptions options;
        options.preserve_bound_context_resources_ = true;
        return internal_python::PrettyPrintJsonAsPythonRepr(
            self.value.ToJson(options), "KvStore.Spec(", ")");
      },
      R"(
Returns a string representation based on the  :json:schema:`JSON representation<KvStore>`.

Example:

    >>> ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
    KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

)");

  EnableGarbageCollectedObjectPicklingFromSerialization(cls);
}

KeyRangeCls MakeKeyRangeClass(KvStoreCls& kvstore_cls) {
  return KeyRangeCls(kvstore_cls, "KeyRange", R"(
Half-open interval of byte string keys, according to lexicographical order.
)");
}

void DefineKeyRangeAttributes(KeyRangeCls& cls) {
  using Self = KeyRange;
  cls.def(py::init([](std::string inclusive_min, std::string exclusive_max) {
            return KeyRange(std::move(inclusive_min), std::move(exclusive_max));
          }),
          py::arg("inclusive_min") = "", py::arg("exclusive_max") = "",
          R"(
Constructs a key range from the specified half-open bounds.

Args:

  inclusive_min: Inclusive lower bound of the range.  In accordance with the
    usual lexicographical order, an empty string indicates no lower bound.

  exclusive_max: Exclusive upper bound of the range.  As a special case, an
    empty string indicates no upper bound.

)");

  cls.def_property(
      "inclusive_min",
      [](const Self& self) -> std::string_view { return self.inclusive_min; },
      [](Self& self, std::string value) { self.inclusive_min = value; }, R"(
Inclusive lower bound of the range.

In accordance with the usual lexicographical order, an empty string indicates no
lower bound.

Group:
  Accessors
)");

  cls.def_property(
      "exclusive_max",
      [](const Self& self) -> std::string_view { return self.exclusive_max; },
      [](Self& self, std::string value) { self.exclusive_max = value; }, R"(
Exclusive upper bound of the range.

As a special case, an empty string indicates no upper bound.

Group:
  Accessors
)");

  cls.def(
      "copy", [](const Self& self) { return self; }, R"(
Returns a copy of the range.

Group:
  Accessors
)");

  cls.def("__copy__", [](const Self& self) { return self; });
  cls.def(
      "__deepcopy__", [](const Self& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(cls);

  cls.def_property_readonly(
      "empty", [](const KeyRange& self) { return self.empty(); },
      R"(
Indicates if the range contains no keys.

Example:

    >>> r = ts.KvStore.KeyRange(b'x', b'y')
    >>> r.empty
    False
    >>> r = ts.KvStore.KeyRange(b'x', b'x')
    >>> r.empty
    True
    >>> r = ts.KvStore.KeyRange(b'y', b'x')
    >>> r.empty
    True

Group:
  Accessors
)");

  cls.def(
      "__eq__",
      [](const KeyRange& self, const KeyRange& other) { return self == other; },
      py::arg("other"), R"(
Compares with another range for equality.
)");
}

TimestampedStorageGenerationCls MakeTimestampedStorageGenerationClass(
    KvStoreCls& kvstore_cls) {
  return TimestampedStorageGenerationCls(kvstore_cls,
                                         "TimestampedStorageGeneration", R"(
Specifies a storage generation identifier and a timestamp.
)");
}

void DefineTimestampedStorageGenerationAttributes(
    TimestampedStorageGenerationCls& cls) {
  using Self = TimestampedStorageGeneration;
  cls.def(py::init([](std::string generation, double time) {
            return Self(StorageGeneration{std::move(generation)},
                        internal_python::FromPythonTimestamp(time));
          }),
          py::arg("generation") = "",
          py::arg("time") = -std::numeric_limits<double>::infinity(),
          R"(
Constructs from a storage generation and time.
)");

  cls.def("__repr__", [](const Self& self) {
    return tensorstore::StrCat("KvStore.TimestampedStorageGeneration(",
                               py::repr(py::bytes(self.generation.value)), ", ",
                               internal_python::ToPythonTimestamp(self.time),
                               ")");
  });

  cls.def_property(
      "generation",
      [](const Self& self) -> py::bytes {
        return py::bytes(self.generation.value);
      },
      [](Self& self, std::string value) {
        self.generation.value = std::move(value);
      },
      R"(
Identifies a specific version of a key-value store entry.

An empty string :python:`b''` indicates an unspecified version.

Group:
  Accessors
)");

  cls.def_property(
      "time",
      [](const Self& self) -> double {
        return internal_python::ToPythonTimestamp(self.time);
      },
      [](Self& self, double value) {
        self.time = internal_python::FromPythonTimestamp(value);
      },
      R"(
Time (seconds since Unix epoch) at which :py:obj:`.generation` is valid.

Group:
  Accessors
)");

  cls.def(
      "__eq__",
      [](const Self& self, const Self& other) { return self == other; },
      py::arg("other"), R"(
Compares two timestamped storage generations for equality.
)");

  cls.def("__copy__", [](const Self& self) { return self; });
  cls.def(
      "__deepcopy__", [](const Self& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(cls);
}

ReadResultCls MakeReadResultClass(KvStoreCls& kvstore_cls) {
  return ReadResultCls(kvstore_cls, "ReadResult", R"(
Specifies the result of a read operation.
)");
}

void DefineReadResultAttributes(ReadResultCls& cls) {
  using Self = kvstore::ReadResult;

  cls.def(py::init([](Self::State state, std::string value,
                      TimestampedStorageGeneration stamp) {
            return Self(state, absl::Cord(std::move(value)), std::move(stamp));
          }),
          py::arg("state") = Self::State::kUnspecified, py::arg("value") = "",
          py::arg("stamp") = TimestampedStorageGeneration(), R"(
Constructs a read result.
)");

  cls.def_property(
      "state", [](const Self& self) { return self.state; },
      [](Self& self, Self::State state) { self.state = state; },
      R"(
Indicates the interpretation of :py:obj:`.value`.
)");

  cls.def_property(
      "value",
      [](const Self& self) -> py::bytes { return CordToPython(self.value); },
      [](Self& self, std::string value) {
        self.value = absl::Cord(std::move(value));
      },
      R"(
Value associated with the key.
)");

  cls.def_property(
      "stamp", [](const Self& self) { return self.stamp; },
      [](Self& self, TimestampedStorageGeneration stamp) {
        self.stamp = std::move(stamp);
      },
      R"(
Generation and timestamp associated with the value.
)");

  cls.def("__repr__", [](const Self& self) {
    return tensorstore::StrCat(
        "KvStore.ReadResult(state=", pybind11::repr(py::cast(self.state)),
        ", value=", pybind11::repr(py::bytes(std::string(self.value))),
        ", stamp=", pybind11::repr(py::cast(self.stamp)), ")");
  });

  cls.def("__copy__", [](const Self& self) { return self; });
  cls.def(
      "__deepcopy__", [](const Self& self, py::dict memo) { return self; },
      py::arg("memo"));

  EnablePicklingFromSerialization(cls);
}

void RegisterKvStoreBindings(pybind11::module m, Executor defer) {
  auto kvstore_cls = MakeKvStoreClass(m);
  defer([kvstore_cls]() mutable { DefineKvStoreAttributes(kvstore_cls); });
  defer([cls = MakeKvStoreSpecClass(kvstore_cls)]() mutable {
    DefineKvStoreSpecAttributes(cls);
  });
  defer([cls = MakeKeyRangeClass(kvstore_cls)]() mutable {
    DefineKeyRangeAttributes(cls);
  });
  defer([cls = MakeTimestampedStorageGenerationClass(kvstore_cls)]() mutable {
    DefineTimestampedStorageGenerationAttributes(cls);
  });
  defer([cls = MakeReadResultClass(kvstore_cls)]() mutable {
    DefineReadResultAttributes(cls);
  });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterKvStoreBindings, /*priority=*/-550);
}

}  // namespace
}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

handle type_caster<tensorstore::kvstore::ReadResult::State>::cast(
    tensorstore::kvstore::ReadResult::State value,
    return_value_policy /* policy */, handle /* parent */) {
  using State = tensorstore::kvstore::ReadResult::State;
  std::string_view s;
  switch (value) {
    case State::kUnspecified:
      s = "unspecified";
      break;
    case State::kMissing:
      s = "missing";
      break;
    case State::kValue:
      s = "value";
      break;
  }
  return pybind11::cast(s).release();
}

bool type_caster<tensorstore::kvstore::ReadResult::State>::load(handle src,
                                                                bool convert) {
  using State = tensorstore::kvstore::ReadResult::State;
  const char* s;
  Py_ssize_t size;
  s = PyUnicode_AsUTF8AndSize(src.ptr(), &size);
  if (!s) {
    PyErr_Clear();
    return false;
  }
  std::string_view str(s, size);
  if (str == "unspecified") {
    value = State::kUnspecified;
    return true;
  } else if (str == "missing") {
    value = State::kMissing;
    return true;
  } else if (str == "value") {
    value = State::kValue;
    return true;
  }
  return false;
}

handle type_caster<tensorstore::internal_python::BytesVector>::cast(
    const tensorstore::internal_python::BytesVector& value,
    return_value_policy policy, handle parent) {
  auto list = reinterpret_steal<object>(PyList_New(value.value.size()));
  if (!list) throw error_already_set();
  for (size_t i = 0; i < value.value.size(); ++i) {
    PyList_SET_ITEM(list.ptr(), i, bytes(value.value[i]).release().ptr());
  }
  return list.release();
}

}  // namespace detail
}  // namespace pybind11
