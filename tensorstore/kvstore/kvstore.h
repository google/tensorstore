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

#ifndef TENSORSTORE_KVSTORE_KVSTORE_H_
#define TENSORSTORE_KVSTORE_KVSTORE_H_

#include "tensorstore/context.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/option.h"

namespace tensorstore {
namespace kvstore {

/// Options that may be specified when requesting the `Spec` for an open
/// `KvStore`.
///
/// Refer to the documentation of `KvStore::spec` for details.
///
/// \relates Spec
struct SpecRequestOptions : public DriverSpecOptions {
  ContextBindingMode context_binding_mode = ContextBindingMode::unspecified;

  template <typename T>
  constexpr static bool IsOption = DriverSpecOptions::IsOption<T>;

  using DriverSpecOptions::Set;

  void Set(ContextBindingMode value) {
    if (value > context_binding_mode) context_binding_mode = value;
  }
};

template <>
constexpr inline bool SpecRequestOptions::IsOption<ContextBindingMode> = true;

/// Combines a `DriverPtr` with a string path that serves as a key prefix, and
/// an optional transaction.
///
/// \ingroup kvstore
class KvStore {
 public:
  /// Constructs an invalid (null) kvstore.
  ///
  /// \id default
  KvStore() = default;

  /// Constructs from a driver, and optional path and transaction.
  ///
  /// \id driver
  KvStore(DriverPtr driver) : driver(std::move(driver)) {}
  explicit KvStore(DriverPtr driver, Transaction transaction)
      : driver(std::move(driver)), transaction(std::move(transaction)) {}
  explicit KvStore(DriverPtr driver, std::string path,
                   Transaction transaction = no_transaction)
      : driver(std::move(driver)),
        path(std::move(path)),
        transaction(std::move(transaction)) {}

  /// Appends `suffix` to the `path`.
  ///
  /// There is no special treatment of '/'.
  void AppendSuffix(std::string_view suffix) { path += suffix; }

  /// Joins a '/'-separated path component to the end `path`.
  void AppendPathComponent(std::string_view component) {
    internal::AppendPathComponent(path, component);
  }

  /// Returns `true` if this is a valid (non-null) kvstore.
  bool valid() const { return static_cast<bool>(driver); }

  /// Driver spec.
  DriverPtr driver;

  /// Path within the `driver`.
  std::string path;

  /// Returns a Spec that can be used to re-open this `KvStore`.
  ///
  /// Options that modify the returned `Spec` may be specified in any order.
  /// The meaning of the option is determined by its type.
  ///
  /// Supported options are:
  ///
  /// - ContextBindingMode: Defaults to `ContextBindingMode::strip`, such that
  ///   the returned `Spec` does not specify any context resources.  To retain
  ///   the bound context resources, such that the returned `Spec` may be used
  ///   to re-open the `KvStore` with the identical context resources, specify
  ///   `ContextBindingMode::retain`.  Specifying `ContextBindingMode::unbind`
  ///   converts all context resources to context resource specs that may be
  ///   used to re-open the `KvStore` with a graph of new context resources
  ///   isomorphic to the existing graph of context resources.
  ///
  /// \param option Any option compatible with `SpecRequestOptions`.
  /// \error `absl::StatusCode::kUnimplemented` if a JSON representation is
  ///     not supported.  (This behavior is provided by the default
  ///     implementation.)
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecRequestOptions, Option...>,
                   Result<Spec>>
  spec(Option&&... option) const {
    SpecRequestOptions options;
    (options.Set(std::move(option)), ...);
    return spec(std::move(options));
  }
  Result<Spec> spec(SpecRequestOptions&& options) const;

  /// Returns the URL representation if available.
  Result<std::string> ToUrl() const;

  /// Checks if the `driver`, `path`, and `transaction` are identical.
  friend bool operator==(const KvStore& a, const KvStore& b);
  friend bool operator!=(const KvStore& a, const KvStore& b) {
    return !(a == b);
  }

  /// Returns the corresponding non-transactional `KvStore`.
  KvStore non_transactional() const& { return KvStore(driver, path); }
  KvStore non_transactional() && {
    return KvStore(std::move(driver), std::move(path));
  }

  /// Changes to a new transaction.
  ///
  /// Fails if `store` is already associated with an uncommitted transaction.
  ///
  /// This is intended to be used with the "pipeline" `operator|`.
  ///
  /// Example::
  ///
  ///     tensorstore::KvStore store = ...;
  ///     auto transaction = tensorstore::Transaction(tensorstore::isolated);
  ///     TENSORSTORE_ASSIGN_OR_RETURN(store, store | transaction);
  ///
  friend Result<KvStore> ApplyTensorStoreTransaction(KvStore store,
                                                     Transaction transaction) {
    TENSORSTORE_RETURN_IF_ERROR(
        internal::ChangeTransaction(store.transaction, std::move(transaction)));
    return store;
  }

  /// "Pipeline" operator.
  ///
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then `operator|` applies ``y`` to the value of ``x``,
  /// returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
  template <typename Func>
  PipelineResultType<const KvStore&, Func> operator|(Func&& func) const& {
    return std::forward<Func>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<KvStore&&, Func> operator|(Func&& func) && {
    return std::forward<Func>(func)(std::move(*this));
  }

  /// Bound transaction to use for I/O.
  Transaction transaction = no_transaction;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.driver, x.path, x.transaction);
  };
};

/// Driver-agnostic options that may be specified when opening a `DriverPtr`.
///
/// Refer to the documentation of `kvstore::Open` for details.
///
/// \relates Spec
struct DriverOpenOptions {
  Context context;

  template <typename T>
  constexpr static bool IsOption = false;

  void Set(Context value) { context = std::move(value); }
};

/// Driver-agnostic options that may be specified when opening a `KvStore`.
///
/// Refer to the documentation of `kvstore::Open` for details.
///
/// \relates Spec
struct OpenOptions : public DriverOpenOptions {
  Transaction transaction = no_transaction;

  template <typename T>
  constexpr static bool IsOption = DriverOpenOptions::IsOption<T>;

  using DriverOpenOptions::Set;

  void Set(Transaction transaction) {
    this->transaction = std::move(transaction);
  }
};

template <>
constexpr inline bool DriverOpenOptions::IsOption<Context> = true;
template <>
constexpr inline bool OpenOptions::IsOption<Transaction> = true;

/// Opens a `KvStore` based on an already-parsed `kvstore::Spec` and an optional
/// sequence of options.
///
/// Options may be specified in any order, and are identified by their type.
/// Supported option types are:
///
/// - Context: specifies the context in which to obtain any unbound context
///   resources in `spec`.  Any already-bound context resources in `spec`
///   remain unmodified.  If not specified, `Context::Default()` is used.
///
/// - Transaction: specifies a transaction to bind to the returned `KvStore`.
///   Currently this does not affect the open operation itself.
///
/// \param spec KvStore specification.
/// \param json_spec JSON value (which may be a string URL) to be parsed as a
///     `Spec`.
/// \param option Any option compatible with `OpenOptions`.
/// \param options Options for opening the spec.
/// \relates KvStore
Future<KvStore> Open(Spec spec, OpenOptions&& options);
Future<KvStore> Open(::nlohmann::json json_spec, OpenOptions&& options);
template <typename... Option>
static std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                        Future<KvStore>>
Open(Spec spec, Option&&... option) {
  OpenOptions options;
  (options.Set(option), ...);
  return kvstore::Open(std::move(spec), std::move(options));
}
template <typename... Option>
static std::enable_if_t<IsCompatibleOptionSequence<OpenOptions, Option...>,
                        Future<KvStore>>
Open(::nlohmann::json j, Option&&... option) {
  OpenOptions options;
  (options.Set(option), ...);
  return kvstore::Open(std::move(j), std::move(options));
}

}  // namespace kvstore

/// Convenience alias of `kvstore::KvStore`.
///
/// \relates kvstore::KvStore
using KvStore = kvstore::KvStore;

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::kvstore::KvStore)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::KvStore)

#endif  // TENSORSTORE_KVSTORE_KVSTORE_H_
