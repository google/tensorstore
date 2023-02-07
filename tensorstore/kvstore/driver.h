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

#ifndef TENSORSTORE_KVSTORE_DRIVER_H_
#define TENSORSTORE_KVSTORE_DRIVER_H_

#include "absl/status/status.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/any_sender.h"
#include "tensorstore/util/garbage_collection/fwd.h"

namespace tensorstore {
namespace kvstore {

/// Abstract base class representing a key-value store specification, for
/// creating a `Driver` from a JSON representation.
///
/// A `DriverSpec` object specifies:
///
/// - The driver id (as a string);
///
/// - Any driver-specific options, such as a cloud storage bucket or path to the
///   data, and `Context::Resource` objects for any necessary credentials or
///   concurrency pools.
///
/// - A `Context::Spec` with context resource specifications that may be
///   referenced by driver-specific context resource specifications; these
///   context resource specifications override any resources provided by the
///   `Context` object used to bind/open the `DriverSpec`.
///
/// \relates Spec
class DriverSpec : public internal::AtomicReferenceCount<DriverSpec> {
  // For each `Derived` kvstore driver implementation that supports a JSON
  // representation, `internal_kvstore::RegisteredDriverSpec<Derived>` defined
  // in `registry.h` serves as the corresponding `DriverSpec` implementation.
 public:
  virtual ~DriverSpec();

  // Normalizes the spec and `path`, possibly by moving information between the
  // path and driver spec.  This is used by the `http` driver.
  //
  // The default implementation simply returns `absl::OkStatus()`.
  virtual absl::Status NormalizeSpec(std::string& path);

  // Resolves any context references using `context`.
  virtual absl::Status BindContext(const Context& context) = 0;

  // Converts any bound context resources to unbound resource specs.
  virtual void UnbindContext(const internal::ContextSpecBuilder& builder) = 0;

  // Replaces any context resources with default context resource specs.
  virtual void StripContext() = 0;

  // Modifies this `DriverSpec` according to `options`.  This must only be
  // called if `use_count() == 1`.
  virtual absl::Status ApplyOptions(DriverSpecOptions&& options);

  // Encodes any relevant parameters as a cache key.  This should only include
  // parameters relevant after the `Driver` is open that determine whether two
  // `Driver` objects may be used interchangeably.  Parameters that only affect
  // creation should be excluded.
  virtual void EncodeCacheKey(std::string* out) const = 0;

  /// Returns the driver identifier.
  virtual std::string_view driver_id() const = 0;

  // Returns the URL
  virtual Result<std::string> ToUrl(std::string_view path) const;

  // Returns a copy of this spec, used to implement copy-on-write behavior.
  virtual DriverSpecPtr Clone() const = 0;

  // Opens a `Driver` using this spec.
  //
  // \pre All context resources must be bound.
  virtual Future<DriverPtr> DoOpen() const = 0;

  virtual void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const = 0;

  friend class DriverSpecPtr;
  friend class Spec;

  // Specifies context resource overrides.
  Context::Spec context_spec_;

  // Indicates the binding state.
  ContextBindingState context_binding_state_ = ContextBindingState::unknown;
};

/// Abstract base class representing a key-value store.
///
/// Support for different storage systems is provided by individual key-value
/// store drivers, which are defined as derived classes of `Driver`.  Drivers
/// that support a JSON representation should inherit from the CRTP base
/// `internal_kvstore::RegisteredDriver` defined in `registry.h`.
///
/// `Driver` uses intrusive reference counting. `Driver` objects must always be
/// heap-allocated with ownership managed through a `kvstore::DriverPtr`.
///
/// Destroying all references to the `Future` returned from `Read`, `Write`, or
/// `Delete` may (depending on the derived class implementation) cancel the
/// operation.
///
/// The user is not required to hold a reference to the `Driver` while
/// operations are outstanding; releasing the last externally held reference to
/// a `Driver` object does not cancel outstanding operations.
class Driver {
 public:
  using ReadResult = kvstore::ReadResult;
  using ReadOptions = kvstore::ReadOptions;
  using WriteOptions = kvstore::WriteOptions;
  using Key = kvstore::Key;
  using Value = kvstore::Value;
  using ListOptions = kvstore::ListOptions;
  using ReadModifyWriteSource = kvstore::ReadModifyWriteSource;
  using ReadModifyWriteTarget = kvstore::ReadModifyWriteTarget;

  /// Attempts to read the specified key.
  ///
  /// \param key The key to read.
  /// \param options Specifies options for reading.
  /// \returns A Future that resolves when the read completes successfully or
  ///     with an error.
  virtual Future<ReadResult> Read(Key key, ReadOptions options = {});

  /// Performs an optionally-conditional write.
  ///
  /// Atomically updates or deletes the value stored for `key` subject to the
  /// conditions specified in `options`.
  ///
  /// \param key The key to write or delete.
  /// \param value The value to write, or `std::nullopt` to delete.
  /// \returns A Future that resolves to the generation corresponding to the new
  ///     value on success, or to `StorageGeneration::Unknown()` if the
  ///     conditions in `options` are not satisfied.
  virtual Future<TimestampedStorageGeneration> Write(Key key,
                                                     std::optional<Value> value,
                                                     WriteOptions options = {});

  /// Performs an optionally-conditional delete.
  ///
  /// Equivalent to calling `Write` with `value` equal to `std::nullopt`.
  Future<TimestampedStorageGeneration> Delete(Key key,
                                              WriteOptions options = {}) {
    return Write(key, std::nullopt, std::move(options));
  }

  /// Registers a transactional read-modify-write operation.
  ///
  /// Any actual reading will be deferred until requested by `source`, and any
  /// actual writing will be deferred until `transaction` is committed.
  ///
  /// The default implementation tracks the read-modify-write operations in a
  /// data structure but ultimately just calls `Driver::Read` and
  /// `Driver::Write`.  `Driver` implementations that use caching should
  /// override the default implementation.
  ///
  /// \param transaction[in,out] Specifies either an existing transaction or a
  ///     pointer to an `internal::TransactionState::ImplicitHandle` to be set
  ///     to the implicit transaction that was used.  In the case that no
  ///     existing transaction was specified, the default implementation always
  ///     creates a new implicit transaction, but `Driver` implementations that
  ///     use caching may return an existing implicit transaction with which
  ///     this write will be coalesced.
  /// \param key The key to write.
  /// \param source The write source.
  virtual absl::Status ReadModifyWrite(
      internal::OpenTransactionPtr& transaction, size_t& phase, Key key,
      ReadModifyWriteSource& source);

  /// Registers a transactional delete range operation.
  ///
  /// The actual deletion will not occur until the transaction is committed.
  ///
  /// The default implementation tracks the operation in a data structure but
  /// ultimately just calls the non-transactional `Driver::DeleteRange`.
  /// `Driver` implementations that support multi-key transactions should
  /// override the default implementation.
  ///
  /// \param transaction Open transaction in which to perform the operation.
  /// \param range Range of keys to delete.
  virtual absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction, KeyRange range);

  /// Deletes all keys in the specified range.
  ///
  /// This operation is not guaranteed to be atomic with respect to other
  /// operations affecting keys in `range`.  If there are concurrent writes to
  /// keys in `range`, this operation may fail with an error or indicate success
  /// despite not having removed the newly-added keys.
  ///
  /// \returns A Future that becomes ready when the operation has completed
  ///     either successfully or with an error.
  virtual Future<const void> DeleteRange(KeyRange range);

  /// Implementation of `List` that driver implementations must define.
  virtual void ListImpl(ListOptions options,
                        AnyFlowReceiver<absl::Status, Key> receiver);

  /// List keys in the key-value store.
  ///
  /// The keys are emitted in arbitrary order.
  ///
  /// This simply forwards to `ListImpl`.
  AnyFlowSender<absl::Status, Key> List(ListOptions options);

  /// Returns a Spec that can be used to re-open this key-value store.
  ///
  /// Options that modify the returned `Spec` may be specified in any order.
  ///
  /// Refer to `KvStore::spec` for details.
  ///
  /// \param option Any option compatible with `SpecRequestOptions`.
  /// \error `absl::StatusCode::kUnimplemented` if a JSON representation is not
  ///     supported.  (This behavior is provided by the default implementation.)
  template <typename... Option>
  std::enable_if_t<IsCompatibleOptionSequence<SpecRequestOptions, Option...>,
                   Result<DriverSpecPtr>>
  spec(Option&&... option) const {
    SpecRequestOptions options;
    (options.Set(std::move(option)), ...);
    return spec(std::move(options));
  }

  /// Returns a Spec that can be used to re-open this key-value store.
  ///
  /// \param option Options that may modify the returned `Spec`.
  Result<DriverSpecPtr> spec(SpecRequestOptions&& options) const;

  /// Encodes relevant state as a cache key.
  ///
  /// Typically this should be called indirectly via
  /// `tensorstore::internal::EncodeCacheKey`.
  ///
  /// The default implementation simply encodes the pointer value `this`, and is
  /// used for Driver implementations that do not support a JSON representation
  /// or are incompatible with the cache key mechanism.
  ///
  /// For drivers that do support a JSON representation, this is defined
  /// automatically by `internal_kvstore::RegisteredDriver` in `registry.h`.
  virtual void EncodeCacheKey(std::string* out) const;

  /// Returns a human-readable description of a key for use in error messages.
  ///
  /// By default, returns `QuoteString(key)`.
  virtual std::string DescribeKey(std::string_view key);

  /// Equivalent to
  /// `AnnotateErrorWithKeyDescription(DescribeKey(key), action, error)`.
  absl::Status AnnotateError(std::string_view key, std::string_view action,
                             const absl::Status& error);

  /// Annotates `error` with a message including `key_description`, which should
  /// normally be the result of a call to `DescribeKey`.
  ///
  /// To avoid redundancy, if applied to an error message that already includes
  /// `key_description`, simply returns `error` unmodified.
  ///
  /// \param key_description Description of the key associated with the error.
  /// \param action Action, e.g. "reading" or "writing".
  /// \param error The error to annotate.
  static absl::Status AnnotateErrorWithKeyDescription(
      std::string_view key_description, std::string_view action,
      const absl::Status& error);

  /// Returns a Spec that can be used to re-open this key-value store.
  ///
  /// Returns `absl::StatusCode::kUnimplemented` if a JSON representation is not
  /// supported.  (This behavior is provided by the default implementation.)
  ///
  /// For drivers that do support a JSON representation, this is defined
  /// automatically by `internal_kvstore::RegisteredDriver` in `registry.h`.
  virtual Result<DriverSpecPtr> GetBoundSpec() const;

  virtual void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const = 0;

  virtual ~Driver();

  // Treat as private: Identifier in the open kvstore cache.
  std::string cache_identifier_;

 private:
  void DestroyLastReference();

  friend void intrusive_ptr_increment(Driver* p);
  friend void intrusive_ptr_decrement(Driver* p);

  std::atomic<size_t> reference_count_{0};
};

/// Opens a `Driver` based on an already-parsed `DriverSpec`.
///
/// \param spec Driver specification.
/// \param options Options for opening the spec.
Future<DriverPtr> Open(DriverSpecPtr spec, DriverOpenOptions&& options);

/// Opens a `Driver` based on an already-parsed `DriverSpec` and an optional
/// sequence of options.
///
/// Options may be specified in any order, and are identified by their type.
/// Supported option types are:
///
/// - Context: specifies the context in which to obtain any unbound context
///   resources in `spec`.  Any already-bound context resources in `spec`
///   remain unmodified.  If not specified, `Context::Default()` is used.
///
/// \param spec Driver specification.
/// \param option Any option compatible with `DriverOpenOptions`.
template <typename... Option>
static std::enable_if_t<
    IsCompatibleOptionSequence<DriverOpenOptions, Option...>, Future<DriverPtr>>
Open(DriverSpecPtr spec, Option&&... option) {
  DriverOpenOptions options;
  (options.Set(option), ...);
  return Open(std::move(spec), std::move(options));
}

}  // namespace kvstore
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::kvstore::DriverPtr)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Driver)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr)

#endif  // TENSORSTORE_KVSTORE_DRIVER_H_
