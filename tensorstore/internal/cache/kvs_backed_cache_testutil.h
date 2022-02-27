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

#ifndef TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_TESTUTIL_H_
#define TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_TESTUTIL_H_

/// \file
///
/// Defines `KvsBackedTestCache`, a test cache derived from `KvsBackedCache`,
/// which is used to test both `KvsBackedCache` itself as well as
/// `KeyValueStore` implementations.

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/random/bit_gen_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/transaction_impl.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {


/// Cache that may be used for testing `KvsBackedCache` and transactional
/// `KeyValueStore` operations.
///
/// This class simply caches the stored value directly; the encoding and
/// decoding operations are identity transforms.  A missing value in the
/// `KeyValueStore` is cached as an empty string.
class KvsBackedTestCache
    : public KvsBackedCache<KvsBackedTestCache, AsyncCache> {
  using Base = KvsBackedCache<KvsBackedTestCache, AsyncCache>;

 public:
  using Base::Base;

  /// For this test cache, encoding and decoding are simply identity transforms.
  using ReadData = absl::Cord;

  /// Function that is called on the existing read value to validate it.
  using Validator = std::function<absl::Status(const absl::Cord& input)>;

  class TransactionNode;

  class Entry : public Base::Entry {
   public:
    using OwningCache = KvsBackedTestCache;

    /// Applies a modification within a transaction.
    ///
    /// \param transaction The transaction to use, or `nullptr` to use an
    ///     implicit transaction.
    /// \param clear If `true`, clear (i.e. truncate to 0 length) the existing
    ///     value.  This takes effect before `append_value`.
    /// \param append_value String to append, or empty to not append anything.
    Result<OpenTransactionNodePtr<TransactionNode>> Modify(
        const OpenTransactionPtr& transaction, bool clear,
        std::string_view append_value);

    /// Ensures that the transaction is not committed unless the value satisfies
    /// the specified validator function.
    ///
    /// The validator is applied to the value with any previously requested
    /// modifications applied, but without any modifications requested after
    /// calling this function.
    ///
    /// The transaction will be aborted if the validator returns an error.
    ///
    /// This is intended to simulate a form of "consistent read" behavior.
    Result<OpenTransactionNodePtr<TransactionNode>> Validate(
        const OpenTransactionPtr& transaction, Validator validator);

    /// Returns the current value, with any previously requested modifications
    /// applied.
    Future<absl::Cord> ReadValue(
        OpenTransactionPtr transaction,
        absl::Time staleness_bound = absl::InfinitePast());

   private:
    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override;

    void DoEncode(std::shared_ptr<const absl::Cord> data,
                  EncodeReceiver receiver) override;
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using Base::TransactionNode::TransactionNode;
    using OwningCache = KvsBackedTestCache;
    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;
    std::vector<Validator> validators;
    bool cleared = false;
    std::string value;

    bool IsUnconditional() { return validators.empty() && cleared; }
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  static CachePtr<KvsBackedTestCache> Make(
      kvstore::DriverPtr kvstore, CachePool::StrongPtr pool = {},
      std::string_view cache_identifier = {});
};

struct KvsBackedCacheBasicTransactionalTestOptions {
  /// Name to use when registering the test suite.  Must be unique within the
  /// test binary.  The name should include the driver name, and if
  /// `RegisterKvsBackedCacheBasicTransactionalTest` is called multiple times
  /// for the driver,the name should also include an additional unique
  /// identifier.
  std::string test_name;

  /// Returns a new instance of the `KeyValueStore` to test.  The returned
  /// `KeyValueStore` should be empty, and if this function is called multiple
  /// times, the returned `KeyValueStore` objects should not share underlying
  /// storage.
  std::function<kvstore::DriverPtr()> get_store;

  /// Returns a function that maps arbitrary strings to unique valid keys.
  ///
  /// Multiple calls to `get_key_getter` need not return equivalent mapping
  /// functions, but any given mapping function returned by a call to
  /// `get_key_getter` must return a consistent, unique key for any given input
  /// string that is valid for use with a `KeyValueStore` returned by
  /// `get_store`.
  ///
  /// In most cases the mapping function can be the identity function, but a
  /// different mapping may be used if the `KeyValueStore` returned by
  /// `get_store` requires keys in a particular format, such as 8-byte values
  /// encoding 64-bit integers.
  std::function<std::function<std::string(std::string)>()> get_key_getter = [] {
    return [](std::string key) { return key; };
  };

  /// Indicates whether `TransactionalDeleteRange` is supported by the
  /// `KeyValueStore` returned by `get_store`.  If `false`, tests that depend on
  /// it will not be included.
  bool delete_range_supported = true;

  /// Indicates whether the `KeyValueStore` returned by `get_store` supports
  /// atomic transactions involving more than one key.  If `false`, tests that
  /// depend on multi-key atomic transactions will not be included.
  bool multi_key_atomic_supported = true;
};

/// Registers a test suite for transactional KeyValueStore operations.
///
/// This should be used for each KeyValueStore driver that defines custom
/// support for transactional operations, in addition to the standard
/// non-transactional tests declared in `key_value_store_testutil.h`.
///
/// For drivers that rely on the default support for single-key transactional
/// operations, the tests in `kvs_backed_cache_test.cc` against the "memory"
/// KeyValueStore should be sufficient.
void RegisterKvsBackedCacheBasicTransactionalTest(
    const KvsBackedCacheBasicTransactionalTestOptions& options);

/// KvsRandomOperationTester implements random/fuzz based testing for
/// KeyValueStore. The absl::BitGenRef must outlive the
/// KvsRandomOperationTester.
class KvsRandomOperationTester {
 public:
  using Map = std::map<std::string, std::string>;

  explicit KvsRandomOperationTester(
      absl::BitGenRef gen, kvstore::DriverPtr kvstore,
      std::function<std::string(std::string)> get_key);

  void SimulateDeleteRange(const KeyRange& range);

  void SimulateWrite(const std::string& key, bool clear,
                     const std::string& append);

  std::string SampleKey();
  std::string SampleKeyOrEmpty();

  void PerformRandomAction(OpenTransactionPtr transaction);

  void PerformRandomActions();

  absl::BitGenRef gen;  // Not owned.
  kvstore::DriverPtr kvstore;
  Map map;

  std::vector<std::string> keys;
  std::vector<CachePtr<KvsBackedTestCache>> caches;

  double write_probability = 0.8;
  double clear_probability = 0.5;
  double barrier_probability = 0.05;
  size_t write_number = 0;
  bool log = true;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CACHE_KVS_BACKED_CACHE_TESTUTIL_H_
