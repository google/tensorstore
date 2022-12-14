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

#include "tensorstore/driver/driver_testutil.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/sync_flow_sender.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

namespace {
void TestMinimalSpecRoundTrips(
    Context context, const TestTensorStoreDriverSpecRoundtripOptions& options,
    Transaction transaction) {
  // Test that the minimal spec round trips for opening existing TensorStore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2, tensorstore::Open(options.minimal_spec, context, transaction,
                                     tensorstore::OpenMode::open)
                       .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto full_spec_obj2,
      store2.spec(SpecRequestOptions(options.spec_request_options)));
  EXPECT_THAT(full_spec_obj2.ToJson(options.to_json_options),
              ::testing::Optional(MatchesJson(options.full_spec)));
  auto minimal_spec_request_options = options.spec_request_options;
  TENSORSTORE_ASSERT_OK(
      minimal_spec_request_options.Set(tensorstore::MinimalSpec{true}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto minimal_spec_obj2,
      store2.spec(std::move(minimal_spec_request_options)));
  EXPECT_THAT(minimal_spec_obj2.ToJson(options.to_json_options),
              ::testing::Optional(MatchesJson(options.minimal_spec)));
}

// Tests that the full Spec round trips for creating a new TensorStore.
void TestTensorStoreDriverSpecRoundtrip(
    TestTensorStoreDriverSpecRoundtripOptions options, TransactionMode mode) {
  Transaction transaction(mode);
  auto context = Context::Default();
  if (options.check_not_found_before_create) {
    EXPECT_THAT(tensorstore::Open(options.minimal_spec, context,
                                  tensorstore::OpenMode::open)
                    .result(),
                MatchesStatus(absl::StatusCode::kNotFound));
  }

  // Use separate block to ensure `store` does not remain cached in the
  // `context`.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto store, tensorstore::Open(options.full_spec, context, transaction,
                                      tensorstore::OpenMode::create)
                        .result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto full_spec_obj,
        store.spec(SpecRequestOptions(options.spec_request_options)));
    EXPECT_THAT(full_spec_obj.ToJson(options.to_json_options),
                ::testing::Optional(MatchesJson(options.full_spec)));
    auto minimal_spec_request_options = options.spec_request_options;
    TENSORSTORE_ASSERT_OK(
        minimal_spec_request_options.Set(tensorstore::MinimalSpec{true}));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto minimal_spec_obj,
        store.spec(std::move(minimal_spec_request_options)));
    EXPECT_THAT(minimal_spec_obj.ToJson(options.to_json_options),
                ::testing::Optional(MatchesJson(options.minimal_spec)));

    if (options.write_value_to_create) {
      // Write a single value to `store` to ensure it is created.
      TENSORSTORE_ASSERT_OK(
          tensorstore::Write(tensorstore::AllocateArray(
                                 span<const Index>(), tensorstore::c_order,
                                 tensorstore::value_init, store.dtype()),
                             store | tensorstore::AllDims().IndexSlice(
                                         store.domain().origin()))
              .result());
    }
  }
  if (mode != no_transaction) {
    if (options.check_transactional_open_before_commit) {
      // Test that the minimal spec round trips before the transaction is
      // committed.
      TestMinimalSpecRoundTrips(context, options, transaction);
    }
    if (options.check_not_found_before_commit) {
      // Test that the minimal spec cannot be opened outside the transaction
      // before the transaction is committed.
      EXPECT_THAT(tensorstore::Open(options.minimal_spec, context,
                                    tensorstore::OpenMode::open)
                      .result(),
                  MatchesStatus(absl::StatusCode::kNotFound));
    }
    TENSORSTORE_EXPECT_OK(transaction.CommitAsync().result());
  }
  // Test that the minimal spec round trips outside the transaction (if any).
  TestMinimalSpecRoundTrips(context, options, no_transaction);
}
}  // namespace

void RegisterTensorStoreDriverSpecRoundtripTest(
    TestTensorStoreDriverSpecRoundtripOptions options) {
  if (options.create_spec.is_discarded()) {
    options.create_spec = options.full_spec;
  }
  const auto RegisterVariant = [&](TransactionMode mode) {
    internal::RegisterGoogleTestCaseDynamically(
        "TensorStoreDriverSpecRoundtripTest",
        tensorstore::StrCat(options.test_name, "/transaction_mode=", mode),
        [=] { TestTensorStoreDriverSpecRoundtrip(options, mode); },
        TENSORSTORE_LOC);
  };
  RegisterVariant(no_transaction);
  for (auto transaction_mode : options.supported_transaction_modes) {
    RegisterVariant(transaction_mode);
  }
}

void TestTensorStoreDriverSpecConvertImpl(
    ::nlohmann::json orig_spec, ::nlohmann::json expected_converted_spec,
    SpecConvertOptions&& options) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_obj,
                                   tensorstore::Spec::FromJson(orig_spec));
  TENSORSTORE_ASSERT_OK(spec_obj.Set(std::move(options)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto converted_spec,
      spec_obj.ToJson(tensorstore::IncludeDefaults{false}));
  EXPECT_THAT(converted_spec, MatchesJson(expected_converted_spec));
}

namespace {

// TODO(jbms): Consider migrating to/unifying this with
// `MakeRandomIndexTransform` in `index_space/index_transform_testutil.h`.
IndexTransform<> GetRandomTransform(absl::BitGenRef gen,
                                    IndexDomainView<> domain) {
  auto transform = IdentityTransform(domain);
  const auto ApplyExpression = [&](auto e) {
    transform = (transform | e).value();
  };
  for (DimensionIndex i = 0; i < domain.rank(); ++i) {
    if (domain[i].empty()) continue;
    switch (absl::Uniform(absl::IntervalClosedClosed, gen, 0, 1)) {
      case 0: {
        const Index start =
            absl::Uniform(absl::IntervalClosedOpen, gen,
                          domain[i].inclusive_min(), domain[i].exclusive_max());
        const Index stop = absl::Uniform(absl::IntervalOpenClosed, gen, start,
                                         domain[i].exclusive_max());
        const Index stride =
            absl::Uniform(absl::IntervalClosedClosed, gen, Index(1), Index(3));
        ApplyExpression(Dims(i).HalfOpenInterval(start, stop, stride));
        if (absl::Bernoulli(gen, 0.5)) {
          ApplyExpression(Dims(i).HalfOpenInterval(kImplicit, kImplicit, -1));
        }
        break;
      }
      case 1: {
        std::vector<Index> values(domain[i].size());
        absl::c_iota(values, domain[i].inclusive_min());
        std::minstd_rand derived_rng(absl::Uniform<uint32_t>(gen));
        absl::c_shuffle(values, derived_rng);
        values.resize(absl::Uniform(absl::IntervalClosedClosed, gen,
                                    std::size_t(1), values.size()));
        ApplyExpression(
            Dims(i).OuterIndexArraySlice(MakeCopy(MakeArrayView((values)))));
        break;
      }
    }
  }
  return transform;
}

}  // namespace

DriverRandomOperationTester::DriverRandomOperationTester(
    absl::BitGenRef gen, TensorStoreDriverBasicFunctionalityTestOptions options)
    : gen(gen), options(std::move(options)) {}

void DriverRandomOperationTester::TestBasicFunctionality(
    TransactionMode transaction_mode, size_t num_iterations) {
  SCOPED_TRACE(tensorstore::StrCat("create_spec=", options.create_spec));
  Transaction transaction(transaction_mode);
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(options.create_spec, context, transaction,
                                    tensorstore::OpenMode::create)
                      .result());
  ASSERT_EQ(options.expected_domain, store.domain());
  ASSERT_EQ(options.initial_value.dtype(), store.dtype());
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto resolved_store,
                                     ResolveBounds(store).result());
    EXPECT_EQ(options.expected_domain, resolved_store.domain());
  }
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto resolved_store,
        ResolveBounds(store, fix_resizable_bounds).result());
    EXPECT_EQ((IdentityTransform(options.expected_domain) |
               tensorstore::AllDims().MarkBoundsExplicit())
                  .value()
                  .domain(),
              resolved_store.domain());
  }

  // Read fill value.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result, Read(store).result());
    SCOPED_TRACE("Compare to fill value");
    options.compare_arrays(options.initial_value, read_result);
  }

  auto expected_value = MakeCopy(options.initial_value);

  constexpr auto kMaxWaitDuration = absl::Seconds(20);

  for (std::size_t i = 0; i < num_iterations; ++i) {
    auto transform = GetRandomTransform(gen, options.expected_domain);
    auto random_array = MakeRandomArray(gen, transform.domain().box(),
                                        options.initial_value.dtype());
    if (log) TENSORSTORE_LOG("i = ", i);
    SCOPED_TRACE(tensorstore::StrCat("i=", i));
    SCOPED_TRACE(tensorstore::StrCat("transform=", transform));
    SCOPED_TRACE(tensorstore::StrCat("original_domain=",
                                     options.initial_value.domain()));
    auto write_future =
        tensorstore::Write(random_array, store | transform).commit_future;
    TENSORSTORE_ASSERT_OK(write_future.result());
    ASSERT_TRUE(write_future.WaitFor(kMaxWaitDuration));
    TENSORSTORE_ASSERT_OK(write_future.result());
    auto read_part_future = Read(store | transform);
    ASSERT_TRUE(read_part_future.WaitFor(kMaxWaitDuration));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_part_result,
                                     read_part_future.result());
    {
      SCOPED_TRACE("Compare read result to random array");
      options.compare_arrays(random_array, read_part_result);
    }

    TENSORSTORE_ASSERT_OK(CopyTransformedArray(
        random_array, (expected_value | transform).value()));

    auto read_full_future = Read(store);
    ASSERT_TRUE(read_full_future.WaitFor(kMaxWaitDuration));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_full_result,
                                     read_full_future.result());

    {
      SCOPED_TRACE("Compare full read result to expected value");
      options.compare_arrays(expected_value, read_full_result);
    }
  }

  if (transaction != no_transaction) {
    if (options.check_not_found_before_commit) {
      EXPECT_THAT(tensorstore::Open(options.create_spec, context,
                                    tensorstore::OpenMode::open)
                      .result(),
                  MatchesStatus(absl::StatusCode::kNotFound));
    }
    ASSERT_FALSE(transaction.commit_started());
    EXPECT_THAT(store | no_transaction,
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Cannot rebind transaction when existing "
                              "transaction is uncommitted"));
    TENSORSTORE_ASSERT_OK(transaction.CommitAsync().result());
    EXPECT_THAT(Read(store).result(),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Transaction not open"));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto non_transactional_store,
        tensorstore::Open(options.create_spec, context,
                          tensorstore::OpenMode::open)
            .result());

    {
      SCOPED_TRACE("Re-opened: Compare full read result to expected value");
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_full_result,
                                       Read(non_transactional_store).result());
      options.compare_arrays(expected_value, read_full_result);
    }

    {
      SCOPED_TRACE(
          "Switched to non-transacitonal: Compare full read result to "
          "expected value");
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_full_result,
                                       Read(store | no_transaction).result());
      options.compare_arrays(expected_value, read_full_result);
    }

    {
      SCOPED_TRACE(
          "Switched to new transaction: Compare full read result to "
          "expected value");
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto read_full_result,
          Read(store | Transaction(transaction.mode())).result());
      options.compare_arrays(expected_value, read_full_result);
    }
  }
}

void DriverRandomOperationTester::TestMultiTransactionWrite(
    TransactionMode mode, size_t num_transactions, size_t num_iterations,
    bool use_random_values) {
  SCOPED_TRACE(tensorstore::StrCat("create_spec=", options.create_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(options.create_spec, tensorstore::OpenMode::create)
          .result());
  // transactions[0] is a null transaction.
  std::vector<Transaction> transactions(num_transactions, no_transaction);
  for (size_t i = 1; i < num_transactions; ++i) {
    transactions[i] = Transaction(mode);
  }
  struct WriteEntry {
    IndexTransform<> transform;
    SharedOffsetArray<const void> array;
  };
  std::vector<std::vector<WriteEntry>> transaction_entries(num_transactions);

  for (size_t i = 0; i < num_iterations; ++i) {
    auto transform = GetRandomTransform(gen, options.expected_domain);
    SharedOffsetArray<const void> array;
    if (use_random_values) {
      array = MakeRandomArray(gen, transform.domain().box(),
                              options.initial_value.dtype());
    } else {
      array =
          MakeCopy(
              SharedOffsetArray<const void>(
                  std::make_shared<size_t>(i + 1),
                  StridedLayout<dynamic_rank, offset_origin>(
                      transform.domain().box(),
                      GetConstantVector<Index, 0>(transform.domain().rank()))),
              skip_repeated_elements, options.initial_value.dtype())
              .value();
    }
    size_t transaction_i = i % num_transactions;
    transaction_entries[transaction_i].push_back(WriteEntry{transform, array});
    TENSORSTORE_ASSERT_OK(
        Write(array, store | transactions[transaction_i] | transform).result());
  }

  for (size_t i = 1; i < num_transactions; ++i) {
    TENSORSTORE_ASSERT_OK(transactions[i].CommitAsync().result());
  }

  auto expected_value = MakeCopy(options.initial_value);
  for (const auto& entries : transaction_entries) {
    for (const auto& entry : entries) {
      TENSORSTORE_ASSERT_OK(CopyTransformedArray(
          entry.array, (expected_value | entry.transform).value()));
    }
  }
  {
    SCOPED_TRACE("Compare full read result to expected value");
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_full_result,
                                     Read(store).result());
    options.compare_arrays(expected_value, read_full_result);
  }
}

void RegisterTensorStoreDriverBasicFunctionalityTest(
    TensorStoreDriverBasicFunctionalityTestOptions options) {
  if (!options.compare_arrays) {
    options.compare_arrays = [](auto expected, auto actual) {
      EXPECT_EQ(expected, actual);
    };
  }

  // NOTE: The test seeds are a bit verbose; we should make them less so.

  const auto RegisterVariant = [&](TransactionMode mode,
                                   size_t num_iterations) {
    internal::RegisterGoogleTestCaseDynamically(
        "TensorStoreDriverBasicFunctionalityTest",
        tensorstore::StrCat(options.test_name, "/basic_functionality",
                            "/transaction_mode=", mode,
                            "/num_iterations=", num_iterations),
        [=] {
          std::minstd_rand gen{internal::GetRandomSeedForTest(
              "TENSORSTORE_INTERNAL_DRIVER_BASIC_FUNCTIONALITY")};
          DriverRandomOperationTester tester(gen, std::move(options));
          tester.TestBasicFunctionality(mode, num_iterations);
        },
        TENSORSTORE_LOC);
  };
  RegisterVariant(no_transaction, 20);
  for (auto transaction_mode : options.supported_transaction_modes) {
    RegisterVariant(transaction_mode, /*num_iterations=*/5);
    const auto RegisterMultiTransaction = [&](size_t num_transactions,
                                              size_t num_iterations,
                                              bool use_random_values) {
      internal::RegisterGoogleTestCaseDynamically(
          "TensorStoreDriverBasicFunctionalityTest",
          tensorstore::StrCat(options.test_name, "/multi_transaction_write",
                              "/transaction_mode=", transaction_mode,
                              "/num_transactions=", num_transactions,
                              "/num_iterations=", num_iterations,
                              "/use_random_values=", use_random_values),
          [=] {
            std::minstd_rand gen{internal::GetRandomSeedForTest(
                "TENSORSTORE_INTERNAL_DRIVER_MULTI_TRANSACTION")};
            DriverRandomOperationTester tester(gen, std::move(options));
            tester.TestMultiTransactionWrite(transaction_mode, num_transactions,
                                             num_iterations, use_random_values);
          },
          TENSORSTORE_LOC);
    };
    RegisterMultiTransaction(/*num_transactions=*/2, /*num_iterations=*/3,
                             /*use_random_values=*/false);
    RegisterMultiTransaction(/*num_transactions=*/3, /*num_iterations=*/20,
                             /*use_random_values=*/true);
  }
}

namespace {
void TestMetadataOnlyResize(const TestTensorStoreDriverResizeOptions& options,
                            TransactionMode mode) {
  Transaction transaction(mode);
  Box<> bounds(options.initial_bounds);
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(options.get_create_spec(bounds), context,
                                    tensorstore::OpenMode::create)
                      .result());
  const auto initial_domain = store.domain();
  const DimensionIndex rank = initial_domain.rank();
  std::vector<DimensionIndex> resizable_bounds;
  for (DimensionIndex i = 0; i < rank; ++i) {
    if (initial_domain.implicit_lower_bounds()[i]) {
      resizable_bounds.push_back(i);
    }
    if (initial_domain.implicit_upper_bounds()[i]) {
      resizable_bounds.push_back(i + rank);
    }
  }

  // Test no-op resize.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto noop_resized,
        tensorstore::Resize(
            store | transaction,
            tensorstore::GetConstantVector<Index, kImplicit>(rank),
            tensorstore::GetConstantVector<Index, kImplicit>(rank),
            tensorstore::resize_metadata_only)
            .result());
    EXPECT_EQ(initial_domain, noop_resized.domain());
  }

  IndexDomain<> expected_domain;
  for (auto bound_i : resizable_bounds) {
    const DimensionIndex dim_i = bound_i % rank;
    const bool is_upper = static_cast<bool>(bound_i / rank);
    const Index existing_value = is_upper ? bounds[dim_i].exclusive_max()
                                          : bounds[dim_i].inclusive_min();
    for (Index offset : {-1, 1}) {
      std::vector<Index> lower_and_upper_bound_changes[2] = {
          std::vector<Index>(rank, kImplicit),
          std::vector<Index>(rank, kImplicit)};
      std::vector<Index> lower_and_upper_bounds[2] = {std::vector<Index>(rank),
                                                      std::vector<Index>(rank)};
      for (DimensionIndex j = 0; j < rank; ++j) {
        lower_and_upper_bounds[0][j] = bounds[j].inclusive_min();
        lower_and_upper_bounds[1][j] = bounds[j].exclusive_max();
      }
      lower_and_upper_bounds[is_upper][dim_i] =
          lower_and_upper_bound_changes[is_upper][dim_i] =
              existing_value + offset;
      SCOPED_TRACE(tensorstore::StrCat(
          "dim_i=", dim_i, ", is_upper=", is_upper,
          ", new_inclusive_min=", span(lower_and_upper_bound_changes[0]),
          ", new_exclusive_max=", span(lower_and_upper_bound_changes[1])));
      for (DimensionIndex j = 0; j < rank; ++j) {
        bounds[j] = IndexInterval::UncheckedHalfOpen(
            lower_and_upper_bounds[0][j], lower_and_upper_bounds[1][j]);
      }
      // Verify that specifying an unsatisfied `expand_only` or `shrink_only`
      // constraint leads to an error.
      EXPECT_THAT(tensorstore::Resize(store, lower_and_upper_bound_changes[0],
                                      lower_and_upper_bound_changes[1],
                                      ((offset > 0) == is_upper)
                                          ? ResizeMode::shrink_only
                                          : ResizeMode::expand_only)
                      .result(),
                  MatchesStatus(absl::StatusCode::kFailedPrecondition));
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto resized_store,
          tensorstore::Resize(
              store | transaction, lower_and_upper_bound_changes[0],
              lower_and_upper_bound_changes[1], resize_metadata_only)
              .result());
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          expected_domain, IndexDomainBuilder(rank)
                               .domain(initial_domain)
                               .origin(lower_and_upper_bounds[0])
                               .exclusive_max(lower_and_upper_bounds[1])
                               .Finalize());
      EXPECT_EQ(expected_domain, resized_store.domain());
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto resolved_store,
          tensorstore::ResolveBounds(store | transaction).result());
      EXPECT_EQ(expected_domain, resolved_store.domain());
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_obj, resized_store.spec());
      EXPECT_THAT(
          spec_obj.ToJson(IncludeDefaults{false}),
          ::testing::Optional(MatchesJson(options.get_create_spec(bounds))));

      // Verify that re-opening gives the new size.
      TENSORSTORE_ASSERT_OK_AND_ASSIGN(
          auto reopened_store,
          tensorstore::Open(options.get_create_spec(bounds), context,
                            transaction, tensorstore::OpenMode::open)
              .result());
      EXPECT_EQ(expected_domain, reopened_store.domain());

      // Verify that resolving bounds without the transaction gives the old
      // size.
      if (transaction != no_transaction) {
        TENSORSTORE_ASSERT_OK_AND_ASSIGN(
            auto resolved_non_transactional,
            tensorstore::ResolveBounds(store).result());
        EXPECT_EQ(initial_domain, resolved_non_transactional.domain());
      }
    }
  }
  if (transaction != no_transaction) {
    TENSORSTORE_ASSERT_OK(transaction.CommitAsync().result());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto resolved_non_transactional,
        tensorstore::ResolveBounds(store).result());
    EXPECT_EQ(expected_domain, resolved_non_transactional.domain());
  }
}
}  // namespace

void RegisterTensorStoreDriverResizeTest(
    TestTensorStoreDriverResizeOptions options) {
  const auto RegisterVariant = [&](TransactionMode mode) {
    internal::RegisterGoogleTestCaseDynamically(
        "TensorStoreDriverResizeTest",
        tensorstore::StrCat(options.test_name, "/transaction_mode=", mode),
        [=] { TestMetadataOnlyResize(options, mode); }, TENSORSTORE_LOC);
  };
  RegisterVariant(no_transaction);
  for (auto transaction_mode : options.supported_transaction_modes) {
    RegisterVariant(transaction_mode);
  }
}

Future<std::vector<std::pair<SharedOffsetArray<void>, IndexTransform<>>>>
ReadAsIndividualChunks(TensorStore<> store) {
  using ChunkPair = std::pair<SharedOffsetArray<void>, IndexTransform<>>;
  using ChunkVec = std::vector<ChunkPair>;
  struct ReceiverImpl {
    Promise<ChunkVec> promise_;
    DataType dtype_;
    FutureCallbackRegistration cancel_registration_;
    void set_starting(AnyCancelReceiver cancel) {
      cancel_registration_ = promise_.ExecuteWhenNotNeeded(std::move(cancel));
    }
    void set_value(ReadChunk chunk, IndexTransform<> request_transform) {
      auto array = AllocateArray(chunk.transform.domain().box(), c_order,
                                 default_init, dtype_);
      auto& r = promise_.raw_result();
      if (!r.ok()) return;
      r->emplace_back(array, request_transform);
      TENSORSTORE_RETURN_IF_ERROR(
          internal::CopyReadChunk(chunk.impl, std::move(chunk.transform),
                                  TransformedArray(array)),
          this->set_error(_));
    }
    void set_done() { promise_ = {}; }
    void set_error(absl::Status error) {
      promise_.SetResult(std::move(error));
      promise_ = {};
    }
    void set_stopping() { cancel_registration_.Unregister(); }
  };
  auto [promise, future] = PromiseFuturePair<ChunkVec>::Make(std::in_place);
  auto transformed_driver = TensorStoreAccess::handle(store);
  TENSORSTORE_ASSIGN_OR_RETURN(auto transaction,
                               internal::AcquireOpenTransactionPtrOrError(
                                   transformed_driver.transaction));
  transformed_driver.driver->Read(
      transaction, transformed_driver.transform,
      SyncFlowReceiver<ReceiverImpl>{ReceiverImpl{
          std::move(promise), transformed_driver.driver->dtype()}});
  return future;
}

Future<std::vector<std::pair<ReadChunk, IndexTransform<>>>> CollectReadChunks(
    TensorStore<> store) {
  using ChunkPair = std::pair<ReadChunk, IndexTransform<>>;
  using ChunkVec = std::vector<ChunkPair>;
  struct ReceiverImpl {
    Promise<ChunkVec> promise_;
    FutureCallbackRegistration cancel_registration_;
    void set_starting(AnyCancelReceiver cancel) {
      cancel_registration_ = promise_.ExecuteWhenNotNeeded(std::move(cancel));
    }
    void set_value(ReadChunk chunk, IndexTransform<> request_transform) {
      auto& r = promise_.raw_result();
      if (!r.ok()) return;
      r->emplace_back(std::move(chunk), request_transform);
    }
    void set_done() { promise_ = {}; }
    void set_error(absl::Status error) {
      promise_.SetResult(std::move(error));
      promise_ = {};
    }
    void set_stopping() { cancel_registration_.Unregister(); }
  };
  auto [promise, future] = PromiseFuturePair<ChunkVec>::Make(std::in_place);
  auto transformed_driver = TensorStoreAccess::handle(store);
  TENSORSTORE_ASSIGN_OR_RETURN(auto transaction,
                               internal::AcquireOpenTransactionPtrOrError(
                                   transformed_driver.transaction));
  transformed_driver.driver->Read(
      transaction, transformed_driver.transform,
      SyncFlowReceiver<ReceiverImpl>{ReceiverImpl{std::move(promise)}});
  return future;
}

void MockDriver::Read(internal::OpenTransactionPtr transaction,
                      IndexTransform<> transform, ReadChunkReceiver receiver) {
  read_requests.push(ReadRequest{std::move(transaction), std::move(transform),
                                 std::move(receiver)});
}

void MockDriver::Write(internal::OpenTransactionPtr transaction,
                       IndexTransform<> transform,
                       WriteChunkReceiver receiver) {
  write_requests.push(WriteRequest{std::move(transaction), std::move(transform),
                                   std::move(receiver)});
}

void MockDriver::GarbageCollectionVisit(
    garbage_collection::GarbageCollectionVisitor& visitor) const {
  // No-op
}

TensorStore<> MockDriver::Wrap(IndexTransform<> transform) {
  if (!transform.valid()) transform = IdentityTransform(rank_);
  return TensorStoreAccess::Construct<TensorStore<>>(
      tensorstore::internal::Driver::Handle{
          tensorstore::internal::DriverPtr(this, ReadWriteMode::read_write),
          std::move(transform)});
}

ReadChunk MakeArrayBackedReadChunk(TransformedArray<Shared<const void>> data) {
  /// Implementation of the `ReadChunk::Impl` Poly interface.
  struct ReadChunkImpl {
    SharedElementPointer<const void> data;
    absl::Status operator()(LockCollection& lock_collection) {
      return absl::OkStatus();
    }

    Result<NDIterable::Ptr> operator()(ReadChunk::BeginRead,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena) {
      return GetTransformedArrayNDIterable({data, chunk_transform}, arena);
    }
  };
  ReadChunk chunk;
  chunk.impl = ReadChunkImpl{data.element_pointer()};
  chunk.transform = data.transform();
  return chunk;
}

void TestTensorStoreCreateWithSchemaImpl(::nlohmann::json json_spec,
                                         const Schema& schema) {
  SCOPED_TRACE(tensorstore::StrCat("json=", json_spec));
  SCOPED_TRACE(tensorstore::StrCat("schema=", schema));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(json_spec, OpenMode::create, schema).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_schema, store.schema());
  TENSORSTORE_ASSERT_OK(store_schema.Set(schema));
}

void TestTensorStoreCreateCheckSchemaImpl(::nlohmann::json json_spec,
                                          const Schema& schema) {
  SCOPED_TRACE(tensorstore::StrCat("json=", json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store, tensorstore::Open(json_spec, OpenMode::create).result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_schema, store.schema());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto store_schema_json,
                                   store_schema.ToJson());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema_json, schema.ToJson());
  EXPECT_THAT(store_schema_json, MatchesJson(schema_json));
}

void TestTensorStoreCreateCheckSchema(::nlohmann::json json_spec,
                                      ::nlohmann::json json_schema) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, Schema::FromJson(json_schema));
  TestTensorStoreCreateCheckSchema(std::move(json_spec), schema);
}

void TestSpecSchemaImpl(::nlohmann::json json_spec, const Schema& schema) {
  SCOPED_TRACE(tensorstore::StrCat("json=", json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec,
                                   tensorstore::Spec::FromJson(json_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_schema, spec.schema());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_schema_json, spec_schema.ToJson());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema_json, schema.ToJson());
  EXPECT_THAT(spec_schema_json, MatchesJson(schema_json));
}

void TestSpecSchema(::nlohmann::json json_spec, ::nlohmann::json json_schema) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, Schema::FromJson(json_schema));
  TestSpecSchemaImpl(std::move(json_spec), schema);
}

}  // namespace internal
}  // namespace tensorstore
