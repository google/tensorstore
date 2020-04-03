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
#include <nlohmann/json.hpp>
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal {

void TestTensorStoreDriverSpecRoundtrip(::nlohmann::json full_spec,
                                        ::nlohmann::json minimal_spec,
                                        ContextToJsonOptions options) {
  // Test that the full Spec round trips for creating a new TensorStore.
  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(context, full_spec, tensorstore::OpenMode::create)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto full_spec_obj, store.spec());
  EXPECT_THAT(full_spec_obj.ToJson(options), ::testing::Optional(full_spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto minimal_spec_obj,
                                   store.spec(tensorstore::MinimalSpec{true}));
  EXPECT_THAT(minimal_spec_obj.ToJson(options),
              ::testing::Optional(minimal_spec));

  // Test that the minimal spec round trips for opening existing TensorStore.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store2,
      tensorstore::Open(context, minimal_spec, tensorstore::OpenMode::open)
          .result());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto full_spec_obj2, store2.spec());
  EXPECT_THAT(full_spec_obj2.ToJson(options), ::testing::Optional(full_spec));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto minimal_spec_obj2,
                                   store2.spec(tensorstore::MinimalSpec{true}));
  EXPECT_THAT(minimal_spec_obj2.ToJson(options),
              ::testing::Optional(minimal_spec));
}

void TestTensorStoreDriverSpecConvert(
    ::nlohmann::json orig_spec, const SpecRequestOptions& options,
    ::nlohmann::json expected_converted_spec) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto spec_obj,
                                   tensorstore::Spec::FromJson(orig_spec));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto converted_spec_obj,
                                   spec_obj.Convert(options));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto converted_spec,
      converted_spec_obj.ToJson(tensorstore::IncludeDefaults{false}));
  EXPECT_EQ(expected_converted_spec, converted_spec);
}

IndexTransform<> GetRandomTransform(IndexDomainView<> domain,
                                    absl::BitGen* gen) {
  auto transform = IdentityTransform(domain);
  const auto ApplyExpression = [&](auto e) {
    transform = ChainResult(transform, e).value();
  };
  for (DimensionIndex i = 0; i < domain.rank(); ++i) {
    if (domain[i].empty()) continue;
    switch (absl::Uniform(absl::IntervalClosedClosed, *gen, 0, 1)) {
      case 0: {
        const Index start =
            absl::Uniform(absl::IntervalClosedOpen, *gen,
                          domain[i].inclusive_min(), domain[i].exclusive_max());
        const Index stop = absl::Uniform(absl::IntervalOpenClosed, *gen, start,
                                         domain[i].exclusive_max());
        const Index stride =
            absl::Uniform(absl::IntervalClosedClosed, *gen, Index(1), Index(3));
        ApplyExpression(Dims(i).HalfOpenInterval(start, stop, stride));
        if (absl::Bernoulli(*gen, 0.5)) {
          ApplyExpression(Dims(i).HalfOpenInterval(kImplicit, kImplicit, -1));
        }
        break;
      }
      case 1: {
        std::vector<Index> values(domain[i].size());
        absl::c_iota(values, domain[i].inclusive_min());
        absl::c_shuffle(values, *gen);
        values.resize(absl::Uniform(absl::IntervalClosedClosed, *gen,
                                    std::size_t(1), values.size()));
        ApplyExpression(
            Dims(i).OuterIndexArraySlice(MakeCopy(MakeArrayView((values)))));
        break;
      }
    }
  }
  return transform;
}

void TestTensorStoreDriverBasicFunctionality(
    ::nlohmann::json create_spec, std::vector<std::string> expected_labels,
    OffsetArrayView<const void> initial_value) {
  SCOPED_TRACE(StrCat("create_spec=", create_spec));
  auto context = Context::Default();
  auto store_result =
      tensorstore::Open(context, create_spec, tensorstore::OpenMode::create)
          .result();
  ASSERT_EQ(Status(), GetStatus(store_result));
  auto store = *store_result;
  const IndexDomain<> expected_domain(
      IndexTransformBuilder<>(initial_value.rank(), 0)
          .input_bounds(initial_value.domain())
          .input_labels(expected_labels)
          .Finalize()
          .value());
  ASSERT_EQ(expected_domain, store.domain());
  ASSERT_EQ(initial_value.data_type(), store.data_type());
  {
    auto resolved_store = ResolveBounds(store).result();
    ASSERT_EQ(Status(), GetStatus(resolved_store));
    EXPECT_EQ(expected_domain, resolved_store->domain());
  }
  {
    auto resolved_store = ResolveBounds(store, fix_resizable_bounds).result();
    ASSERT_EQ(Status(), GetStatus(resolved_store));
    EXPECT_EQ(ChainResult(IdentityTransform(expected_domain),
                          tensorstore::AllDims().MarkBoundsExplicit())
                  .value()
                  .domain(),
              resolved_store->domain());
  }

  // Read fill value.
  {
    auto read_result = Read(store).result();
    ASSERT_EQ(Status(), GetStatus(read_result));
    EXPECT_EQ(initial_value, *read_result);
  }

  auto expected_value = MakeCopy(initial_value);

  const std::size_t iterations = 20;
  absl::BitGen gen;

  for (std::size_t i = 0; i < iterations; ++i) {
    auto transform = GetRandomTransform(expected_domain, &gen);
    auto random_array = MakeRandomArray(gen, transform.domain().box(),
                                        initial_value.data_type());
    TENSORSTORE_LOG("i = ", i);
    SCOPED_TRACE(StrCat("i=", i));
    SCOPED_TRACE(StrCat("transform=", transform));
    SCOPED_TRACE(StrCat("original_domain=", initial_value.domain()));
    auto write_future =
        tensorstore::Write(random_array, ChainResult(store, transform))
            .commit_future;
    ASSERT_TRUE(write_future.WaitFor(absl::Seconds(5)));
    ASSERT_EQ(Status(), GetStatus(write_future.result()));
    auto read_part_future = Read(ChainResult(store, transform));
    ASSERT_TRUE(read_part_future.WaitFor(absl::Seconds(5)));
    auto& read_part_result = read_part_future.result();
    ASSERT_EQ(Status(), GetStatus(read_part_result));
    EXPECT_EQ(random_array, *read_part_result);

    ASSERT_EQ(Status(), CopyTransformedArray(
                            random_array,
                            ChainResult(expected_value, transform).value()));

    auto read_full_future = Read(store);
    ASSERT_TRUE(read_full_future.WaitFor(absl::Seconds(5)));
    auto& read_full_result = read_full_future.result();
    ASSERT_EQ(Status(), GetStatus(read_full_result));
    EXPECT_EQ(expected_value, *read_full_result);
  }
}

}  // namespace internal
}  // namespace tensorstore
