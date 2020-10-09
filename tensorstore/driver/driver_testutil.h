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

#ifndef TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
#define TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_

#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/spec_request_options.h"
#include "tensorstore/transaction.h"

namespace tensorstore {
namespace internal {

struct TestTensorStoreDriverSpecRoundtripOptions {
  std::string test_name;
  ::nlohmann::json full_spec;
  ::nlohmann::json create_spec{::nlohmann::json::value_t::discarded};
  ::nlohmann::json minimal_spec;
  ContextToJsonOptions to_json_options = IncludeDefaults{false};
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated, tensorstore::atomic_isolated};
  bool check_not_found_before_create = true;
  bool check_not_found_before_commit = true;
  bool check_transactional_open_before_commit = true;
  bool write_value_to_create = false;
};

/// Tests that a TensorStore can be successfully created from `full_spec`, that
/// its full and minimal specs are `full_spec` and `minimal_spec`, respectively,
/// and that a TensorStore can then be successfully opened from `minimal_spec`,
/// and that the resultant TensorStore also has full and minimal specs of
/// `full_spec` and `minimal_spec`, respectively.
void RegisterTensorStoreDriverSpecRoundtripTest(
    TestTensorStoreDriverSpecRoundtripOptions options);

/// Tests that applying `options` to `orig_spec` (via `Spec::Convert`) results
/// in `expected_converted_spec`.
void TestTensorStoreDriverSpecConvert(::nlohmann::json orig_spec,
                                      const SpecRequestOptions& options,
                                      ::nlohmann::json expected_converted_spec);

struct TensorStoreDriverBasicFunctionalityTestOptions {
  std::string test_name;
  ::nlohmann::json create_spec;
  IndexDomain<> expected_domain;
  SharedOffsetArray<const void> initial_value;
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated};

  /// Optional.  Specifies a function to use to compare arrays (e.g. to perform
  /// approximate comparison to accommodate lossy compression).
  ///
  /// The `compare_arrays` function should call a GTest assertion to record any
  /// mismatch.  If not specified, the default function simply calls:
  /// `EXPECT_EQ(expected, actual);`.
  std::function<void(OffsetArrayView<const void> expected,
                     OffsetArrayView<const void> actual)>
      compare_arrays;

  bool check_not_found_before_commit = true;
};

/// Tests that `create_spec` creates a TensorStore with the specified domain and
/// data type, and that `Read`, `Write`, and `ResolveBounds` functions work.
void RegisterTensorStoreDriverBasicFunctionalityTest(
    TensorStoreDriverBasicFunctionalityTestOptions options);

struct TestTensorStoreDriverResizeOptions {
  std::string test_name;
  /// Specifies the initial bounds to use.
  Box<> initial_bounds;
  /// Returns a TensorStore spec for the specified bounds.
  std::function<::nlohmann::json(BoxView<> bounds)> get_create_spec;
  std::vector<TransactionMode> supported_transaction_modes = {
      tensorstore::isolated};
};

/// Tests metadata-only resize functionality.
void RegisterTensorStoreDriverResizeTest(
    TestTensorStoreDriverResizeOptions options);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
