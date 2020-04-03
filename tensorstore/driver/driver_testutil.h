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
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/spec_request_options.h"

namespace tensorstore {
namespace internal {

/// Tests that a TensorStore can be successfully created from `full_spec`, that
/// its full and minimal specs are `full_spec` and `minimal_spec`, respectively,
/// and that a TensorStore can then be successfully opened from `minimal_spec`,
/// and that the resultant TensorStore also has full and minimal specs of
/// `full_spec` and `minimal_spec`, respectively.
void TestTensorStoreDriverSpecRoundtrip(
    ::nlohmann::json full_spec, ::nlohmann::json minimal_spec,
    ContextToJsonOptions options = IncludeDefaults{false});

/// Tests that applying `options` to `orig_spec` (via `Spec::Convert`) results
/// in `expected_converted_spec`.
void TestTensorStoreDriverSpecConvert(::nlohmann::json orig_spec,
                                      const SpecRequestOptions& options,
                                      ::nlohmann::json expected_converted_spec);

/// Tests that `create_spec` creates a TensorStore with the specified domain and
/// data type, and that `Read`, `Write`, and `ResolveBounds` functions work.
void TestTensorStoreDriverBasicFunctionality(
    ::nlohmann::json create_spec, std::vector<std::string> expected_labels,
    OffsetArrayView<const void> initial_value);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DRIVER_TESTUTIL_H_
