// Copyright 2023 The TensorStore Authors
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

// Tests of the zarr driver `GetStorageStatistics` support.

#include <gtest/gtest.h>
#include "tensorstore/driver/zarr/storage_statistics_test_util.h"

namespace {

using ::tensorstore::internal_zarr::ZarrLikeStorageStatisticsTest;
using ::tensorstore::internal_zarr::ZarrLikeStorageStatisticsTestParams;

INSTANTIATE_TEST_SUITE_P(
    ZarrStorageStatisticsTest, ZarrLikeStorageStatisticsTest,
    ::testing::Values(
        ZarrLikeStorageStatisticsTestParams{"zarr", '.'},
        ZarrLikeStorageStatisticsTestParams{
            "zarr", '/', {{"metadata", {{"dimension_separator", "/"}}}}}),
    ::testing::PrintToStringParamName());

}  // namespace
