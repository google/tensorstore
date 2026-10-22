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

#include <stdint.h>

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/declare.h"
#include "absl/flags/flag.h"
#include "absl/flags/reflection.h"
#include "tensorstore/internal/curl/default_factory.h"
#include "tensorstore/internal/env.h"

ABSL_DECLARE_FLAG(std::optional<uint32_t>,
                  tensorstore_curl_connect_timeout_seconds);

namespace {

using ::tensorstore::internal_http::DefaultCurlHandleFactory;

TEST(DefaultCurlHandleFactoryTest, DefaultConfig) {
  auto config = DefaultCurlHandleFactory::DefaultConfig();
  EXPECT_EQ(config.connect_timeout_seconds, 30);
  EXPECT_EQ(config.low_speed_time_seconds, 60);
  EXPECT_EQ(config.low_speed_limit_bytes, 256);
}

TEST(DefaultCurlHandleFactoryTest, FlagOverride) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_tensorstore_curl_connect_timeout_seconds, 15);
  auto config = DefaultCurlHandleFactory::DefaultConfig();
  EXPECT_EQ(config.connect_timeout_seconds, 15);
}

TEST(DefaultCurlHandleFactoryTest, EnvOverride) {
  absl::FlagSaver flag_saver;
  absl::SetFlag(&FLAGS_tensorstore_curl_connect_timeout_seconds, std::nullopt);
  tensorstore::internal::SetEnv("TENSORSTORE_CURL_CONNECT_TIMEOUT_SECONDS",
                                "45");
  auto config = DefaultCurlHandleFactory::DefaultConfig();
  EXPECT_EQ(config.connect_timeout_seconds, 45);
  tensorstore::internal::UnsetEnv("TENSORSTORE_CURL_CONNECT_TIMEOUT_SECONDS");
}

TEST(DefaultCurlHandleFactoryTest, CreateHandle) {
  DefaultCurlHandleFactory factory(DefaultCurlHandleFactory::DefaultConfig());
  auto handle = factory.CreateHandle();
  EXPECT_NE(handle, nullptr);
}

}  // namespace
