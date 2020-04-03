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

#include "tensorstore/internal/http/curl_request_builder.h"

#include <gtest/gtest.h>

using tensorstore::internal_http::CurlRequestBuilder;

namespace {

TEST(CurlEscapeString, EscapeStringTest) {
  auto escaped = tensorstore::internal_http::CurlEscapeString(
      "abc!@#$%^&*()_~ \t<>[]{},.123");

  EXPECT_EQ(
      "abc%21%40%23%24%25%5E%26%2A%28%29_~%20%09%3C%3E%5B%5D%7B%7D%2C.123",
      escaped);
}

TEST(CurlRequestBuilderTest, BuildRequest) {
  CurlRequestBuilder request_builder(
      "http://127.0.0.1:0/",
      tensorstore::internal_http::GetDefaultCurlHandleFactory());

  auto request = request_builder.BuildRequest();

  EXPECT_EQ("http://127.0.0.1:0/", request.url());

  // This fails with a "Cannot connect to server error.
  auto response = request.IssueRequest("");
  EXPECT_FALSE(response.ok());
}

}  // namespace
