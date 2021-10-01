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

#ifndef TENSORSTORE_SERIALIZATION_TEST_UTIL_H_
#define TENSORSTORE_SERIALIZATION_TEST_UTIL_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/serialization/batch.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace serialization {

template <typename T, typename Serializer = serialization::Serializer<T>>
Result<T> SerializationRoundTrip(const T& value,
                                 const Serializer serializer = {}) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto encoded, EncodeBatch(value, serializer));
  T decoded;
  TENSORSTORE_RETURN_IF_ERROR(DecodeBatch(encoded, decoded, serializer));
  return {std::in_place, std::move(decoded)};
}

template <typename T, typename Serializer = serialization::Serializer<T>>
void TestSerializationRoundTrip(const T& value,
                                const Serializer serializer = {}) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(value, serializer));
  T decoded;
  auto decode_status = DecodeBatch(encoded, decoded, serializer);
  if (!decode_status.ok()) {
    TENSORSTORE_ASSERT_OK(decode_status)
        << "Encoded: " << tensorstore::QuoteString(encoded);
  }
  EXPECT_EQ(decoded, value);
}

template <typename T, typename Serializer = serialization::Serializer<T>>
void TestSerializationRoundTripCorrupt(const T& value,
                                       const Serializer serializer = {}) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(value, serializer));
  ASSERT_GT(encoded.size(), 0);
  encoded.resize(encoded.size() - 1);
  T decoded;
  auto decode_status = DecodeBatch(encoded, decoded, serializer);
  EXPECT_THAT(decode_status,
              tensorstore::MatchesStatus(absl::StatusCode::kDataLoss));
}

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_TEST_UTIL_H_
