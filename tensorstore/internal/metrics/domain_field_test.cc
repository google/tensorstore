// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/metrics/domain_field.h"

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string_view>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/metrics/domain_impl.h"

namespace {

using ::tensorstore::internal_metrics::ConstexprPerfectHashIndexer;
using ::tensorstore::internal_metrics::DomainField;
using ::tensorstore::internal_metrics::HasDuplicates;
using ::tensorstore::internal_metrics::kInvalidMetricIndex;
using ::tensorstore::internal_metrics::MetricIndex;

// --- DomainString Tests ---

struct TestDomain {
  static constexpr std::array<std::string_view, 3> kValues = {
      "Foo",
      "Bar",
      "Baz",
  };
  // FIND_SEED Foo Bar Baz
  static constexpr uint32_t kSeed = 2;
  static constexpr size_t kTableSize = 3;
};

using TestDomainString = DomainField<TestDomain>;

TEST(DomainStringTest, HasDuplicates) {
  static constexpr std::array<std::string_view, 3> kDupKeys = {"A", "a", "C"};
  static constexpr std::array<std::string_view, 3> kUniqueKeys = {"A", "B",
                                                                  "C"};
  static_assert(HasDuplicates<std::string_view, 3, false>(kDupKeys), "");
  static_assert(!HasDuplicates<std::string_view, 3, false>(kUniqueKeys), "");
  static_assert(!HasDuplicates<std::string_view, 3, true>(kDupKeys), "");
}

struct CaseSensitiveDomain {
  static constexpr std::array<std::string_view, 3> kValues = {"A", "a", "C"};
  // FIND_SEED --case_sensitive=true A a C
  static constexpr uint32_t kSeed = 5;
  static constexpr size_t kTableSize = 3;
};

using CaseSensitiveDomainString =
    DomainField<CaseSensitiveDomain, /*CaseSensitive=*/true>;

TEST(DomainStringTest, CaseSensitivity) {
  CaseSensitiveDomainString dsA("A");
  CaseSensitiveDomainString dsa("a");

  EXPECT_TRUE(dsA.valid());
  EXPECT_TRUE(dsa.valid());
  EXPECT_NE(dsA.index(), dsa.index());
  EXPECT_EQ(dsA.value(), "A");
  EXPECT_EQ(dsa.value(), "a");

  CaseSensitiveDomainString dsInvalid("d");
  EXPECT_FALSE(dsInvalid.valid());
}

struct SpecCaseSensitiveDomain {
  static constexpr std::array<std::string_view, 3> kValues = {"A", "a", "C"};
  static constexpr bool kCaseSensitive [[maybe_unused]] = true;
  // FIND_SEED --case_sensitive=true A a C
  static constexpr uint32_t kSeed = 5;
  static constexpr size_t kTableSize = 3;
};
using SpecCaseSensitiveDomainString = DomainField<SpecCaseSensitiveDomain>;

TEST(DomainStringTest, SpecCaseSensitivity) {
  SpecCaseSensitiveDomainString dsA("A");
  SpecCaseSensitiveDomainString dsa("a");

  EXPECT_TRUE(dsA.valid());
  EXPECT_TRUE(dsa.valid());
  EXPECT_NE(dsA.index(), dsa.index());
  EXPECT_EQ(dsA.value(), "A");
  EXPECT_EQ(dsa.value(), "a");
}

struct SpecCaseSensitiveNotPresentDomain {
  static constexpr std::array<std::string_view, 2> kValues = {"A", "B"};
  // FIND_SEED --type=string
  // A B
  static constexpr uint32_t kSeed = 0;
  static constexpr size_t kTableSize = 2;
};
using SpecCaseSensitiveNotPresentField =
    DomainField<SpecCaseSensitiveNotPresentDomain>;

TEST(DomainStringTest, SpecCaseSensitivityNotPresentIsFalse) {
  static_assert(!SpecCaseSensitiveNotPresentField::kCaseSensitive);
}

struct SpecCaseSensitiveIntegralDomain {
  static constexpr std::array<int64_t, 2> kValues = {1, 2};
  static constexpr bool kCaseSensitive [[maybe_unused]] = true;
};
using SpecCaseSensitiveIntegralField =
    DomainField<SpecCaseSensitiveIntegralDomain>;

TEST(DomainIntTest, SpecCaseSensitivityIntegralIsFalse) {
  static_assert(!SpecCaseSensitiveIntegralField::kCaseSensitive);
}

TEST(DomainStringTest, PerfectHash) {
  static constexpr std::array<std::string_view, 3> kKeys = {"A", "B", "C"};
  static constexpr uint32_t kSeed = 8;
  static constexpr ConstexprPerfectHashIndexer<std::string_view, 3, false, 3>
      kIndexer{kKeys, kSeed};

  // Check that all keys map to distinct indices in [0, 2]
  size_t idxA = kIndexer.GetIndex("A");
  size_t idxB = kIndexer.GetIndex("B");
  size_t idxC = kIndexer.GetIndex("C");

  EXPECT_NE(kInvalidMetricIndex, idxA);
  EXPECT_NE(kInvalidMetricIndex, idxB);
  EXPECT_NE(kInvalidMetricIndex, idxC);

  EXPECT_LT(idxA, 3);
  EXPECT_LT(idxB, 3);
  EXPECT_LT(idxC, 3);

  EXPECT_NE(idxA, idxB);
  EXPECT_NE(idxA, idxC);
  EXPECT_NE(idxB, idxC);

  // Check case-insensitive lookup
  EXPECT_EQ(idxA, kIndexer.GetIndex("a"));
  EXPECT_EQ(idxB, kIndexer.GetIndex("b"));
  EXPECT_EQ(idxC, kIndexer.GetIndex("c"));

  // Check invalid key
  EXPECT_EQ(kInvalidMetricIndex, kIndexer.GetIndex("D"));

  // Check GetKey
  EXPECT_EQ("A", kIndexer.GetKey(idxA));
  EXPECT_EQ("B", kIndexer.GetKey(idxB));
  EXPECT_EQ("C", kIndexer.GetKey(idxC));
}

TEST(DomainStringTest, ConstructionAndValidity) {
  // Valid construction
  TestDomainString ds1("Foo");
  EXPECT_TRUE(ds1.valid());
  EXPECT_EQ(ds1.value(), "Foo");
  EXPECT_EQ(ds1.index(), TestDomainString::kIndexer.GetIndex("Foo"));

  // Case-insensitive valid construction
  TestDomainString ds2("fOo");
  EXPECT_TRUE(ds2.valid());
  EXPECT_EQ(ds2.value(), "Foo");

  // Invalid construction
  TestDomainString ds3("InvalidValue");
  EXPECT_FALSE(ds3.valid());
  EXPECT_EQ(ds3.value(), "");
  EXPECT_EQ(ds3.index(), kInvalidMetricIndex);

  // Construct from index
  size_t idx = TestDomainString::kIndexer.GetIndex("Bar");
  TestDomainString ds4(MetricIndex{idx});
  EXPECT_TRUE(ds4.valid());
  EXPECT_EQ(ds4.value(), "Bar");
  EXPECT_EQ(ds4.index(), idx);
}

TEST(DomainStringTest, StringViewConversion) {
  TestDomainString ds("Bar");
  std::string_view sv = ds;
  EXPECT_EQ(sv, "Bar");

  TestDomainString invalid_ds("Invalid");
  std::string_view invalid_sv = invalid_ds;
  EXPECT_EQ(invalid_sv, "");
}

TEST(DomainStringTest, AbslStringify) {
  TestDomainString ds("Baz");
  EXPECT_EQ(absl::StrCat("Value: ", ds), "Value: Baz");

  TestDomainString invalid_ds("invalid");
  EXPECT_EQ(absl::StrCat("Value: ", invalid_ds), "Value: ");
}

// --- DomainInt Tests ---

struct TestDomainIntSpec {
  static constexpr std::array<int64_t, 3> kValues = {10, 20, 30};
  // FIND_SEED --type=int64_t
  // 10 20 30
  static constexpr uint32_t kSeed = 0;
  static constexpr size_t kTableSize = 3;
};
using TestDomainInt = DomainField<TestDomainIntSpec>;

TEST(DomainIntTest, HasDuplicates) {
  static constexpr std::array<int64_t, 3> kDupKeys = {1, 2, 1};
  static constexpr std::array<int64_t, 3> kUniqueKeys = {1, 2, 3};
  static_assert(HasDuplicates<int64_t, 3>(kDupKeys), "");
  static_assert(!HasDuplicates<int64_t, 3>(kUniqueKeys), "");
}

TEST(DomainIntTest, PerfectHash) {
  static constexpr std::array<int64_t, 3> kKeys = {10, 20, 30};
  static constexpr uint32_t kSeed = 0;
  static constexpr ConstexprPerfectHashIndexer<int64_t, 3, false, 3> kIndexer{
      kKeys, kSeed};

  size_t idx10 = kIndexer.GetIndex(10);
  size_t idx20 = kIndexer.GetIndex(20);
  size_t idx30 = kIndexer.GetIndex(30);

  EXPECT_NE(kInvalidMetricIndex, idx10);
  EXPECT_NE(kInvalidMetricIndex, idx20);
  EXPECT_NE(kInvalidMetricIndex, idx30);

  EXPECT_LT(idx10, 3);
  EXPECT_LT(idx20, 3);
  EXPECT_LT(idx30, 3);

  EXPECT_NE(idx10, idx20);
  EXPECT_NE(idx10, idx30);
  EXPECT_NE(idx20, idx30);

  // Check invalid key
  EXPECT_EQ(kInvalidMetricIndex, kIndexer.GetIndex(40));

  // Check GetKey
  EXPECT_EQ(10, kIndexer.GetKey(idx10));
  EXPECT_EQ(20, kIndexer.GetKey(idx20));
  EXPECT_EQ(30, kIndexer.GetKey(idx30));
}

TEST(DomainIntTest, ConstructionAndValidity) {
  // Valid construction
  TestDomainInt di1(10);
  EXPECT_TRUE(di1.valid());
  EXPECT_EQ(di1.value(), 10);
  EXPECT_EQ(di1.index(), TestDomainInt::kIndexer.GetIndex(10));

  // Invalid construction
  TestDomainInt di2(40);
  EXPECT_FALSE(di2.valid());
  EXPECT_EQ(di2.value(), 0);
  EXPECT_EQ(di2.index(), kInvalidMetricIndex);

  // Construct from index
  size_t idx = TestDomainInt::kIndexer.GetIndex(20);
  TestDomainInt di3(MetricIndex{idx});
  EXPECT_TRUE(di3.valid());
  EXPECT_EQ(di3.value(), 20);
  EXPECT_EQ(di3.index(), idx);
}

TEST(DomainIntTest, AbslStringify) {
  TestDomainInt di(30);
  EXPECT_EQ(absl::StrCat("Value: ", di), "Value: 30");

  TestDomainInt invalid_di(40);
  EXPECT_EQ(absl::StrCat("Value: ", invalid_di), "Value: ");
}

// --- DomainEnum Tests ---

enum class TestEnum { kA = 5, kB = 10, kC = 15 };

struct TestDomainEnumSpec {
  static constexpr std::array<TestEnum, 3> kValues = {
      TestEnum::kA, TestEnum::kB, TestEnum::kC};
  // FIND_SEED --type=int32_t
  // 5 10 15
  static constexpr uint32_t kSeed = 16;
  static constexpr size_t kTableSize = 3;
};
using TestDomainEnum = DomainField<TestDomainEnumSpec>;

TEST(DomainEnumTest, PerfectHash) {
  static constexpr std::array<TestEnum, 3> kKeys = {TestEnum::kA, TestEnum::kB,
                                                    TestEnum::kC};
  static constexpr uint32_t kSeed = 16;
  static constexpr ConstexprPerfectHashIndexer<TestEnum, 3, false, 3> kIndexer{
      kKeys, kSeed};

  size_t idxA = kIndexer.GetIndex(TestEnum::kA);
  size_t idxB = kIndexer.GetIndex(TestEnum::kB);
  size_t idxC = kIndexer.GetIndex(TestEnum::kC);

  EXPECT_NE(kInvalidMetricIndex, idxA);
  EXPECT_NE(kInvalidMetricIndex, idxB);
  EXPECT_NE(kInvalidMetricIndex, idxC);

  EXPECT_LT(idxA, 3);
  EXPECT_LT(idxB, 3);
  EXPECT_LT(idxC, 3);

  EXPECT_NE(idxA, idxB);
  EXPECT_NE(idxA, idxC);
  EXPECT_NE(idxB, idxC);

  // Check invalid key
  EXPECT_EQ(kInvalidMetricIndex, kIndexer.GetIndex(static_cast<TestEnum>(0)));

  // Check GetKey
  EXPECT_EQ(TestEnum::kA, kIndexer.GetKey(idxA));
  EXPECT_EQ(TestEnum::kB, kIndexer.GetKey(idxB));
  EXPECT_EQ(TestEnum::kC, kIndexer.GetKey(idxC));
}

TEST(DomainEnumTest, ConstructionAndValidity) {
  // Valid construction
  TestDomainEnum di1(TestEnum::kA);
  EXPECT_TRUE(di1.valid());
  EXPECT_EQ(di1.value(), TestEnum::kA);
  EXPECT_EQ(di1.index(), TestDomainEnum::kIndexer.GetIndex(TestEnum::kA));

  // Invalid construction
  TestDomainEnum di2(static_cast<TestEnum>(0));
  EXPECT_FALSE(di2.valid());
  EXPECT_EQ(di2.value(), static_cast<TestEnum>(0));
  EXPECT_EQ(di2.index(), kInvalidMetricIndex);

  // Construct from index
  size_t idx = TestDomainEnum::kIndexer.GetIndex(TestEnum::kB);
  TestDomainEnum di3(MetricIndex{idx});
  EXPECT_TRUE(di3.valid());
  EXPECT_EQ(di3.value(), TestEnum::kB);
  EXPECT_EQ(di3.index(), idx);
}

TEST(DomainEnumTest, AbslStringify) {
  TestDomainEnum di(TestEnum::kC);
  EXPECT_EQ(absl::StrCat("Value: ", di), "Value: 15");

  TestDomainEnum invalid_di(static_cast<TestEnum>(0));
  EXPECT_EQ(absl::StrCat("Value: ", invalid_di), "Value: ");
}

enum class HttpStatus {
  // Informational 1xx
  kContinue = 100,            // RFC 9110, 15.2.1
  kSwitchingProtocols = 101,  // RFC 9110, 15.2.2
  kEarlyHints = 103,          // RFC 8297

  // Successful 2xx
  kOk = 200,              // RFC 9110, 15.3.1
  kCreated = 201,         // RFC 9110, 15.3.2
  kAccepted = 202,        // RFC 9110, 15.3.3
  kNoContent = 204,       // RFC 9110, 15.3.5
  kPartialContent = 206,  // RFC 9110, 15.3.7
  kMultiStatus = 207,     // RFC 4918, 11.1

  // Redirection 3xx
  kMovedPermanently = 301,   // RFC 9110, 15.4.2
  kFound = 302,              // RFC 9110, 15.4.3
  kNotModified = 304,        // RFC 9110, 15.4.5
  kTemporaryRedirect = 307,  // RFC 9110, 15.4.8
  kPermanentRedirect = 308,  // RFC 9110, 15.4.9

  // Client Error 4xx
  kBadRequest = 400,                  // RFC 9110, 15.5.1
  kUnauthorized = 401,                // RFC 9110, 15.5.2
  kForbidden = 403,                   // RFC 9110, 15.5.4
  kNotFound = 404,                    // RFC 9110, 15.5.5
  kMethodNotAllowed = 405,            // RFC 9110, 15.5.6
  kNotAcceptable = 406,               // RFC 9110, 15.5.7
  kRequestTimeout = 408,              // RFC 9110, 15.5.9
  kConflict = 409,                    // RFC 9110, 15.5.10
  kGone = 410,                        // RFC 9110, 15.5.11
  kPreconditionFailed = 412,          // RFC 9110, 15.5.13
  kUnsupportedMediaType = 415,        // RFC 9110, 15.5.16
  kRangeNotSatisfiable = 416,         // RFC 9110, 15.5.17
  kTeapot = 418,                      // RFC 2324, 2.3.2 (I'm a teapot)
  kTooManyRequests = 429,             // RFC 6585, 4
  kUnavailableForLegalReasons = 451,  // RFC 7725, 3

  // Server Error 5xx
  kInternalServerError = 500,  // RFC 9110, 15.6.1
  kNotImplemented = 501,       // RFC 9110, 15.6.2
  kBadGateway = 502,           // RFC 9110, 15.6.3
  kServiceUnavailable = 503,   // RFC 9110, 15.6.4
  kGatewayTimeout = 504,       // RFC 9110, 15.6.5
};

struct HttpStatusDomain {
  static constexpr std::array<HttpStatus, 34> kValues = {
      HttpStatus::kContinue,
      HttpStatus::kSwitchingProtocols,
      HttpStatus::kEarlyHints,
      HttpStatus::kOk,
      HttpStatus::kCreated,
      HttpStatus::kAccepted,
      HttpStatus::kNoContent,
      HttpStatus::kPartialContent,
      HttpStatus::kMultiStatus,
      HttpStatus::kMovedPermanently,
      HttpStatus::kFound,
      HttpStatus::kNotModified,
      HttpStatus::kTemporaryRedirect,
      HttpStatus::kPermanentRedirect,
      HttpStatus::kBadRequest,
      HttpStatus::kUnauthorized,
      HttpStatus::kForbidden,
      HttpStatus::kNotFound,
      HttpStatus::kMethodNotAllowed,
      HttpStatus::kNotAcceptable,
      HttpStatus::kRequestTimeout,
      HttpStatus::kConflict,
      HttpStatus::kGone,
      HttpStatus::kPreconditionFailed,
      HttpStatus::kUnsupportedMediaType,
      HttpStatus::kRangeNotSatisfiable,
      HttpStatus::kTeapot,
      HttpStatus::kTooManyRequests,
      HttpStatus::kUnavailableForLegalReasons,
      HttpStatus::kInternalServerError,
      HttpStatus::kNotImplemented,
      HttpStatus::kBadGateway,
      HttpStatus::kServiceUnavailable,
      HttpStatus::kGatewayTimeout,
  };

  // FIND_SEED --type=int32_t --table_size=42
  // 100 101 103 200 201 202 204 206 207 301 302 304 307 308
  // 400 401 403 404 405 406 408 409 410 412 415 416 418 429
  // 451 500 501 502 503 504
  static constexpr uint32_t kSeed = 5799507;
  static constexpr size_t kTableSize = 42;
};

using HttpStatusField = DomainField<HttpStatusDomain, false>;

TEST(HttpStatusDomainTest, PerfectHash) {
  HttpStatusField fieldOk(HttpStatus::kOk);
  EXPECT_TRUE(fieldOk.valid());
  EXPECT_EQ(fieldOk.value(), HttpStatus::kOk);

  HttpStatusField fieldInternal(HttpStatus::kInternalServerError);
  EXPECT_TRUE(fieldInternal.valid());
  EXPECT_EQ(fieldInternal.value(), HttpStatus::kInternalServerError);

  HttpStatusField fieldInvalid(static_cast<HttpStatus>(-1));
  EXPECT_FALSE(fieldInvalid.valid());
}

TEST(HttpStatusDomainTest, Interfaces) {
  HttpStatusField fieldOk(HttpStatus::kOk);
  HttpStatusField fieldInternal(HttpStatus::kInternalServerError);
  HttpStatusField fieldInvalid(static_cast<HttpStatus>(-1));

  // 1. index()
  EXPECT_LT(fieldOk.index(), 42);
  EXPECT_LT(fieldInternal.index(), 42);
  EXPECT_EQ(fieldInvalid.index(), kInvalidMetricIndex);

  // 2. Constructor from MetricIndex
  HttpStatusField fieldFromIndex(MetricIndex(fieldOk.index()));
  EXPECT_TRUE(fieldFromIndex.valid());
  EXPECT_EQ(fieldFromIndex.value(), HttpStatus::kOk);
  EXPECT_EQ(fieldFromIndex.index(), fieldOk.index());

  HttpStatusField fieldFromInvalidIndex{MetricIndex{kInvalidMetricIndex}};
  EXPECT_FALSE(fieldFromInvalidIndex.valid());
  EXPECT_EQ(fieldFromInvalidIndex.index(), kInvalidMetricIndex);

  // Test unoccupied and out of table-size bounds indices in DomainField
  // Find unoccupied index in the perfect hash table
  size_t unoccupied_idx = kInvalidMetricIndex;
  for (size_t i = 0; i < HttpStatusField::kTableSize; ++i) {
    if (!HttpStatusField::kIndexer.occupied(i)) {
      unoccupied_idx = i;
      break;
    }
  }
  ASSERT_NE(unoccupied_idx, kInvalidMetricIndex);
  HttpStatusField fieldFromUnoccupiedIndex{MetricIndex{unoccupied_idx}};
  EXPECT_FALSE(fieldFromUnoccupiedIndex.valid());
  EXPECT_EQ(fieldFromUnoccupiedIndex.index(), kInvalidMetricIndex);

  // index out of table-size bounds (e.g. 100)
  HttpStatusField fieldFromOutOfBoundsIndex{MetricIndex{100}};
  EXPECT_FALSE(fieldFromOutOfBoundsIndex.valid());
  EXPECT_EQ(fieldFromOutOfBoundsIndex.index(), kInvalidMetricIndex);

  // 3. Conversion operators
  HttpStatus val = fieldOk;
  EXPECT_EQ(val, HttpStatus::kOk);

  int u_val = static_cast<int>(fieldInternal);
  EXPECT_EQ(u_val, static_cast<int>(HttpStatus::kInternalServerError));

  // 4. Comparison operators
  EXPECT_TRUE(fieldOk == fieldOk);
  EXPECT_FALSE(fieldOk == fieldInternal);
  EXPECT_TRUE(fieldOk != fieldInternal);
  EXPECT_FALSE(fieldOk != fieldOk);

  // 5. AbslStringify
  EXPECT_EQ(absl::StrCat(fieldOk), "200");
  EXPECT_EQ(absl::StrCat(fieldInternal), "500");
  EXPECT_EQ(absl::StrCat(fieldInvalid), "");
}

TEST(HttpStatusDomainTest, PerfectHashIndexer) {
  static constexpr std::array<HttpStatus, 34> kKeys = HttpStatusDomain::kValues;
  static constexpr uint32_t kSeed = HttpStatusDomain::kSeed;
  static constexpr ConstexprPerfectHashIndexer<HttpStatus, /*N=*/34, false,
                                               HttpStatusDomain::kTableSize>
      kIndexer{kKeys, kSeed};

  // Check that all keys map to distinct indices in [0, 41]
  std::array<bool, 42> used = {};
  for (auto key : kKeys) {
    size_t idx = kIndexer.GetIndex(key);
    EXPECT_NE(kInvalidMetricIndex, idx);
    EXPECT_LT(idx, 42);
    EXPECT_FALSE(used[idx])
        << "Duplicate index for key " << static_cast<int>(key);
    used[idx] = true;
    EXPECT_EQ(kIndexer.GetKey(idx), key);
  }

  // Test invalid key
  EXPECT_EQ(kInvalidMetricIndex,
            kIndexer.GetIndex(static_cast<HttpStatus>(-1)));
}

struct SpecExtractedSizeDomain {
  static constexpr std::array<int64_t, 3> kValues = {1, 2, 4};
  // FIND_SEED --type=int64_t --table_size=6 1 2 4
  static constexpr uint32_t kSeed = 1;
  static constexpr size_t kTableSize = 6;
};
using SpecExtractedField = DomainField<SpecExtractedSizeDomain, false>;

TEST(DomainFieldTest, SpecExtractedTableSize) {
  EXPECT_EQ(6, SpecExtractedField::kTableSize);
  SpecExtractedField field1(1);
  EXPECT_TRUE(field1.valid());
  EXPECT_EQ(field1.value(), 1);
  EXPECT_LT(field1.index(), 6);
}

// Constexpr compile-time checks
static constexpr TestDomainString kConstexprString("Foo");
static_assert(kConstexprString.valid());
static_assert(kConstexprString.value() == "Foo");

static constexpr TestDomainInt kConstexprInt(10);
static_assert(kConstexprInt.valid());
static_assert(kConstexprInt.value() == 10);

// Direct pointer API & negative lookup collision checks
TEST(DomainFieldTest, PointerIsPerfect) {
  static constexpr std::array<int64_t, 3> kKeys = {10, 20, 30};
  bool used[4] = {};

  // FIND_SEED --type=int64_t --table_size=4 10 20 30
  static constexpr uint32_t kSeed = 5;
  static constexpr size_t kTableSize = 4;

  // Test that IsPerfect returns true for a known perfect seed.
  for (size_t i = 0; i < 4; ++i) used[i] = false;
  EXPECT_TRUE((::tensorstore::internal_metrics::IsPerfect<int64_t, false>(
      kKeys.data(), kKeys.size(), kSeed, used, kTableSize)));

  // Test that IsPerfect returns false for a known bad seed.
  for (size_t i = 0; i < 4; ++i) used[i] = false;
  EXPECT_FALSE((::tensorstore::internal_metrics::IsPerfect<int64_t, false>(
      kKeys.data(), kKeys.size(), kSeed - 1, used, kTableSize)));
}

// --- Dense Domain Fields tests ---

// The absl::StatusCode enum has a dense domain starting at 0 (ok).
struct StatusCodeDomain {
  static constexpr std::array<absl::StatusCode, 17> kValues = {
      absl::StatusCode::kOk,
      absl::StatusCode::kCancelled,
      absl::StatusCode::kUnknown,
      absl::StatusCode::kInvalidArgument,
      absl::StatusCode::kDeadlineExceeded,
      absl::StatusCode::kNotFound,
      absl::StatusCode::kAlreadyExists,
      absl::StatusCode::kPermissionDenied,
      absl::StatusCode::kResourceExhausted,
      absl::StatusCode::kFailedPrecondition,
      absl::StatusCode::kAborted,
      absl::StatusCode::kOutOfRange,
      absl::StatusCode::kUnimplemented,
      absl::StatusCode::kInternal,
      absl::StatusCode::kUnavailable,
      absl::StatusCode::kDataLoss,
      absl::StatusCode::kUnauthenticated,
  };
};

using DenseStatusCodeField = DomainField<StatusCodeDomain>;
static_assert(DenseStatusCodeField::kIsDense);

TEST(DomainFieldTest, DenseDirectMapping) {
  DenseStatusCodeField fieldOk(absl::StatusCode::kOk);
  EXPECT_TRUE(fieldOk.valid());
  EXPECT_EQ(fieldOk.value(), absl::StatusCode::kOk);
  EXPECT_EQ(fieldOk.index(), static_cast<size_t>(absl::StatusCode::kOk));

  DenseStatusCodeField fieldInternal(absl::StatusCode::kInternal);
  EXPECT_TRUE(fieldInternal.valid());
  EXPECT_EQ(fieldInternal.value(), absl::StatusCode::kInternal);
  EXPECT_EQ(fieldInternal.index(),
            static_cast<size_t>(absl::StatusCode::kInternal));

  // Test construction from MetricIndex
  DenseStatusCodeField fieldFromIndex(
      MetricIndex(static_cast<size_t>(absl::StatusCode::kInternal)));
  EXPECT_TRUE(fieldFromIndex.valid());
  EXPECT_EQ(fieldFromIndex.value(), absl::StatusCode::kInternal);
  EXPECT_EQ(fieldFromIndex.index(),
            static_cast<size_t>(absl::StatusCode::kInternal));

  DenseStatusCodeField fieldFromInvalidIndex{MetricIndex{kInvalidMetricIndex}};
  EXPECT_FALSE(fieldFromInvalidIndex.valid());

  // Test unoccupied / out of table-size bounds in DenseDomainField
  DenseStatusCodeField fieldFromOutOfBoundsIndex{MetricIndex{100}};
  EXPECT_FALSE(fieldFromOutOfBoundsIndex.valid());
  EXPECT_EQ(fieldFromOutOfBoundsIndex.index(), kInvalidMetricIndex);

  // Test invalid key
  DenseStatusCodeField fieldInvalid(static_cast<absl::StatusCode>(-1));
  EXPECT_FALSE(fieldInvalid.valid());
  EXPECT_EQ(fieldInvalid.index(), kInvalidMetricIndex);

  // Test conversion operators
  absl::StatusCode val = fieldOk;
  EXPECT_EQ(val, absl::StatusCode::kOk);

  int u_val = static_cast<int>(fieldInternal);
  EXPECT_EQ(u_val, static_cast<int>(absl::StatusCode::kInternal));

  // Test comparison operators
  EXPECT_TRUE(fieldOk == fieldOk);
  EXPECT_FALSE(fieldOk == fieldInternal);
  EXPECT_TRUE(fieldOk != fieldInternal);
  EXPECT_FALSE(fieldOk != fieldOk);

  // Test AbslStringify
  EXPECT_EQ(absl::StrCat(fieldOk), "0");
  EXPECT_EQ(absl::StrCat(fieldInternal), "13");
  EXPECT_EQ(absl::StrCat(fieldInvalid), "");

  // Test out of bounds (but positive)
  DenseStatusCodeField fieldOutOfBounds(static_cast<absl::StatusCode>(100));
  EXPECT_FALSE(fieldOutOfBounds.valid());
  EXPECT_EQ(fieldOutOfBounds.index(), kInvalidMetricIndex);
}

enum class OffsetEnum {
  kValue10 = 10,
  kValue11 = 11,
  kValue12 = 12,
};

struct OffsetDomain {
  static constexpr std::array<OffsetEnum, 3> kValues = {
      OffsetEnum::kValue10,
      OffsetEnum::kValue11,
      OffsetEnum::kValue12,
  };
};

using OffsetField = DomainField<OffsetDomain>;
static_assert(OffsetField::kIsDense);

TEST(DomainFieldTest, DenseOffsetMapping) {
  OffsetField field10(OffsetEnum::kValue10);
  EXPECT_TRUE(field10.valid());
  EXPECT_EQ(field10.value(), OffsetEnum::kValue10);
  EXPECT_EQ(field10.index(), 0);

  OffsetField field12(OffsetEnum::kValue12);
  EXPECT_TRUE(field12.valid());
  EXPECT_EQ(field12.value(), OffsetEnum::kValue12);
  EXPECT_EQ(field12.index(), 2);

  // Test invalid
  OffsetField fieldInvalid(static_cast<OffsetEnum>(9));
  EXPECT_FALSE(fieldInvalid.valid());
  EXPECT_EQ(fieldInvalid.index(), kInvalidMetricIndex);

  OffsetField fieldOutOfBounds(static_cast<OffsetEnum>(13));
  EXPECT_FALSE(fieldOutOfBounds.valid());
  EXPECT_EQ(fieldOutOfBounds.index(), kInvalidMetricIndex);

  // Test conversion operators
  OffsetEnum val = field10;
  EXPECT_EQ(val, OffsetEnum::kValue10);

  int u_val = static_cast<int>(field12);
  EXPECT_EQ(u_val, 12);

  // Test comparison operators
  EXPECT_TRUE(field10 == field10);
  EXPECT_FALSE(field10 == field12);
  EXPECT_TRUE(field10 != field12);
  EXPECT_FALSE(field10 != field10);

  // Test AbslStringify
  EXPECT_EQ(absl::StrCat(field10), "10");
  EXPECT_EQ(absl::StrCat(field12), "12");
  EXPECT_EQ(absl::StrCat(fieldInvalid), "");
}

enum class SignedEnum : int { kNeg1 = -1, kZero = 0, kOne = 1 };

struct SignedDomain {
  static constexpr std::array<SignedEnum, 3> kValues = {
      SignedEnum::kNeg1,
      SignedEnum::kZero,
      SignedEnum::kOne,
  };
};

using SignedField = DomainField<SignedDomain>;
static_assert(SignedField::kIsDense);

TEST(DomainFieldTest, DenseSignedEnumMapping) {
  SignedField fieldNeg1(SignedEnum::kNeg1);
  EXPECT_TRUE(fieldNeg1.valid());
  EXPECT_EQ(fieldNeg1.value(), SignedEnum::kNeg1);
  EXPECT_EQ(fieldNeg1.index(), 0);

  SignedField fieldZero(SignedEnum::kZero);
  EXPECT_TRUE(fieldZero.valid());
  EXPECT_EQ(fieldZero.value(), SignedEnum::kZero);
  EXPECT_EQ(fieldZero.index(), 1);

  SignedField fieldOne(SignedEnum::kOne);
  EXPECT_TRUE(fieldOne.valid());
  EXPECT_EQ(fieldOne.value(), SignedEnum::kOne);
  EXPECT_EQ(fieldOne.index(), 2);

  // Test invalid entries
  SignedField fieldInvalid(static_cast<SignedEnum>(-2));
  EXPECT_FALSE(fieldInvalid.valid());
  EXPECT_EQ(fieldInvalid.index(), kInvalidMetricIndex);

  SignedField fieldOutOfBounds(static_cast<SignedEnum>(2));
  EXPECT_FALSE(fieldOutOfBounds.valid());
  EXPECT_EQ(fieldOutOfBounds.index(), kInvalidMetricIndex);

  // Test converting back to enum and underlying int
  SignedEnum val = fieldNeg1;
  EXPECT_EQ(val, SignedEnum::kNeg1);

  int u_val = static_cast<int>(fieldNeg1);
  EXPECT_EQ(u_val, -1);
}

}  // namespace
