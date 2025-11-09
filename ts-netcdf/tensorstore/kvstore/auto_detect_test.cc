// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/auto_detect.h"

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/clock.h"
#include <nlohmann/json_fwd.hpp>
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/mock_kvstore.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::InlineExecutor;
using ::tensorstore::JsonSubValuesMatch;
using ::tensorstore::KvStore;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::StorageGeneration;
using ::tensorstore::TimestampedStorageGeneration;
using ::tensorstore::internal::MockKeyValueStore;
using ::tensorstore::internal_kvstore::AutoDetectDirectorySpec;
using ::tensorstore::internal_kvstore::AutoDetectFileSpec;
using ::tensorstore::internal_kvstore::AutoDetectFormat;
using ::tensorstore::internal_kvstore::AutoDetectMatch;
using ::tensorstore::internal_kvstore::AutoDetectRegistration;
using ::tensorstore::kvstore::ReadResult;

class AutoDetectTest : public ::testing::Test {
 public:
  AutoDetectTest() { AutoDetectRegistration::ClearRegistrations(); }
  ~AutoDetectTest() { AutoDetectRegistration::ClearRegistrations(); }
};

// Tests auto detection with the specified mock kvstore read handler.
//
// Returns the auto-detect response and the mock kvstore request log.
std::pair<Result<std::vector<AutoDetectMatch>>, std::deque<::nlohmann::json>>
TestMatch(std::string path, MockKeyValueStore::ReadHandler read_handler) {
  auto mock_kvstore = MockKeyValueStore::Make();
  mock_kvstore->log_requests = true;
  mock_kvstore->read_handler = read_handler;
  auto future = AutoDetectFormat(InlineExecutor{}, KvStore(mock_kvstore, path));
  ABSL_CHECK(future.ready());
  return {future.result(), mock_kvstore->request_log.pop_all()};
}

// Tests that no requests are made and an empty match list is returned for a
// file if nothing is registered.
TEST_F(AutoDetectTest, NothingRegisteredFilePath) {
  EXPECT_THAT(TestMatch("test", /*read_handler=*/{}),
              ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                              ::testing::ElementsAre()));
}

// Tests that no requests are made and an empty match list is returned for a
// directory if nothing is registered.
TEST_F(AutoDetectTest, NothingRegisteredDirectoryPath) {
  EXPECT_THAT(TestMatch("test/", /*read_handler=*/{}),
              ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                              ::testing::ElementsAre()));
}

// Tests that no requests are made and an empty match list is returned for a
// directory if only a file matcher is registered.
TEST_F(AutoDetectTest, OnlyFileMatcherRegisteredDirectoryPath) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  EXPECT_THAT(TestMatch("test/", /*read_handler=*/{}),
              ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                              ::testing::ElementsAre()));
}

// Tests that "ReadResult::Missing" for prefix-only matcher leads to no matches.
TEST_F(AutoDetectTest, FilePrefixMatcherRegisteredNotFound) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_exclusive_max", 1}}))));
}

// Tests that mismatched prefix-only matcher results in no matches.
TEST_F(AutoDetectTest, FilePrefixMatcherRegisteredMismatch) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("Y"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_exclusive_max", 1}}))));
}

// Tests that prefix-only matcher matches successfully.
TEST_F(AutoDetectTest, FilePrefixMatcherRegisteredMatch) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("X"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre(
                          AutoDetectMatch{"prefix-scheme"})),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_exclusive_max", 1}}))));
}

// Tests that read error gets propagated if nothing matches.
TEST_F(AutoDetectTest, FilePrefixMatcherRegisteredReadError) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(absl::FailedPreconditionError("test"));
                }),
      ::testing::Pair(
          MatchesStatus(
              absl::StatusCode::kFailedPrecondition,
              "Format auto-detection failed: Error reading \"test\": test"),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}))));
}

// Tests that "ReadResult::Missing" for suffix-only matcher leads to no matches.
TEST_F(AutoDetectTest, FileSuffixMatcherRegisteredNotFound) {
  AutoDetectRegistration(
      AutoDetectFileSpec::SuffixSignature("suffix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_inclusive_min", -1}}))));
}

// Tests that mismatched suffix-only matcher results in no matches.
TEST_F(AutoDetectTest, FileSuffixMatcherRegisteredMismatch) {
  AutoDetectRegistration(
      AutoDetectFileSpec::SuffixSignature("suffix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("Y"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_inclusive_min", -1}}))));
}

// Tests that suffix-only matcher matches successfully.
TEST_F(AutoDetectTest, FileSuffixMatcherRegisteredMatch) {
  AutoDetectRegistration(
      AutoDetectFileSpec::SuffixSignature("suffix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("X"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre(
                          AutoDetectMatch{"suffix-scheme"})),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_inclusive_min", -1}}))));
}

// Tests that read error gets propagated if nothing matches.
TEST_F(AutoDetectTest, FileSuffixMatcherRegisteredReadError) {
  AutoDetectRegistration(
      AutoDetectFileSpec::SuffixSignature("suffix-scheme", "X"));
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(absl::FailedPreconditionError("test"));
                }),
      ::testing::Pair(
          MatchesStatus(
              absl::StatusCode::kFailedPrecondition,
              "Format auto-detection failed: Error reading \"test\": test"),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -1}}))));
}

// Tests that longest prefix is requested when there are multiple prefix
// matchers.
TEST_F(AutoDetectTest, MultiplePrefixFileMatchersMatch) {
  AutoDetectRegistration(AutoDetectFileSpec::PrefixSignature("x", "X"));
  AutoDetectRegistration(AutoDetectFileSpec::PrefixSignature("xy", "XY"));

  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("X"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre(AutoDetectMatch{"x"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 2}}))));

  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("XY"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::UnorderedElementsAre(
                          AutoDetectMatch{"x"}, AutoDetectMatch{"xy"})),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_exclusive_max", 2}}))));

  // Out of range.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.exclusive_max > 1) {
                    req.promise.SetResult(
                        absl::OutOfRangeError("out of range"));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("X"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::UnorderedElementsAre(AutoDetectMatch{"x"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 2}}),
              JsonSubValuesMatch({{"/type", "read"}, {"/key", "test"}}))));
}

TEST_F(AutoDetectTest, PrefixAndSuffixMatch) {
  AutoDetectRegistration(AutoDetectFileSpec::PrefixSignature("prefix", "P"));
  AutoDetectRegistration(AutoDetectFileSpec::SuffixSignature("suffix", "SX"));

  // Mismatch
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("X"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("ZZ"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre()),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Match prefix only
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("P"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("ZZ"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"prefix"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Match suffix only
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("X"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("SX"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"suffix"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Match prefix and suffix
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("P"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("SX"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre(
              AutoDetectMatch{"prefix"}, AutoDetectMatch{"suffix"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Suffix error
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("P"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  } else {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"prefix"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Prefix error
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.options.byte_range.inclusive_min == 0) {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord("SX"),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"suffix"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));

  // Missing
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre()),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -2}}))));
}

TEST_F(AutoDetectTest, DirectoryMatcher) {
  AutoDetectRegistration(
      AutoDetectDirectorySpec::SingleFile("scheme", "filename"));

  for (auto path : {"test/", "test"}) {
    SCOPED_TRACE(tensorstore::StrCat("path=", path));

    // Filename not found
    EXPECT_THAT(
        TestMatch(path,
                  [](MockKeyValueStore::ReadRequest req) {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  }),
        ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                        ::testing::ElementsAre(JsonSubValuesMatch(
                            {{"/type", "read"},
                             {"/key", "test/filename"},
                             {"/byte_range_exclusive_max", 0}}))));

    // Filename found
    EXPECT_THAT(
        TestMatch(path,
                  [](MockKeyValueStore::ReadRequest req) {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }),
        ::testing::Pair(::testing::Optional(
                            ::testing::ElementsAre(AutoDetectMatch{"scheme"})),
                        ::testing::ElementsAre(JsonSubValuesMatch(
                            {{"/type", "read"},
                             {"/key", "test/filename"},
                             {"/byte_range_exclusive_max", 0}}))));

    // Read error
    EXPECT_THAT(
        TestMatch(path,
                  [](MockKeyValueStore::ReadRequest req) {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  }),
        ::testing::Pair(MatchesStatus(absl::StatusCode::kFailedPrecondition,
                                      "Format auto-detection failed: Error "
                                      "reading \"test/filename\": test"),
                        ::testing::ElementsAre(JsonSubValuesMatch(
                            {{"/type", "read"},
                             {"/key", "test/filename"},
                             {"/byte_range_exclusive_max", 0}}))));
  }
}

TEST_F(AutoDetectTest, MultipleDirectoryMatchers) {
  AutoDetectRegistration(AutoDetectDirectorySpec::SingleFile("scheme-a", "a"));
  AutoDetectRegistration(AutoDetectDirectorySpec::SingleFile("scheme-b", "b"));

  // Both not found
  EXPECT_THAT(
      TestMatch("test/",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre()),
          ::testing::UnorderedElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/b"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // Both found
  EXPECT_THAT(
      TestMatch("test/",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord(),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::UnorderedElementsAre(
              AutoDetectMatch{"scheme-a"}, AutoDetectMatch{"scheme-b"})),
          ::testing::UnorderedElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/b"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // One read error, one not found
  EXPECT_THAT(
      TestMatch("test/",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test/a") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  }
                }),
      ::testing::Pair(
          MatchesStatus(
              absl::StatusCode::kFailedPrecondition,
              "Format auto-detection failed: Error reading \"test/a\": test"),
          ::testing::UnorderedElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/b"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // One read error, one found
  EXPECT_THAT(
      TestMatch("test/",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test/a") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"scheme-b"})),
          ::testing::UnorderedElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/b"},
                                  {"/byte_range_exclusive_max", 0}}))));
}

TEST_F(AutoDetectTest, PrefixAndDirectoryMatchers) {
  AutoDetectRegistration(
      AutoDetectFileSpec::PrefixSignature("prefix-scheme", "X"));
  AutoDetectRegistration(AutoDetectDirectorySpec::SingleFile("scheme-a", "a"));

  // File and directory entry not found
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre()),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> found, not matching.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("Z"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_exclusive_max", 1}}))));

  // File -> read error, directory entry -> not found
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  }
                }),
      ::testing::Pair(
          MatchesStatus(
              absl::StatusCode::kFailedPrecondition,
              "Format auto-detection failed: Error reading \"test\": test"),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> read error, directory entry -> found.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"scheme-a"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> not found, directory entry -> found.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"scheme-a"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_exclusive_max", 1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));
}

TEST_F(AutoDetectTest, SuffixAndDirectoryMatchers) {
  AutoDetectRegistration(
      AutoDetectFileSpec::SuffixSignature("suffix-scheme", "X"));
  AutoDetectRegistration(AutoDetectDirectorySpec::SingleFile("scheme-a", "a"));

  // File and directory entry not found
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Missing(absl::Now()));
                }),
      ::testing::Pair(
          ::testing::Optional(::testing::ElementsAre()),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> found, not matching.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  req.promise.SetResult(ReadResult::Value(
                      absl::Cord("Z"),
                      TimestampedStorageGeneration(
                          StorageGeneration::FromString("g1"), absl::Now())));
                }),
      ::testing::Pair(::testing::Optional(::testing::ElementsAre()),
                      ::testing::ElementsAre(JsonSubValuesMatch(
                          {{"/type", "read"},
                           {"/key", "test"},
                           {"/byte_range_inclusive_min", -1}}))));

  // File -> read error, directory entry -> not found
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  }
                }),
      ::testing::Pair(
          MatchesStatus(
              absl::StatusCode::kFailedPrecondition,
              "Format auto-detection failed: Error reading \"test\": test"),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> read error, directory entry -> found.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(
                        absl::FailedPreconditionError("test"));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"scheme-a"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));

  // File -> not found, directory entry -> found.
  EXPECT_THAT(
      TestMatch("test",
                [](MockKeyValueStore::ReadRequest req) {
                  if (req.key == "test") {
                    req.promise.SetResult(ReadResult::Missing(absl::Now()));
                  } else {
                    req.promise.SetResult(ReadResult::Value(
                        absl::Cord(),
                        TimestampedStorageGeneration(
                            StorageGeneration::FromString("g1"), absl::Now())));
                  }
                }),
      ::testing::Pair(
          ::testing::Optional(
              ::testing::ElementsAre(AutoDetectMatch{"scheme-a"})),
          ::testing::ElementsAre(
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test"},
                                  {"/byte_range_inclusive_min", -1}}),
              JsonSubValuesMatch({{"/type", "read"},
                                  {"/key", "test/a"},
                                  {"/byte_range_exclusive_max", 0}}))));
}

}  // namespace
