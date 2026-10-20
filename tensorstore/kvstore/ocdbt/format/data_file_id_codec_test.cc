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

#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"

#include <stdint.h>

#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/varint/varint_writing.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_ocdbt::BasePath;
using ::tensorstore::internal_ocdbt::DataFileId;
using ::tensorstore::internal_ocdbt::DataFileTable;
using ::tensorstore::internal_ocdbt::DataFileTableBuilder;
using ::tensorstore::internal_ocdbt::FinalizeReader;
using ::tensorstore::internal_ocdbt::FinalizeWriter;
using ::tensorstore::internal_ocdbt::kMaxPathLength;
using ::tensorstore::internal_ocdbt::ReadDataFileTable;
using ::testing::HasSubstr;

Result<absl::Cord> Encode(const DataFileTable& table) {
  DataFileTableBuilder builder;
  for (const auto& file : table.files) {
    builder.Add(file);
  }
  absl::Cord cord;
  {
    riegeli::CordWriter writer{&cord};
    bool success = builder.Finalize(writer);
    TENSORSTORE_RETURN_IF_ERROR(FinalizeWriter(writer, success));
  }
  return cord;
}

Result<DataFileTable> Decode(const absl::Cord& cord,
                             const BasePath& base_path = {}) {
  DataFileTable new_table;
  {
    riegeli::CordReader reader{&cord};
    bool success = ReadDataFileTable(reader, base_path, new_table);
    TENSORSTORE_RETURN_IF_ERROR(FinalizeReader(reader, success));
  }
  return new_table;
}

Result<DataFileTable> RoundTrip(const DataFileTable& table,
                                const BasePath& base_path = {}) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto cord, Encode(table));
  return Decode(cord, base_path);
}

TEST(DataFileBuilderTest, Empty) {
  DataFileTable table;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, Encode(table));
  EXPECT_EQ(1, encoded.size());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_table, Decode(encoded));
  EXPECT_EQ(table.files, new_table.files);
}

TEST(DataFileBuilderTest, Simple) {
  DataFileTable table;
  table.files = {
      {"b", "d"}, {"a", "c"}, {"a", "b"}, {"b", "e"}, {"b", "d"},
  };
  // Currently `DataFileBuilder` sorts the ids.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_table, RoundTrip(table, ""));
  ASSERT_THAT(new_table.files, ::testing::ElementsAreArray({
                                   DataFileId{"a", "b"},
                                   DataFileId{"a", "c"},
                                   DataFileId{"b", "d"},
                                   DataFileId{"b", "e"},
                               }));
}

TEST(DataFileBuilderTest, Prefixes) {
  DataFileTable table;
  table.files = {
      {"", ""},       //
      {"", "a"},      //
      {"", "ab"},     //
      {"", "ac"},     //
      {"a", ""},      //
      {"a", "x"},     //
      {"a", "xy"},    //
      {"a", "xyz"},   //
      {"ab", ""},     //
      {"ab", "xy"},   //
      {"ab", "xyz"},  //
      {"ac", "xy"},   //
  };
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_table, RoundTrip(table, ""));
  ASSERT_THAT(new_table.files, ::testing::ElementsAreArray(table.files));
}

TEST(DataFileBuilderTest, AddBasePath) {
  DataFileTable table;
  table.files = {
      {"b", "d"}, {"a", "c"}, {"a", "b"}, {"b", "e"}, {"b", "d"}, {"", "y"},
  };
  // Currently `DataFileBuilder` sorts the ids.
  BasePath base_path = "x/";
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto new_table, RoundTrip(table, base_path));
  ASSERT_THAT(new_table.files, ::testing::ElementsAreArray({
                                   DataFileId{"x/", "y"},
                                   DataFileId{"x/a", "b"},
                                   DataFileId{"x/a", "c"},
                                   DataFileId{"x/b", "d"},
                                   DataFileId{"x/b", "e"},
                               }));
  // Verify that specified base path is used directly if the encoded base path
  // is the empty string.
  EXPECT_EQ(base_path.data(), new_table.files[0].base_path.data());
  // Verify that base paths are deduplicated.
  EXPECT_EQ(new_table.files[1].base_path.data(),
            new_table.files[2].base_path.data());
  EXPECT_EQ(new_table.files[3].base_path.data(),
            new_table.files[4].base_path.data());
}

TEST(DataFileBuilderTest, Truncated) {
  DataFileTable table;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, Encode(table));
  ASSERT_EQ(1, encoded.size());
  EXPECT_THAT(Decode(encoded.Subcord(0, 0)),
              StatusIs(absl::StatusCode::kDataLoss));
}

TEST(DataFileBuilderTest, BasePathTooLongWithPrefix) {
  DataFileTable table;
  DataFileId long_id{std::string_view(std::string(kMaxPathLength, 'x'))};
  table.files = {long_id};
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, Encode(table));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded, Decode(encoded));
  ASSERT_EQ(table.files, decoded.files);

  EXPECT_THAT(Decode(encoded, "z"),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("path_length")));
}

TEST(DataFileBuilderTest, SuffixLengthTooLong) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  // num_files
  ASSERT_TRUE(riegeli::WriteVarint64(1, writer));
  // path_suffix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(kMaxPathLength + 1, writer));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(Decode(encoded),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("Invalid 16-bit varint value")));
}

TEST(DataFileBuilderTest, BasePathLengthTooLong) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  // num_files
  ASSERT_TRUE(riegeli::WriteVarint64(1, writer));
  // path_suffix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(5, writer));
  // base_path_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(65536, writer));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(Decode(encoded),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("Invalid 16-bit varint value")));
}

TEST(DataFileBuilderTest, PrefixLengthTooLong) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  // num_files
  ASSERT_TRUE(riegeli::WriteVarint64(2, writer));
  // prefix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(kMaxPathLength + 1, writer));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(Decode(encoded),
              StatusIs(absl::StatusCode::kDataLoss,
                       HasSubstr("Invalid 16-bit varint value")));
}

TEST(DataFileBuilderTest, BasePathLongerThanPath) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  // num_files
  ASSERT_TRUE(riegeli::WriteVarint64(1, writer));
  // path_suffix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(5, writer));
  // base_path_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(6, writer));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(Decode(encoded), StatusIs(absl::StatusCode::kDataLoss,
                                        HasSubstr("base_path_length")));
}

TEST(DataFileBuilderTest, PrefixLengthLongerThanPrevBasePath) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  // num_files
  ASSERT_TRUE(riegeli::WriteVarint64(2, writer));

  // path_prefix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(2, writer));

  // path_suffix_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(2, writer));
  // path_suffix_length[1]
  ASSERT_TRUE(riegeli::WriteVarint64(0, writer));

  // base_path_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(0, writer));
  // base_path_length[0]
  ASSERT_TRUE(riegeli::WriteVarint64(1, writer));
  ASSERT_TRUE(writer.Close());
  EXPECT_THAT(Decode(encoded), StatusIs(absl::StatusCode::kDataLoss,
                                        HasSubstr("path_prefix_length")));
}

// Encodes a data file table directly from its column representation,
// bypassing `DataFileTableBuilder` so that encoder invariants (sort order,
// prefix lengths) may be violated.
absl::Cord EncodeRawDataFileTable(std::vector<uint64_t> prefix_lengths,
                                  std::vector<uint64_t> suffix_lengths,
                                  std::vector<uint64_t> base_path_lengths,
                                  std::string_view suffix_data) {
  absl::Cord encoded;
  riegeli::CordWriter writer{&encoded};
  ABSL_CHECK(riegeli::WriteVarint64(suffix_lengths.size(), writer));
  for (uint64_t x : prefix_lengths) {
    ABSL_CHECK(riegeli::WriteVarint64(x, writer));
  }
  for (uint64_t x : suffix_lengths) {
    ABSL_CHECK(riegeli::WriteVarint64(x, writer));
  }
  for (uint64_t x : base_path_lengths) {
    ABSL_CHECK(riegeli::WriteVarint64(x, writer));
  }
  ABSL_CHECK(writer.Write(suffix_data));
  ABSL_CHECK(writer.Close());
  return encoded;
}

MATCHER_P2(FileIdIs, base_path, relative_path, "") {
  return std::string_view(arg.base_path) == base_path &&
         std::string_view(arg.relative_path) == relative_path;
}

// The common prefix covers the entire (shorter) base path of entry 1, but the
// base paths differ.  The base path of entry 1 must not alias the longer base
// path of entry 0.
TEST(DataFileBuilderTest, PrefixEqualToShorterBasePath) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{2},
                                    /*suffix_lengths=*/{5, 1},
                                    /*base_path_lengths=*/{4, 2}, "abcdxy")),
      ::tensorstore::IsOkAndHolds(::testing::Field(
          &DataFileTable::files,
          ::testing::ElementsAre(FileIdIs("abcd", "x"), FileIdIs("ab", "y")))));
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{2},
                                    /*suffix_lengths=*/{5, 1},
                                    /*base_path_lengths=*/{4, 2}, "abcdxy"),
             "t/"),
      ::tensorstore::IsOkAndHolds(
          ::testing::Field(&DataFileTable::files,
                           ::testing::ElementsAre(FileIdIs("t/abcd", "x"),
                                                  FileIdIs("t/ab", "y")))));
}

// Mirror of PrefixEqualToShorterBasePath: entry 0 has the shorter base path
// and entry 1 extends it.
TEST(DataFileBuilderTest, PrefixEqualToPrevShorterBasePath) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{2},
                                    /*suffix_lengths=*/{3, 3},
                                    /*base_path_lengths=*/{2, 4}, "abxcdy")),
      ::tensorstore::IsOkAndHolds(::testing::Field(
          &DataFileTable::files,
          ::testing::ElementsAre(FileIdIs("ab", "x"), FileIdIs("abcd", "y")))));
}

// Positive boundary: prefix_length equals the full previous path length.
TEST(DataFileBuilderTest, PrefixEqualToPrevPathLength) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{4},
                                    /*suffix_lengths=*/{4, 1},
                                    /*base_path_lengths=*/{2, 2}, "abcde")),
      ::tensorstore::IsOkAndHolds(
          ::testing::Field(&DataFileTable::files,
                           ::testing::ElementsAre(FileIdIs("ab", "cd"),
                                                  FileIdIs("ab", "cde")))));
}

// prefix_length[1] exceeds the length of the previous (empty base path)
// relative path.  Previously this read out of bounds of the previous entry's
// relative path buffer.
TEST(DataFileBuilderTest, PrefixLengthLongerThanPrevRelativePath) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{60000},
                                    /*suffix_lengths=*/{8, 0},
                                    /*base_path_lengths=*/{0, 0}, "AAAAAAAA")),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("path_prefix_length")));
}

// Same as above, but with a shared non-empty base path.
TEST(DataFileBuilderTest, PrefixLengthLongerThanPrevPathSharedBasePath) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{1000},
                                    /*suffix_lengths=*/{3, 0},
                                    /*base_path_lengths=*/{2, 2}, "abc")),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("path_prefix_length")));
}

// Same as above, but the previous relative path is empty.
TEST(DataFileBuilderTest, PrefixLengthWithEmptyPrevRelativePath) {
  EXPECT_THAT(
      Decode(EncodeRawDataFileTable(/*prefix_lengths=*/{3},
                                    /*suffix_lengths=*/{2, 2},
                                    /*base_path_lengths=*/{2, 2}, "abcd")),
      StatusIs(absl::StatusCode::kDataLoss, HasSubstr("path_prefix_length")));
}

}  // namespace
