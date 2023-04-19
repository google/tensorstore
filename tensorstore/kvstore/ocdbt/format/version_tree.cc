// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/version_tree.h"

#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference_codec.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree_codec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

CommitTime::CommitTime(absl::Time time)
    : value(absl::ToInt64Milliseconds(time - absl::UnixEpoch())) {}

using absl::Time;

CommitTime::operator Time() const {
  return absl::UnixEpoch() + absl::Milliseconds(value);
}

namespace {
/// Returns the minimum generation number that may be stored in a version tree
/// leaf node ending with the specified generation number.
GenerationNumber GetMinVersionTreeLeafNodeGenerationNumber(
    VersionTreeArityLog2 version_tree_arity_log2,
    GenerationNumber last_generation_number) {
  assert(last_generation_number != 0);
  return last_generation_number -
         (last_generation_number - 1) %
             (GenerationNumber(1) << version_tree_arity_log2);
}
}  // namespace

std::pair<GenerationNumber, GenerationNumber>
GetVersionTreeLeafNodeRangeContainingGeneration(
    VersionTreeArityLog2 version_tree_arity_log2,
    GenerationNumber generation_number) {
  auto min_generation_number = GetMinVersionTreeLeafNodeGenerationNumber(
      version_tree_arity_log2, generation_number);
  auto max_generation_number =
      min_generation_number +
      ((GenerationNumber(1) << version_tree_arity_log2) - 1);
  return {min_generation_number, max_generation_number};
}

[[nodiscard]] bool ReadVersionTreeLeafNode(
    VersionTreeArityLog2 version_tree_arity_log2, riegeli::Reader& reader,
    const DataFileTable& data_file_table,
    VersionTreeNode::LeafNodeEntries& entries) {
  const size_t max_num_entries = static_cast<size_t>(1)
                                 << version_tree_arity_log2;
  if (!VersionTreeLeafNodeEntryArrayCodec<DataFileTable>{
          data_file_table, max_num_entries}(reader, entries)) {
    return false;
  }
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateVersionTreeLeafNodeEntries(version_tree_arity_log2, entries),
      reader.Fail(_), false);
  return true;
}

[[nodiscard]] bool WriteVersionTreeNodeEntries(
    const Config& config, riegeli::Writer& writer,
    const DataFileTableBuilder& data_file_table,
    const VersionTreeNode::LeafNodeEntries& entries) {
  assert(!entries.empty());
  const size_t max_num_entries = size_t(1) << config.version_tree_arity_log2;
  if (!VersionTreeLeafNodeEntryArrayCodec<DataFileTableBuilder>{
          data_file_table, max_num_entries}(writer, entries)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool WriteVersionTreeNodeEntries(
    const Config& config, riegeli::Writer& writer,
    const DataFileTableBuilder& data_file_table,
    const VersionTreeNode::InteriorNodeEntries& entries) {
  const size_t max_num_entries = size_t(1) << config.version_tree_arity_log2;
  if (!VersionTreeInteriorNodeEntryArrayCodec<DataFileTableBuilder>{
          data_file_table, max_num_entries, /*include_entry_height=*/false}(
          writer, entries)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool ReadVersionTreeInteriorNode(
    VersionTreeArityLog2 version_tree_arity_log2, riegeli::Reader& reader,
    const DataFileTable& data_file_table, VersionTreeHeight height,
    VersionTreeNode::InteriorNodeEntries& entries) {
  auto max_height = GetMaxVersionTreeHeight(version_tree_arity_log2);
  if (height > max_height) {
    reader.Fail(absl::DataLossError(absl::StrFormat(
        "height=%d exceeds maximum of %d for version_tree_arity_log2=%d",
        height, max_height, version_tree_arity_log2)));
    return false;
  }
  const size_t max_arity = (size_t(1) << version_tree_arity_log2);
  if (!VersionTreeInteriorNodeEntryArrayCodec<DataFileTable>{
          data_file_table, max_arity, /*include_entry_height=*/false}(
          reader, entries)) {
    return false;
  }

  // Initialize heights, since they are not stored.
  for (auto& entry : entries) {
    entry.height = height - 1;
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateVersionTreeInteriorNodeEntries(
                                  version_tree_arity_log2, height, entries),
                              reader.Fail(_), false);
  return true;
}

Result<VersionTreeNode> DecodeVersionTreeNode(const absl::Cord& encoded,
                                              const BasePath& base_path) {
  VersionTreeNode node;
  auto status = DecodeWithOptionalCompression(
      encoded, kVersionTreeNodeMagic, kVersionTreeNodeFormatVersion,
      [&](riegeli::Reader& reader, uint32_t version) -> bool {
        if (!VersionTreeArityLog2Codec{}(reader,
                                         node.version_tree_arity_log2) ||
            !reader.ReadByte(node.height)) {
          return false;
        }
        DataFileTable data_file_table;
        if (!ReadDataFileTable(reader, base_path, data_file_table)) {
          return false;
        }
        if (node.height == 0) {
          return ReadVersionTreeLeafNode(
              node.version_tree_arity_log2, reader, data_file_table,
              node.entries.emplace<VersionTreeNode::LeafNodeEntries>());
        } else {
          return ReadVersionTreeInteriorNode(
              node.version_tree_arity_log2, reader, data_file_table,
              node.height,
              node.entries.emplace<VersionTreeNode::InteriorNodeEntries>());
        }
      });
  if (!status.ok()) {
    return tensorstore::MaybeAnnotateStatus(status,
                                            "Error decoding version tree node");
  }
#ifndef NDEBUG
  CheckVersionTreeNodeInvariants(node);
#endif
  return node;
}

Result<absl::Cord> EncodeVersionTreeNode(const Config& config,
                                         const VersionTreeNode& node) {
#ifndef NDEBUG
  assert(node.version_tree_arity_log2 == config.version_tree_arity_log2);
  CheckVersionTreeNodeInvariants(node);
#endif
  return EncodeWithOptionalCompression(
      config, kVersionTreeNodeMagic, kVersionTreeNodeFormatVersion,
      [&](riegeli::Writer& writer) -> bool {
        if (!VersionTreeArityLog2Codec{}(writer,
                                         node.version_tree_arity_log2) ||
            !VersionTreeHeightCodec{}(writer, node.height)) {
          return false;
        }
        DataFileTableBuilder data_file_table;
        std::visit(
            [&](auto& entries) {
              internal_ocdbt::AddDataFiles(data_file_table, entries);
            },
            node.entries);
        if (!data_file_table.Finalize(writer)) return false;
        return std::visit(
            [&](auto& entries) {
              return WriteVersionTreeNodeEntries(config, writer,
                                                 data_file_table, entries);
            },
            node.entries);
      });
}

absl::Status ValidateVersionTreeNodeReference(
    const VersionTreeNode& node, const Config& config,
    GenerationNumber last_generation_number, VersionTreeHeight height) {
  if (node.height != height) {
    return absl::DataLossError(absl::StrFormat(
        "Expected height of %d but received: %d", height, node.height));
  }
  if (node.version_tree_arity_log2 != config.version_tree_arity_log2) {
    return absl::DataLossError(absl::StrFormat(
        "Expected version_tree_arity_log2=%d but received: %d",
        config.version_tree_arity_log2, node.version_tree_arity_log2));
  }
  if (auto generation_number = node.generation_number();
      generation_number != last_generation_number) {
    return absl::DataLossError(
        absl::StrFormat("Expected generation number %d but received: %d",
                        last_generation_number, generation_number));
  }
  return absl::OkStatus();
}

bool operator==(const BtreeGenerationReference& a,
                const BtreeGenerationReference& b) {
  return a.root == b.root && a.generation_number == b.generation_number &&
         a.root_height == b.root_height && a.commit_time == b.commit_time;
}

std::ostream& operator<<(std::ostream& os, const BtreeGenerationReference& x) {
  return os << "{root=" << x.root
            << ", generation_number=" << x.generation_number
            << ", root_height=" << static_cast<int>(x.root_height)
            << ", commit_time=" << x.commit_time << "}";
}

bool operator==(const VersionNodeReference& a, const VersionNodeReference& b) {
  return a.location == b.location &&
         a.generation_number == b.generation_number && a.height == b.height &&
         a.num_generations == b.num_generations &&
         a.commit_time == b.commit_time;
}

std::ostream& operator<<(std::ostream& os, const VersionNodeReference& e) {
  return os << "{location=" << e.location
            << ", generation_number=" << e.generation_number
            << ", height=" << static_cast<size_t>(e.height)
            << ", num_generations=" << e.num_generations
            << ", commit_time=" << e.commit_time << "}";
}

std::ostream& operator<<(std::ostream& os, const VersionTreeNode::Entries& e) {
  std::visit([&](const auto& entries) { os << tensorstore::span(entries); }, e);
  return os;
}

GenerationNumber VersionTreeNode::generation_number() const {
  if (auto* e_leaf = std::get_if<VersionTreeNode::LeafNodeEntries>(&entries)) {
    return e_leaf->back().generation_number;
  } else {
    auto& e_interior = std::get<VersionTreeNode::InteriorNodeEntries>(entries);
    return e_interior.back().generation_number;
  }
}

bool operator==(const VersionTreeNode& a, const VersionTreeNode& b) {
  return a.height == b.height && a.entries == b.entries;
}

std::ostream& operator<<(std::ostream& os, const VersionTreeNode& e) {
  return os << "{height=" << e.height << ", entries=" << e.entries << "}";
}

std::ostream& operator<<(std::ostream& os, CommitTime x) {
  return os << absl::UnixEpoch() + absl::Milliseconds(x.value);
}

absl::Status ValidateVersionTreeLeafNodeEntries(
    VersionTreeArityLog2 version_tree_arity_log2,
    const VersionTreeNode::LeafNodeEntries& entries) {
  const size_t max_num_entries = static_cast<size_t>(1)
                                 << version_tree_arity_log2;
  if (entries.empty() || entries.size() > max_num_entries) {
    return absl::DataLossError(
        absl::StrFormat("num_children=%d outside valid range [1, %d]",
                        entries.size(), max_num_entries));
  }
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries[i];
    if (entry.root.location.IsMissing()) {
      if (entry.root_height != 0) {
        return absl::DataLossError(
            absl::StrFormat("non-zero root_height=%d for empty generation %d\n",
                            entry.root_height, entry.generation_number));
      }
      if (entry.root.statistics != BtreeNodeStatistics{}) {
        return absl::DataLossError(tensorstore::StrCat(
            "non-zero statistics ", entry.root.statistics,
            " for empty generation_number[", i, "]=", entry.generation_number));
      }
    }
    if (entry.generation_number == 0) {
      return absl::DataLossError(
          absl::StrFormat("generation_number[%d] must be non-zero", i));
    }
    if (i != 0) {
      if (entry.generation_number <= entries[i - 1].generation_number) {
        return absl::DataLossError(absl::StrFormat(
            "generation_number[%d]=%d <= generation_number[%d]=%d", i,
            entry.generation_number, i - 1, entries[i - 1].generation_number));
      }
    }
  }
  const GenerationNumber last_generation_number =
      entries.back().generation_number;
  const GenerationNumber first_generation_number =
      entries.front().generation_number;
  const GenerationNumber min_generation_number =
      GetMinVersionTreeLeafNodeGenerationNumber(version_tree_arity_log2,
                                                last_generation_number);
  if (first_generation_number < min_generation_number) {
    return absl::DataLossError(
        absl::StrFormat("Generation range [%d, %d] exceeds maximum of [%d, %d]",
                        first_generation_number, last_generation_number,
                        min_generation_number, last_generation_number));
  }
  return absl::OkStatus();
}

absl::Status ValidateVersionTreeInteriorNodeEntries(
    VersionTreeArityLog2 version_tree_arity_log2, VersionTreeHeight height,
    const VersionTreeNode::InteriorNodeEntries& entries) {
  const size_t max_num_entries = static_cast<size_t>(1)
                                 << version_tree_arity_log2;
  if (entries.empty() || entries.size() > max_num_entries) {
    return absl::DataLossError(
        absl::StrFormat("num_children=%d outside valid range [1, %d]",
                        entries.size(), max_num_entries));
  }
  const GenerationNumber child_generation_number_stride =
      static_cast<GenerationNumber>(1) << (version_tree_arity_log2 * height);
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries[i];
    if (entry.generation_number == 0) {
      return absl::DataLossError(
          absl::StrFormat("generation_number[%d] must be non-zero", i));
    }
    if (i > 0) {
      if (entry.generation_number <= entries[i - 1].generation_number) {
        return absl::DataLossError(absl::StrFormat(
            "generation_number[%d]=%d >= generation_number[%d]=%d", i,
            entry.generation_number, i - 1, entries[i - 1].generation_number));
      }
      if ((entry.generation_number - 1) / child_generation_number_stride ==
          (entries[i - 1].generation_number - 1) /
              child_generation_number_stride) {
        return absl::DataLossError(absl::StrFormat(
            "generation_number[%d]=%d should be in the same child node "
            "as generation_number[%d]=%d",
            i, entry.generation_number, i - 1,
            entries[i - 1].generation_number));
      }
    }
    if ((entry.generation_number % child_generation_number_stride) != 0) {
      return absl::DataLossError(absl::StrFormat(
          "generation_number[%d]=%d is not a multiple of %d", i,
          entry.generation_number, child_generation_number_stride));
    }
    GenerationNumber max_num_generations =
        entry.generation_number % child_generation_number_stride;
    if (max_num_generations == 0) {
      max_num_generations = child_generation_number_stride;
    }
    if (entry.num_generations > max_num_generations) {
      return absl::DataLossError(
          absl::StrFormat("num_generations[%d]=%d for generation_number[%d]=%d "
                          "is greater than %d",
                          i, entry.num_generations, i, entry.generation_number,
                          max_num_generations));
    }
  }

  const GenerationNumber max_arity = static_cast<GenerationNumber>(1)
                                     << version_tree_arity_log2;
  if ((entries.back().generation_number - 1) / child_generation_number_stride /
          max_arity !=
      (entries.front().generation_number - 1) / child_generation_number_stride /
          max_arity) {
    return absl::DataLossError(
        absl::StrFormat("generation_number[0]=%d cannot be in the same node as "
                        "generation_number[%d]=%d",
                        entries.front().generation_number, entries.size() - 1,
                        entries.back().generation_number));
  }
  return absl::OkStatus();
}

bool VersionTreeArityLog2Codec::operator()(riegeli::Reader& reader,
                                           VersionTreeArityLog2& value) const {
  if (!reader.ReadByte(value)) return false;
  if (value == 0 || value > kMaxVersionTreeArityLog2) {
    reader.Fail(absl::InvalidArgumentError(absl::StrFormat(
        "Expected version_tree_arity_log2 in range [1, %d] but received: %d",
        kMaxVersionTreeArityLog2, value)));
    return false;
  }
  return true;
}

#ifndef NDEBUG
void CheckVersionTreeNodeInvariants(const VersionTreeNode& node) {
  assert(node.version_tree_arity_log2 > 0);
  assert(node.version_tree_arity_log2 <= kMaxVersionTreeArityLog2);
  assert(node.height <= GetMaxVersionTreeHeight(node.version_tree_arity_log2));
  if (node.height == 0) {
    assert(
        std::holds_alternative<VersionTreeNode::LeafNodeEntries>(node.entries));
    auto& entries = std::get<VersionTreeNode::LeafNodeEntries>(node.entries);
    TENSORSTORE_CHECK_OK(ValidateVersionTreeLeafNodeEntries(
        node.version_tree_arity_log2, entries));
  } else {
    assert(std::holds_alternative<VersionTreeNode::InteriorNodeEntries>(
        node.entries));
    auto& entries =
        std::get<VersionTreeNode::InteriorNodeEntries>(node.entries);
    TENSORSTORE_CHECK_OK(ValidateVersionTreeInteriorNodeEntries(
        node.version_tree_arity_log2, node.height, entries));
  }
}
#endif  // NDEBUG

}  // namespace internal_ocdbt
}  // namespace tensorstore
