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

#include "tensorstore/kvstore/ocdbt/format/manifest.h"

#include <cassert>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/ocdbt/format/codec_util.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/config_codec.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id_codec.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree_codec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_ocdbt {

constexpr uint32_t kManifestMagic = 0x0cdb3a2a;
constexpr uint8_t kManifestFormatVersion = 0;

void ForEachManifestVersionTreeNodeRef(
    GenerationNumber generation_number, uint8_t version_tree_arity_log2,
    absl::FunctionRef<void(GenerationNumber min_generation_number,
                           GenerationNumber max_generation_number,
                           VersionTreeHeight height)>
        callback) {
  // Determine max generation number of the height 1 node referenced from the
  // manifest (this excludes the final generations that are stored inline in the
  // manifest).
  generation_number = (generation_number - 1) >> version_tree_arity_log2
                                                     << version_tree_arity_log2;
  VersionTreeHeight height = 1;
  while (generation_number) {
    // Round `generation_number - 1` down to the nearest multiple of
    // `2**(((height + 1) * version_tree_arity_log2))`.
    GenerationNumber next_generation_number =
        (generation_number - 1)                    //
        >> (height + 1) * version_tree_arity_log2  //
               << (height + 1) * version_tree_arity_log2;
    GenerationNumber min_generation_number = next_generation_number + 1;
    callback(min_generation_number, generation_number, height);
    ++height;
    generation_number = next_generation_number;
  }
}

absl::Status ValidateManifestVersionTreeNodes(
    VersionTreeArityLog2 version_tree_arity_log2,
    GenerationNumber last_generation_number,
    const std::vector<VersionNodeReference>& entries) {
  const auto max_height = GetMaxVersionTreeHeight(version_tree_arity_log2);
  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries[i];
    if (entry.height == 0 || entry.height > max_height) {
      return absl::DataLossError(absl::StrFormat(
          "entry_height[%d] outside valid range [1, %d]", i, max_height));
    }
    if (entry.generation_number == 0) {
      return absl::DataLossError(
          absl::StrFormat("generation_number[%d] must be non-zero", i));
    }
    if (i > 0) {
      if (entry.generation_number <= entries[i - 1].generation_number) {
        return absl::DataLossError(absl::StrFormat(
            "generation_number[%d]=%d <= generation_number[%d]=%d", i,
            entry.generation_number, i - 1, entries[i - 1].generation_number));
      }
      if (entry.height >= entries[i - 1].height) {
        return absl::DataLossError(
            absl::StrFormat("entry_height[%d]=%d >= entry_height[%d]=%d", i,
                            entry.height, i - 1, entries[i - 1].height));
      }
    }
  }
  size_t i = entries.size();
  absl::Status status;
  ForEachManifestVersionTreeNodeRef(
      last_generation_number, version_tree_arity_log2,
      [&](GenerationNumber min_generation_number,
          GenerationNumber max_generation_number, VersionTreeHeight height) {
        if (!status.ok()) {
          // Error already found.
          return;
        }
        if (i == 0) {
          // Height not present (not an error).
          return;
        }
        auto& entry = entries[i - 1];
        if (entry.height != height) {
          // Height not present
          return;
        }
        --i;
        if (entry.generation_number < min_generation_number ||
            entry.generation_number > max_generation_number) {
          status = absl::DataLossError(
              absl::StrFormat("generation_number[%d]=%d is outside expected "
                              "range [%d, %d] for height %d",
                              i, entry.generation_number, min_generation_number,
                              max_generation_number, entry.height));
        }
      });
  if (!status.ok()) return status;
  if (i != 0) {
    return absl::DataLossError(
        absl::StrFormat("Unexpected child with generation_number[%d]=%d and "
                        "entry_height[%d]=%d given last generation_number=%d",
                        i - 1, entries[i - 1].generation_number, i - 1,
                        entries[i - 1].height, last_generation_number));
  }
  return absl::OkStatus();
}

bool ReadManifestVersionTreeNodes(
    riegeli::Reader& reader, VersionTreeArityLog2 version_tree_arity_log2,
    const DataFileTable& data_file_table,
    std::vector<VersionNodeReference>& version_tree_nodes,
    GenerationNumber last_generation_number) {
  const size_t max_num_entries =
      GetMaxVersionTreeHeight(version_tree_arity_log2);
  if (!VersionTreeInteriorNodeEntryArrayCodec<DataFileTable>{
          data_file_table, max_num_entries, /*include_entry_height=*/true}(
          reader, version_tree_nodes)) {
    return false;
  }
  TENSORSTORE_RETURN_IF_ERROR(
      ValidateManifestVersionTreeNodes(
          version_tree_arity_log2, last_generation_number, version_tree_nodes),
      reader.Fail(_), false);
  return true;
}

Result<absl::Cord> EncodeManifest(const Manifest& manifest,
                                  bool encode_as_single) {
#ifndef NDEBUG
  CheckManifestInvariants(manifest, encode_as_single);
#endif
  return EncodeWithOptionalCompression(
      manifest.config, kManifestMagic, kManifestFormatVersion,
      [&](riegeli::Writer& writer) -> bool {
        if (encode_as_single) {
          Config new_config = manifest.config;
          new_config.manifest_kind = ManifestKind::kSingle;
          if (!ConfigCodec{}(writer, new_config)) return false;
        } else {
          if (!ConfigCodec{}(writer, manifest.config)) return false;
          if (manifest.config.manifest_kind != ManifestKind::kSingle) {
            // This is a config-only manifest.
            return true;
          }
        }
        DataFileTableBuilder data_file_table;
        internal_ocdbt::AddDataFiles(data_file_table, manifest.versions);
        internal_ocdbt::AddDataFiles(data_file_table,
                                     manifest.version_tree_nodes);
        if (!data_file_table.Finalize(writer)) return false;
        if (!WriteVersionTreeNodeEntries(manifest.config, writer,
                                         data_file_table, manifest.versions)) {
          return false;
        }
        if (!VersionTreeInteriorNodeEntryArrayCodec<DataFileTableBuilder>{
                data_file_table,
                /*max_num_entries=*/
                GetMaxVersionTreeHeight(
                    manifest.config.version_tree_arity_log2),
                /*include_height=*/true}(writer, manifest.version_tree_nodes)) {
          return false;
        }
        return true;
      });
}

Result<Manifest> DecodeManifest(const absl::Cord& encoded) {
  Manifest manifest;
  auto status = DecodeWithOptionalCompression(
      encoded, kManifestMagic, kManifestFormatVersion,
      [&](riegeli::Reader& reader, uint32_t version) -> bool {
        if (!ConfigCodec{}(reader, manifest.config)) return false;
        if (manifest.config.manifest_kind != ManifestKind::kSingle) {
          // This is a config-only manifest.
          return true;
        }
        DataFileTable data_file_table;
        if (!ReadDataFileTable(reader, /*base_path=*/{}, data_file_table)) {
          return false;
        }
        if (!ReadVersionTreeLeafNode(manifest.config.version_tree_arity_log2,
                                     reader, data_file_table,
                                     manifest.versions)) {
          return false;
        }
        if (!ReadManifestVersionTreeNodes(
                reader, manifest.config.version_tree_arity_log2,
                data_file_table, manifest.version_tree_nodes,
                manifest.versions.back().generation_number)) {
          return false;
        }
        return true;
      });
  if (!status.ok()) {
    return tensorstore::MaybeAnnotateStatus(status, "Error decoding manifest");
  }
#ifndef NDEBUG
  CheckManifestInvariants(manifest);
#endif
  return manifest;
}

bool operator==(const Manifest& a, const Manifest& b) {
  return a.config == b.config && a.versions == b.versions &&
         a.version_tree_nodes == b.version_tree_nodes;
}

std::ostream& operator<<(std::ostream& os, const Manifest& e) {
  os << "{config=" << e.config;
  if (e.config.manifest_kind == ManifestKind::kSingle) {
    os << ", versions=" << tensorstore::span(e.versions)
       << ", version_tree_nodes=" << tensorstore::span(e.version_tree_nodes);
  }
  return os << "}";
}

std::string GetManifestPath(std::string_view base_path) {
  return tensorstore::StrCat(base_path, "manifest.ocdbt");
}

std::string GetNumberedManifestPath(std::string_view base_path,
                                    GenerationNumber generation_number) {
  return absl::StrFormat("%smanifest.%016x", base_path, generation_number);
}

#ifndef NDEBUG
void CheckManifestInvariants(const Manifest& manifest, bool assume_single) {
  assert(manifest.config.version_tree_arity_log2 > 0);
  assert(manifest.config.version_tree_arity_log2 <= kMaxVersionTreeArityLog2);
  if (manifest.config.manifest_kind == ManifestKind::kSingle || assume_single) {
    TENSORSTORE_CHECK_OK(ValidateVersionTreeLeafNodeEntries(
        manifest.config.version_tree_arity_log2, manifest.versions));
    TENSORSTORE_CHECK_OK(ValidateManifestVersionTreeNodes(
        manifest.config.version_tree_arity_log2,
        manifest.versions.back().generation_number,
        manifest.version_tree_nodes));
  } else {
    assert(manifest.versions.empty());
    assert(manifest.version_tree_nodes.empty());
  }
}
#endif  // NDEBUG

}  // namespace internal_ocdbt
}  // namespace tensorstore
