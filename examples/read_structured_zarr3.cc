// Copyright 2024 The TensorStore Authors
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

// Standalone test for reading structured data from a Zarr v3 array.
//
// This test opens an existing zarr3 array with structured data type,
// reads the "inline" field, and prints all values.
//
// Usage:
//   bazel run //examples:read_structured_zarr3 -- /path/to/zarr/array
//
// Or with cmake:
//   cd examples/build && ./read_structured_zarr3

#include <stdint.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(std::string, zarr_path,
          "/home/ubuntu/source/tensorstore/filt_mig.mdio/headers",
          "Path to the zarr3 array directory");

namespace {

using ::tensorstore::Index;

// Field layout from the zarr.json metadata:
// The structured dtype has the following fields with their byte offsets:
//   trace_seq_num_line: int32 @ 0
//   trace_seq_num_reel: int32 @ 4
//   ... (many more fields) ...
//   inline: int32 @ 180
//   crossline: int32 @ 184
//   cdp_x: int32 @ 188
//   cdp_y: int32 @ 192
//
// Total struct size: 196 bytes (matches blosc typesize)

constexpr size_t kInlineFieldOffset = 180;
constexpr size_t kStructSize = 196;

// Read and parse the zarr.json metadata to display info about structured type
void PrintZarrMetadata(const std::string& zarr_path) {
  std::string metadata_path = zarr_path + "/zarr.json";
  std::ifstream file(metadata_path);
  if (!file.is_open()) {
    std::cerr << "Could not open " << metadata_path << std::endl;
    return;
  }

  nlohmann::json metadata;
  try {
    file >> metadata;
  } catch (const nlohmann::json::parse_error& e) {
    std::cerr << "Failed to parse zarr.json: " << e.what() << std::endl;
    return;
  }

  std::cout << "\n=== Zarr Metadata ===" << std::endl;
  std::cout << "Shape: " << metadata["shape"].dump() << std::endl;
  std::cout << "Dimension names: " << metadata["dimension_names"].dump()
            << std::endl;

  if (metadata.contains("data_type")) {
    auto& dt = metadata["data_type"];
    std::cout << "\nData type format:" << std::endl;
    if (dt.is_object()) {
      std::cout << "  Type: object with name=\"" << dt["name"].get<std::string>()
                << "\"" << std::endl;
      if (dt.contains("configuration") &&
          dt["configuration"].contains("fields")) {
        auto& fields = dt["configuration"]["fields"];
        std::cout << "  Number of fields: " << fields.size() << std::endl;
        std::cout << "  Fields:" << std::endl;
        size_t byte_offset = 0;
        for (const auto& field : fields) {
          std::string name = field[0].get<std::string>();
          std::string type = field[1].get<std::string>();
          size_t size = (type == "int32" || type == "uint32" || type == "float32")
                            ? 4
                            : 2;  // int16/uint16
          std::cout << "    " << name << ": " << type << " @ byte " << byte_offset
                    << std::endl;
          byte_offset += size;
        }
        std::cout << "  Total struct size: " << byte_offset << " bytes"
                  << std::endl;
      }
    } else if (dt.is_string()) {
      std::cout << "  Type: simple \"" << dt.get<std::string>() << "\""
                << std::endl;
    } else if (dt.is_array()) {
      std::cout << "  Type: array with " << dt.size() << " fields" << std::endl;
    }
  }

  if (metadata.contains("codecs")) {
    std::cout << "\nCodecs: " << metadata["codecs"].dump(2) << std::endl;
  }
}

absl::Status Run(const std::string& zarr_path) {
  std::cout << "=== Zarr v3 Structured Data Type Test ===" << std::endl;
  std::cout << "Opening zarr3 array at: " << zarr_path << std::endl;

  // First, display metadata information
  PrintZarrMetadata(zarr_path);

  auto context = tensorstore::Context::Default();

  // Create spec for opening the zarr3 array
  // Note: "field" is at the driver level, not inside kvstore (same as zarr v2)
  ::nlohmann::json spec_json = {
      {"driver", "zarr3"},
      {"kvstore",
       {
           {"driver", "file"},
           {"path", zarr_path + "/"},
       }},
      {"field", "inline"},  // Field at byte offset 180
  };

  std::cout << "\n=== Opening TensorStore ===" << std::endl;
  std::cout << "Spec: " << spec_json.dump(2) << std::endl;

  // Open the TensorStore
  auto open_result =
      tensorstore::Open(spec_json, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result();

  if (!open_result.ok()) {
    std::cout << "\n=== Open Failed ===" << std::endl;
    std::cout << "Status: " << open_result.status() << std::endl;
    std::cout << "\nThis error is expected if the zarr3 driver's dtype parsing\n"
              << "does not yet support the extended structured data type format:\n"
              << "  {\"name\": \"structured\", \"configuration\": {\"fields\": [...]}}\n"
              << std::endl;
    std::cout << "The dtype.cc ParseDTypeNoDerived() function currently handles:\n"
              << "  1. String format: \"int32\"\n"
              << "  2. Array format: [[\"field1\", \"int32\"], ...]\n"
              << "\nBut the zarr.json uses the extended object format shown above."
              << std::endl;
    return open_result.status();
  }

  auto store = std::move(open_result).value();

  // Get information about the array
  auto domain = store.domain();
  std::cout << "\n=== Array Info ===" << std::endl;
  std::cout << "Domain: " << domain << std::endl;
  std::cout << "Dtype: " << store.dtype() << std::endl;
  std::cout << "Rank: " << store.rank() << std::endl;

  auto shape = domain.shape();
  std::cout << "Shape: [";
  for (int i = 0; i < shape.size(); ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << shape[i];
  }
  std::cout << "]" << std::endl;

  // Read all data
  std::cout << "\n=== Reading Data ===" << std::endl;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array, tensorstore::Read<tensorstore::zero_origin>(store).result());

  std::cout << "Read complete. Array size: " << array.num_elements()
            << " elements" << std::endl;
  std::cout << "Data type: " << array.dtype() << std::endl;

  // Since field="inline" was specified, the array contains just int32 values
  // directly - no struct extraction needed!
  Index num_inline = shape[0];
  Index num_crossline = shape[1];

  std::cout << "\n=== Inline field values (shape: " << num_inline << " x "
            << num_crossline << ") ===" << std::endl;

  // Cast to int32 pointer since the data is already the inline field values
  auto int_ptr = reinterpret_cast<const int32_t*>(array.data());

  // Print first 10 rows (or fewer if less data)
  Index rows_to_print = std::min(num_inline, Index{10});
  Index cols_to_print = std::min(num_crossline, Index{10});

  for (Index i = 0; i < rows_to_print; ++i) {
    for (Index j = 0; j < cols_to_print; ++j) {
      std::cout << int_ptr[i * num_crossline + j];
      if (j < cols_to_print - 1) {
        std::cout << "\t";
      }
    }
    if (num_crossline > cols_to_print) {
      std::cout << "\t...";
    }
    std::cout << std::endl;
  }
  if (num_inline > rows_to_print) {
    std::cout << "... (" << (num_inline - rows_to_print) << " more rows)"
              << std::endl;
  }

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Successfully read " << (num_inline * num_crossline)
            << " inline values" << std::endl;
  
  // Show some statistics
  int32_t min_val = int_ptr[0], max_val = int_ptr[0];
  int64_t sum = 0;
  for (Index i = 0; i < num_inline * num_crossline; ++i) {
    min_val = std::min(min_val, int_ptr[i]);
    max_val = std::max(max_val, int_ptr[i]);
    sum += int_ptr[i];
  }
  std::cout << "Min value: " << min_val << std::endl;
  std::cout << "Max value: " << max_val << std::endl;
  std::cout << "Mean value: " << (static_cast<double>(sum) / (num_inline * num_crossline)) << std::endl;

  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string zarr_path = absl::GetFlag(FLAGS_zarr_path);
  if (zarr_path.empty()) {
    std::cerr << "Error: --zarr_path is required" << std::endl;
    return 1;
  }

  auto status = Run(zarr_path);
  if (!status.ok()) {
    std::cerr << "\nFinal status: " << status << std::endl;
    return 1;
  }

  return 0;
}
