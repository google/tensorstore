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

// Standalone test for reading structured data from Zarr v3 arrays.
//
// This test opens two Zarr v3 arrays:
// 1. A structured array with named fields (headers/)
// 2. A raw bytes array containing struct data (raw_headers/)
//
// Both arrays should contain the same data, allowing comparison of:
// - Field-based access vs manual byte extraction
// - Structured dtype parsing vs raw byte handling
//
// Usage:
//   bazel run //examples:read_structured_zarr3 -- /path/to/parent/dir
//
// Or with cmake:
//   cd examples/build && ./read_structured_zarr3 --zarr_path=/path/to/parent/dir
//
// Where the parent dir contains both 'headers/' and 'raw_headers/' subdirs.

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

// Internal headers for testing dtype parsing
#include "tensorstore/driver/zarr3/dtype.h"

// Additional headers for string operations
#include "absl/strings/str_join.h"

ABSL_FLAG(std::string, zarr_path,
          "/home/ubuntu/source/tensorstore/filt_mig.mdio",
          "Path to the parent .mdio directory containing headers/ and raw_headers/");

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

// Helper function to read and display inline field from an array
absl::Status ReadInlineField(const tensorstore::TensorStore<>& store,
                           const std::string& array_name,
                           bool is_raw_bytes = false) {
  // Get information about the array
  auto domain = store.domain();
  std::cout << "\n=== " << array_name << " Array Info ===" << std::endl;
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
  std::cout << "\n=== Reading " << array_name << " Data ===" << std::endl;
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto array, tensorstore::Read<tensorstore::zero_origin>(store).result());

  std::cout << "Read complete. Array size: " << array.num_elements()
            << " elements" << std::endl;
  std::cout << "Data type: " << array.dtype() << std::endl;

  Index num_inline, num_crossline;
  const int32_t* int_ptr;

  if (is_raw_bytes) {
    // For raw bytes, we need to extract the inline field manually
    // Shape is [inline, crossline, struct_size]
    num_inline = shape[0];
    num_crossline = shape[1];
    Index struct_size = shape[2];
    if (struct_size != kStructSize) {
      std::cout << "Warning: Raw struct size (" << struct_size
                << ") differs from expected header struct size (" << kStructSize
                << "). Assuming padding." << std::endl;
    }

    // Extract inline field (4 bytes starting at offset 180)
    auto byte_ptr = reinterpret_cast<const std::byte*>(array.data());
    std::vector<int32_t> inline_values(num_inline * num_crossline);

    for (Index i = 0; i < num_inline; ++i) {
      for (Index j = 0; j < num_crossline; ++j) {
        Index struct_offset = (i * num_crossline + j) * struct_size;
        Index field_offset = struct_offset + kInlineFieldOffset;
        std::memcpy(&inline_values[i * num_crossline + j],
                   byte_ptr + field_offset, 4);
      }
    }

    std::cout << "Extracted inline field from raw bytes at offset "
              << kInlineFieldOffset << std::endl;
    int_ptr = inline_values.data();
  } else {
    // For structured array, field access already gave us int32 values
    num_inline = shape[0];
    num_crossline = shape[1];
    int_ptr = reinterpret_cast<const int32_t*>(array.data());
  }

  std::cout << "\n=== Inline field values from " << array_name
            << " (shape: " << num_inline << " x " << num_crossline << ") ===" << std::endl;

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

  std::cout << "\n=== " << array_name << " Summary ===" << std::endl;
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

absl::Status Run(const std::string& zarr_path) {
  std::cout << "=== Zarr v3 Structured Data Type Test ===" << std::endl;
  std::cout << "Opening zarr3 arrays in: " << zarr_path << std::endl;

  auto context = tensorstore::Context::Default();

  // First, display metadata information for structured array
  std::string headers_path = zarr_path + "/headers";
  PrintZarrMetadata(headers_path);

  // Test raw_bytes parsing by reading and parsing the raw_headers zarr.json
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TESTING RAW_BYTES PARSING" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  std::string raw_metadata_path = zarr_path + "/raw_headers/zarr.json";
  std::ifstream raw_file(raw_metadata_path);
  if (!raw_file.is_open()) {
    std::cout << "Could not open " << raw_metadata_path << std::endl;
    return absl::NotFoundError("Raw headers metadata not found");
  }

  nlohmann::json raw_metadata;
  try {
    raw_file >> raw_metadata;
  } catch (const nlohmann::json::parse_error& e) {
    std::cout << "Failed to parse raw zarr.json: " << e.what() << std::endl;
    return absl::DataLossError("Invalid raw metadata JSON");
  }

  std::cout << "Raw headers data_type: " << raw_metadata["data_type"].dump(2) << std::endl;

  // Test parsing the raw_bytes data type
  std::cout << "Testing raw_bytes dtype parsing..." << std::endl;

  // For now, just verify the JSON structure is what we expect
  if (!raw_metadata.contains("data_type")) {
    std::cout << "FAILED: No data_type in metadata" << std::endl;
    return absl::NotFoundError("Missing data_type");
  }

  auto& dt = raw_metadata["data_type"];
  if (!dt.is_object() || !dt.contains("name") || dt["name"] != "raw_bytes") {
    std::cout << "FAILED: data_type is not raw_bytes extension" << std::endl;
    return absl::InvalidArgumentError("Not raw_bytes extension");
  }

  if (!dt.contains("configuration") || !dt["configuration"].contains("length_bytes")) {
    std::cout << "FAILED: Missing length_bytes in configuration" << std::endl;
    return absl::InvalidArgumentError("Missing length_bytes");
  }

  int length_bytes = dt["configuration"]["length_bytes"];
  std::cout << "SUCCESS: Found raw_bytes extension with length_bytes = " << length_bytes << std::endl;
  std::cout << "This should parse to:" << std::endl;
  std::cout << "  - Single field with byte_t dtype" << std::endl;
  std::cout << "  - Field shape: [" << length_bytes << "]" << std::endl;
  std::cout << "  - Bytes per outer element: " << length_bytes << std::endl;

  // Now actually test the parsing implementation
  std::cout << "\n=== Testing ParseDType Implementation ===" << std::endl;
  auto dtype_result = tensorstore::internal_zarr3::ParseDType(dt);
  if (!dtype_result.ok()) {
    std::cout << "FAILED: Could not parse raw_bytes data type: " << dtype_result.status() << std::endl;
    return dtype_result.status();
  }

  auto dtype = std::move(dtype_result).value();
  std::cout << "SUCCESS: ParseDType worked!" << std::endl;
  std::cout << "  Fields: " << dtype.fields.size() << std::endl;
  std::cout << "  Has fields: " << dtype.has_fields << std::endl;
  std::cout << "  Bytes per outer element: " << dtype.bytes_per_outer_element << std::endl;

  if (!dtype.fields.empty()) {
    const auto& field = dtype.fields[0];
    std::cout << "  Field name: '" << field.name << "'" << std::endl;
    std::cout << "  Field dtype: " << field.dtype << std::endl;
    std::cout << "  Field shape: [" << absl::StrJoin(field.field_shape, ", ") << "]" << std::endl;
    std::cout << "  Field num_inner_elements: " << field.num_inner_elements << std::endl;
    std::cout << "  Field num_bytes: " << field.num_bytes << std::endl;
  }

  // Verify the parsing is correct
  bool parsing_correct = true;
  if (dtype.fields.size() != 1) {
    std::cout << "ERROR: Expected 1 field, got " << dtype.fields.size() << std::endl;
    parsing_correct = false;
  }
  if (dtype.fields[0].name != "") {
    std::cout << "ERROR: Expected empty field name, got '" << dtype.fields[0].name << "'" << std::endl;
    parsing_correct = false;
  }
  if (dtype.fields[0].dtype != tensorstore::dtype_v<tensorstore::dtypes::byte_t>) {
    std::cout << "ERROR: Expected byte_t dtype, got " << dtype.fields[0].dtype << std::endl;
    parsing_correct = false;
  }
  if (dtype.fields[0].field_shape != std::vector<Index>{length_bytes}) {
    std::cout << "ERROR: Expected field shape [" << length_bytes << "], got ["
              << absl::StrJoin(dtype.fields[0].field_shape, ", ") << "]" << std::endl;
    parsing_correct = false;
  }
  if (dtype.bytes_per_outer_element != length_bytes) {
    std::cout << "ERROR: Expected " << length_bytes << " bytes per element, got "
              << dtype.bytes_per_outer_element << std::endl;
    parsing_correct = false;
  }

  if (parsing_correct) {
    std::cout << "\n✅ PARSING VERIFICATION: All checks passed!" << std::endl;
    std::cout << "The raw_bytes extension is correctly parsed." << std::endl;
  } else {
    std::cout << "\n❌ PARSING VERIFICATION: Some checks failed!" << std::endl;
    return absl::InternalError("Parsing verification failed");
  }

  // Test 1: Read from structured array using field access
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TEST 1: Reading from structured 'headers' array" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  ::nlohmann::json headers_spec = ::nlohmann::json::object();
  headers_spec["driver"] = "zarr3";
  headers_spec["kvstore"] = ::nlohmann::json::object();
  headers_spec["kvstore"]["driver"] = "file";
  headers_spec["kvstore"]["path"] = headers_path + "/";
  headers_spec["field"] = "inline";  // Extract inline field (int32 at byte offset 180)

  std::cout << "Spec: " << headers_spec.dump(2) << std::endl;

  auto headers_open_result =
      tensorstore::Open(headers_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result();

  if (!headers_open_result.ok()) {
    std::cout << "\n=== Headers Open Failed ===" << std::endl;
    std::cout << "Status: " << headers_open_result.status() << std::endl;
    return headers_open_result.status();
  }

  auto headers_store = std::move(headers_open_result).value();
  TENSORSTORE_RETURN_IF_ERROR(ReadInlineField(headers_store, "headers"));

  // Test 2: Read from raw bytes array (no special void access needed)
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TEST 2: Reading from raw 'raw_headers' array" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  std::string raw_headers_path = zarr_path + "/raw_headers";
  ::nlohmann::json raw_spec = ::nlohmann::json::object();
  raw_spec["driver"] = "zarr3";
  raw_spec["kvstore"] = ::nlohmann::json::object();
  raw_spec["kvstore"]["driver"] = "file";
  raw_spec["kvstore"]["path"] = raw_headers_path + "/";
  // No field specified - raw_bytes has a single anonymous field

  std::cout << "Spec: " << raw_spec.dump(2) << std::endl;

  auto raw_open_result =
      tensorstore::Open(raw_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result();

  if (!raw_open_result.ok()) {
    std::cout << "\n=== Raw Headers Open Failed ===" << std::endl;
    std::cout << "Status: " << raw_open_result.status() << std::endl;
    return raw_open_result.status();
  }

  auto raw_store = std::move(raw_open_result).value();
  TENSORSTORE_RETURN_IF_ERROR(ReadInlineField(raw_store, "raw_headers", /*is_raw_bytes=*/true));

  // Test 3: Read from headers array as void (field="<void>")
  // Use a fresh context to avoid cache sharing with Test 1
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TEST 3: Reading from 'headers' array as void (field=\"<void>\")" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  auto context_void = tensorstore::Context::Default();

  ::nlohmann::json headers_void_spec = ::nlohmann::json::object();
  headers_void_spec["driver"] = "zarr3";
  headers_void_spec["kvstore"] = ::nlohmann::json::object();
  headers_void_spec["kvstore"]["driver"] = "file";
  headers_void_spec["kvstore"]["path"] = headers_path + "/";
  headers_void_spec["field"] = "<void>";  // Special field for raw byte access

  std::cout << "Spec: " << headers_void_spec.dump(2) << std::endl;

  auto headers_void_open_result =
      tensorstore::Open(headers_void_spec, context_void, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result();

  if (!headers_void_open_result.ok()) {
    std::cout << "\n=== Headers (void) Open Failed ===" << std::endl;
    std::cout << "Status: " << headers_void_open_result.status() << std::endl;
    return headers_void_open_result.status();
  }

  auto headers_void_store = std::move(headers_void_open_result).value();
  TENSORSTORE_RETURN_IF_ERROR(ReadInlineField(headers_void_store, "headers (void)", /*is_raw_bytes=*/true));

  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "COMPARISON: All three methods should give identical inline field values" << std::endl;
  std::cout << std::string(60, '=') << std::endl;
  std::cout << "- Test 1: 'headers' with field=\"inline\" provides field access convenience\n"
            << "- Test 2: 'raw_headers' (raw_bytes type) provides direct byte access\n"
            << "- Test 3: 'headers' with field=\"<void>\" provides raw byte access to structured data\n"
            << "All three extract the inline field from byte offset " << kInlineFieldOffset
            << " in " << kStructSize << "-byte structs." << std::endl;

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

  // Verify the path structure
  std::string headers_path = zarr_path + "/headers";
  std::string raw_headers_path = zarr_path + "/raw_headers";

  std::cout << "Expecting arrays at:" << std::endl;
  std::cout << "  Structured: " << headers_path << std::endl;
  std::cout << "  Raw bytes:  " << raw_headers_path << std::endl;
  std::cout << std::endl;

  auto status = Run(zarr_path);
  if (!status.ok()) {
    std::cerr << "\nFinal status: " << status << std::endl;
    return 1;
  }

  return 0;
}
