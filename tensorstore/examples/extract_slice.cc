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

// Extracts a slice of a volumetric dataset, outputtting it as a 2d jpeg image.
//
// extract_slice --output_file=/tmp/foo.jpg --input_spec=...

#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/json.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/compression/jpeg.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::StrCat;

namespace jpeg = tensorstore::jpeg;

template <typename InputArray>
absl::Status Validate(const InputArray& input) {
  std::vector<std::string> errors;
  if (input.rank() != 2 && input.rank() != 3) {
    errors.push_back(
        StrCat("expected input of rank 2 or 3, not ", input.rank()));
  }

  // Validate data types
  if (input.dtype() != tensorstore::dtype_v<uint8_t> &&
      input.dtype() != tensorstore::dtype_v<char>) {
    errors.push_back("expected input.dtype of uint8 or char");
  }

  // Validate shapes
  auto input_shape = input.domain().shape();
  if (input_shape[0] <= 0 || input_shape[1] <= 0) {
    errors.push_back(
        StrCat("input.shape of ", input_shape, " has invalid x,y dimensions"));
  }
  auto c = input.rank() - 1;
  if (input.rank() > 2 && input_shape[c] != 1 && input_shape[c] != 3) {
    errors.push_back(
        StrCat("input.shape of ", input_shape, " has invalid c dimension"));
  }

  if (!errors.empty()) {
    return absl::InvalidArgumentError(
        StrCat("tensorstore validation failed: ", absl::StrJoin(errors, ", ")));
  }
  return absl::OkStatus();
}

/// Load a 2d tensorstore volume slice and render it as a jpeg.
absl::Status Run(::nlohmann::json input_spec, std::string output_filename) {
  auto context = Context::Default();

  // Open input tensorstore and resolve the bounds.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto input,
      tensorstore::Open(input_spec, context, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read)
          .result());

  /// To render something other than the top-layer, A spec should
  /// include a transform.
  tensorstore::Result<tensorstore::IndexTransform<>> transform(
      std::in_place, tensorstore::IdentityTransform(input.rank()));
  if (input.rank() > 2) {
    /// c is the last dimension; eliminate the others.
    const auto c = input.rank() - 1;
    if (input.rank() > 3) {
      transform = transform | tensorstore::DimRange(2, -1).IndexSlice(0);
    }
    /// eliminate c if it is not rank 3.
    if (input.domain().shape()[c] != 3) {
      transform = transform | tensorstore::Dims(-1).IndexSlice(0);
    }
  }

  auto constrained_input = input | *transform;
  TENSORSTORE_RETURN_IF_ERROR(constrained_input);

  TENSORSTORE_RETURN_IF_ERROR(Validate(*constrained_input));

  std::cerr << "Spec: " << *(constrained_input->spec()) << std::endl;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto slice,
      tensorstore::Read<tensorstore::zero_origin>(constrained_input).result());

  // Construct a jpeg from the slice.
  int num_components = slice.rank() == 2 ? 1 : slice.shape()[2];
  absl::Cord encoded;
  jpeg::EncodeOptions options;
  TENSORSTORE_RETURN_IF_ERROR(jpeg::Encode(
      static_cast<const unsigned char*>(slice.data()), slice.shape()[0],
      slice.shape()[1], num_components, options, &encoded));

  auto write_jpeg = [&encoded](auto& out) {
    for (auto chunk = encoded.chunk_begin(); chunk != encoded.chunk_end();
         ++chunk) {
      out.write(chunk->data(), chunk->size());
    }
  };

  if (output_filename == "-") {
    write_jpeg(std::cout);
  } else {
    std::ofstream out(output_filename, std::ios::binary);
    write_jpeg(out);
    out.close();
  }

  return absl::OkStatus();
}

}  // namespace

::nlohmann::json DefaultInputSpec() {
  return ::nlohmann::json({
      {"open", true},
      {"driver", "n5"},
      {"kvstore", {{"driver", "memory"}}},
      {"path", "input"},
      {"metadata",
       {
           {"compression", {{"type", "raw"}}},
           {"dataType", "uint8"},
           {"blockSize", {16, 16, 1}},
           {"dimensions", {64, 64, 1}},
       }},
  });
}

struct JsonFlag {
  JsonFlag() : json(::nlohmann::json::value_t::discarded) {}
  JsonFlag(::nlohmann::json j) : json(j) {}
  ::nlohmann::json json;
};
std::string AbslUnparseFlag(JsonFlag j) {
  if (j.json.is_discarded()) {
    return {};
  }
  return absl::UnparseFlag(j.json.dump());
}
bool AbslParseFlag(std::string_view in, JsonFlag* out, std::string* error) {
  if (in.empty()) {
    out->json = ::nlohmann::json::value_t::discarded;
    return true;
  }
  out->json = ::nlohmann::json::parse(in, nullptr, false);
  if (!out->json.is_object()) {
    *error = "Failed to parse json flag.";
  }
  return out->json.is_object();
}

/// Required. The output jpeg filepath.
ABSL_FLAG(std::string, output_file, "",
          "Slice will be written to this image file; use - for STDOUT");

/// Required. The DefaultInputSpec() renders a 64x64 black square.
///
/// Specify a transform along with the spec to select a specific region.
/// For example, this renders a 512x512 region from the middle of the H01
/// dataset release.
///
///   --input_spec='{
///     "driver":"neuroglancer_precomputed",
///     "kvstore":{"bucket":"h01-release","driver":"gcs"},
///     "path":"data/20210601/4nm_raw",
///     "scale_metadata":{ "resolution":[8,8,33] },
///     "transform":{
///         "input_inclusive_min":[320553,177054],
///         "input_shape":[512,512],
///         "output":[{"input_dimension":0},
///                   {"input_dimension":1},
///                   {"offset":3667},{}]}
///   }'
ABSL_FLAG(JsonFlag, input_spec, DefaultInputSpec(),
          "tensorstore JSON input specification");

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);

  if (absl::GetFlag(FLAGS_output_file).empty()) {
    std::cerr << "Missing required flag: --output_file" << std::endl;
    return 2;
  }
  std::cerr << "--input_spec="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_input_spec)) << std::endl;

  auto status = Run(absl::GetFlag(FLAGS_input_spec).json,
                    absl::GetFlag(FLAGS_output_file));

  if (!status.ok()) {
    std::cerr << status << std::endl;
  }
  return status.ok() ? 0 : 1;
}
