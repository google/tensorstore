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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/bytes/std_io.h"
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/image/avif_writer.h"
#include "tensorstore/internal/image/image_info.h"
#include "tensorstore/internal/image/image_writer.h"
#include "tensorstore/internal/image/jpeg_writer.h"
#include "tensorstore/internal/image/png_writer.h"
#include "tensorstore/internal/image/webp_writer.h"
#include "tensorstore/internal/init_tensorstore.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Context;
using ::tensorstore::Index;
using ::tensorstore::StrCat;
using ::tensorstore::internal_image::AvifWriter;
using ::tensorstore::internal_image::ImageInfo;
using ::tensorstore::internal_image::ImageWriter;
using ::tensorstore::internal_image::JpegWriter;
using ::tensorstore::internal_image::PngWriter;
using ::tensorstore::internal_image::WebPWriter;

template <typename InputArray>
absl::Status Validate(const InputArray& input) {
  std::vector<std::string> errors;
  if (input.rank() != 2 && input.rank() != 3) {
    errors.push_back(tensorstore::StrCat("expected input of rank 2 or 3, not ",
                                         input.rank()));
  }

  // Validate data types
  if (input.dtype() != tensorstore::dtype_v<uint8_t> &&
      input.dtype() != tensorstore::dtype_v<char>) {
    errors.push_back("expected input.dtype of uint8 or char");
  }

  // Validate shapes
  auto input_shape = input.domain().shape();
  if (input_shape[0] <= 0 || input_shape[1] <= 0) {
    errors.push_back(tensorstore::StrCat("input.shape of ", input_shape,
                                         " has invalid x,y dimensions"));
  }
  auto c = input.rank() - 1;
  if (input.rank() > 2 && input_shape[c] != 1 && input_shape[c] != 3) {
    errors.push_back(tensorstore::StrCat("input.shape of ", input_shape,
                                         " has invalid c dimension"));
  }

  if (!errors.empty()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "tensorstore validation failed: ", absl::StrJoin(errors, ", ")));
  }
  return absl::OkStatus();
}

/// Load a 2d tensorstore volume slice and render it as an image.
absl::Status Run(tensorstore::Spec input_spec, std::string output_filename) {
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
      std::in_place, tensorstore::IdentityTransform(input.domain()));

  std::cerr << std::endl << "Before: " << *transform << std::endl;

  // By convention, assume that the first dimension is Y, and the second is X,
  // and the third is C. The C++ api could use some help with labelling missing
  // dimensions, actually...
  bool has_x = false;
  bool has_y = false;
  bool has_c = false;
  for (auto& l : input.domain().labels()) {
    has_x = has_x || l == "x";
    has_y = has_y || l == "y";
    has_c = has_c || l == "c";
  }
  if (has_y) {
    transform = transform | tensorstore::Dims("y").MoveTo(0);
  }
  if (has_x) {
    transform = transform | tensorstore::Dims("x").MoveTo(1);
  }
  if (has_c) {
    transform = transform | tensorstore::Dims("c").MoveToBack();
  }
  if (input.rank() > 2) {
    transform = transform | tensorstore::DimRange(2, -1).IndexSlice(0);
  }

  auto constrained_input = input | *transform;
  TENSORSTORE_RETURN_IF_ERROR(constrained_input);

  TENSORSTORE_RETURN_IF_ERROR(Validate(*constrained_input));

  std::cerr << "Spec: " << *(constrained_input->spec()) << std::endl;

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto slice,
      tensorstore::Read<tensorstore::zero_origin>(constrained_input).result());

  auto shape_yxc = slice.shape();
  ImageInfo info{/*height=*/static_cast<int32_t>(shape_yxc[0]),
                 /*width=*/static_cast<int32_t>(shape_yxc[1]),
                 /*num_components=*/slice.rank() == 2
                     ? 1
                     : static_cast<int32_t>(shape_yxc[2])};

  std::unique_ptr<ImageWriter> writer;
  std::unique_ptr<riegeli::Writer> output;

  // Select the image format.
  if (absl::EndsWith(output_filename, ".jpg") ||
      absl::EndsWith(output_filename, ".jpeg")) {
    writer = std::make_unique<JpegWriter>();
  } else if (absl::EndsWith(output_filename, ".avif")) {
    writer = std::make_unique<AvifWriter>();
  } else if (absl::EndsWith(output_filename, ".webp")) {
    writer = std::make_unique<WebPWriter>();
  } else if (absl::EndsWith(output_filename, ".png") ||
             output_filename == "-") {
    writer = std::make_unique<PngWriter>();
  } else {
    return absl::InvalidArgumentError(
        "Only .jpeg, .webp, .avif, and .png output formats permitted");
  }

  // Maybe output to stdout.
  if (output_filename == "-" || absl::StartsWith(output_filename, "-.")) {
    // TODO: Also check istty.
    output = std::make_unique<riegeli::StdOut>();
  } else {
    output = std::make_unique<riegeli::FdWriter<>>(output_filename);
  }
  if (!output->ok()) return output->status();

  // And encode the image.
  TENSORSTORE_RETURN_IF_ERROR(writer->Initialize(output.get()));
  TENSORSTORE_RETURN_IF_ERROR(writer->Encode(
      info,
      tensorstore::span(reinterpret_cast<const unsigned char*>(slice.data()),
                        info.width * info.height * info.num_components)));
  return writer->Done();
}

}  // namespace

tensorstore::Spec DefaultInputSpec() {
  return tensorstore::Spec::FromJson(
             {
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
             })
      .value();
}

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
///         "input_labels": ["x", "y"],
///         "input_inclusive_min":[320553,177054],
///         "input_shape":[512,512],
///         "output":[{"input_dimension":0},
///                   {"input_dimension":1},
///                   {"offset":3667},{}]}
///   }'
///
/// And this just copies the image:
///
///   --input_spec='{
///     "driver":"png",
///     "kvstore":"file:///home/data/myfile.png",
///     "domain": { "labels": ["y", "x", "c"] }
///   }'
///
ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, input_spec,
          DefaultInputSpec(), "tensorstore JSON input specification");

/// Required. The output file. Must be a .jpeg, .webp, .avif, or .png.
ABSL_FLAG(std::string, output_file, "-",
          "Slice will be written to this image file; use - for STDOUT");

int main(int argc, char** argv) {
  tensorstore::InitTensorstore(&argc, &argv);

  if (absl::GetFlag(FLAGS_output_file).empty()) {
    std::cerr << "Missing required flag: --output_file" << std::endl;
    return 2;
  }
  std::cerr << "--input_spec="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_input_spec)) << std::endl;

  auto status = Run(absl::GetFlag(FLAGS_input_spec).value,
                    absl::GetFlag(FLAGS_output_file));

  if (!status.ok()) {
    std::cerr << status << std::endl;
  }
  return status.ok() ? 0 : 1;
}
