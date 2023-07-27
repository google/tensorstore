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

#include <math.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <list>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/marshalling.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <half.hpp>
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "examples/data_type_invoker.h"
#include "absl/flags/parse.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/utf8_string.h"

namespace {

using ::tensorstore::AllDims;
using ::tensorstore::Context;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::MaybeAnnotateStatus;
using ::tensorstore::StrCat;
using ::tensorstore::WriteFutures;
using ::tensorstore_examples::DataTypeIdOf;
using ::tensorstore_examples::MakeDataTypeInvoker;

template <typename T>
struct SupportsLess {
  template <typename LessT>
  constexpr auto operator()(const LessT& t) -> decltype(t < t) const {
    return t < t;
  }
  constexpr operator bool() const {
    return std::is_invocable_r<bool, SupportsLess, T>::value;
  }
};

std::pair<Index, Index> GetWindow(Index location, size_t radius, Index length) {
  if (location < radius) {
    return {0, std::min(length, static_cast<Index>(2 * radius + 1))};
  }
  if (location + radius >= length - 1) {
    return {length - 2 * radius - 1, length};
  }
  return {radius > location ? 0 : location - radius,  //
          location + radius + 1};
}

template <typename InputArray, typename OutputArray>
absl::Status ComputeQuantilesValidator(const InputArray& input,
                                       tensorstore::span<double> quantiles,
                                       const OutputArray& output) {
  auto shape = input.domain().shape();

  // validate input and output shapes.
  std::vector<std::string> errors;
  if (input.rank() != 2) {
    errors.push_back(tensorstore::StrCat("expected input rank 2, got ",
                                         static_cast<int>(input.rank())));
  }
  if (output.rank() != 2) {
    errors.push_back(tensorstore::StrCat("expected output rank 2, got ",
                                         static_cast<int>(output.rank())));
  }
  if (shape[1] == 0) {
    errors.push_back("input rank 1 has zero size");
  }
  if (shape[0] != output.domain().shape()[0]) {
    errors.push_back(
        tensorstore::StrCat("expected dimension 0 shape matching, got input ",
                            shape[0], " vs. ", output.domain().shape()[0]));
  }
  if (output.domain().shape()[1] != quantiles.size()) {
    errors.push_back(
        tensorstore::StrCat("expected output dimension 1 to match q, got ",
                            output.domain().shape()[1]));
  }
  if (!errors.empty()) {
    return absl::InvalidArgumentError(absl::StrJoin(errors, ", "));
  }
  return absl::OkStatus();
}

// Computes the quantiles. Currently assumes that the input is something
// like an ArrayView<void, 2>.
template <typename InputArray, typename OutputArray>
absl::Status ComputeQuantiles(InputArray& input,
                              tensorstore::span<double> quantiles,
                              OutputArray& output) {
  // Validates the input and output parameters are valid.
  // input shape(x, t')
  // output shape(x, q)
  // quantiles shape(q)
  TENSORSTORE_RETURN_IF_ERROR(
      ComputeQuantilesValidator(input, quantiles, output));

  // Compute the indices which correspond to each quantile.
  // mimics axis = 1, interpolation = 'nearest'
  const auto shape = input.domain().shape();
  const Index N = shape[1] - 1;
  std::vector<Index> indices_vector;
  indices_vector.reserve(quantiles.size());
  for (const auto p : quantiles) {
    indices_vector.push_back(static_cast<Index>(std::nearbyint(p * N)));
  }
  const auto indices = tensorstore::MakeArrayView(indices_vector);

  // Allocate an array to hold t' values. We later copy the values for
  // each x into t, and sort them.
  auto values =
      tensorstore::AllocateArray({shape[1]}, tensorstore::c_order,
                                 tensorstore::default_init, input.dtype());

  // sort_values is a lambda which takes an unused value, then, if the type
  // of the unused value is comparable, coerces values to a 1-dimensional
  // array of that same type, and sorts those values.
  //
  // sort_values is invoked via TryInvokeWithDataTypeCast which manages
  // the dtype() based dispatch.
  auto sort_values = MakeDataTypeInvoker([](auto t, auto& values) {
    using T = decltype(t);
    if constexpr (SupportsLess<T>()) {
      T* begin = static_cast<T*>(values.data());
      T* end = begin + values.domain().shape()[0];
      std::sort(begin, end);
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError("unsortable type");
  });

  for (Index x = 0; x < shape[0]; ++x) {
    // Copy input[x, :] into the values.
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::CopyTransformedArray(
            input | Dims(0).TranslateTo(0).IndexSlice(x), values),
        MaybeAnnotateStatus(_, "ComputeQuantiles copying values"));

    // Sort the data.
    TENSORSTORE_RETURN_IF_ERROR(
        sort_values(DataTypeIdOf(values), values),
        MaybeAnnotateStatus(_, "ComputeQuantiles sorting values"));

    // Materialize the indices data into the output.
    TENSORSTORE_RETURN_IF_ERROR(
        tensorstore::CopyTransformedArray(
            values |
                Dims(0).IndexArraySlice(tensorstore::UnownedToShared(indices)),
            output | Dims(0).TranslateTo(0).IndexSlice(x)),
        MaybeAnnotateStatus(_, "ComputeQuantiles copying output"));
  }

  return absl::OkStatus();
}

template <typename InputArray, typename OutputArray>
absl::Status ValidateRun(const InputArray& input, const OutputArray& output,
                         tensorstore::span<double> quantiles, size_t radius) {
  // Validate the ranks of the various tensorstores.
  // Specifically we require the following:
  // input shape(x, y, z, t)
  // output shape(x, y, z, t, q)
  // quantiles shape(q)
  std::vector<std::string> errors;
  if (radius <= 0) {
    errors.push_back("radius must be > 0");
  }
  if (input.rank() != 4) {
    errors.push_back(
        tensorstore::StrCat("expected input rank 4, not ", input.rank()));
  }

  // Validate data types
  if (input.dtype() != output.dtype()) {
    errors.push_back("input and output have mismatching datatypes");
  }
  auto is_sortable = MakeDataTypeInvoker([](auto t) {
    using T = decltype(t);
    if constexpr (SupportsLess<T>()) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError("unsortable type");
  });
  if (!is_sortable(DataTypeIdOf(input)).ok()) {
    errors.push_back("datatype is not natively sortable");
  }

  // Validate shapes
  auto input_shape = input.domain().shape();
  auto output_shape = output.domain().shape();
  if (output_shape[4] != quantiles.size()) {
    errors.push_back(tensorstore::StrCat(
        "output shape[4] is ", output.domain().shape()[4],
        " which does not match the number of quantiles ", quantiles.size()));
  }
  if (output.rank() != 5) {
    errors.push_back(
        tensorstore::StrCat("expected output rank 5, got ", output.rank()));
  }

  // Validate shapes
  if (output_shape[4] != quantiles.size()) {
    errors.push_back(tensorstore::StrCat(
        "output shape[4] is ", output.domain().shape()[4],
        " which does not match the number of quantiles ", quantiles.size()));
  }
  for (int i = 0; i < 4; i++) {
    if (i < input_shape.size() && input.domain().shape()[i] == 0) {
      errors.push_back(tensorstore::StrCat("input dimension ", i, " is 0"));
    }
    if (i < output_shape.size() && output.domain().shape()[i] == 0) {
      errors.push_back(tensorstore::StrCat("output dimension ", i, " is 0"));
    }
    if (i < output_shape.size() && i < input_shape.size() &&
        output_shape[i] > input_shape[i]) {
      errors.push_back(tensorstore::StrCat(
          "output dimension ", i, " is greater than the input dimension, ",
          output_shape[i], " vs ", input_shape[i]));
    }
  }

  if (!errors.empty()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "tensorstore validation failed: ", absl::StrJoin(errors, ", ")));
  }
  return absl::OkStatus();
}

absl::Status Run(tensorstore::Spec input_spec, tensorstore::Spec output_spec,
                 std::vector<double> quantiles, size_t radius) {
  auto context = Context::Default();

  // Open input tensorstore and resolve the bounds.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto input, tensorstore::Open(input_spec, context,
                                    tensorstore::OpenMode::open_or_create,
                                    tensorstore::ReadWriteMode::read_write)
                      .result());

  // Open output tensorstore and resolve the bounds.
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto output,
      tensorstore::Open(output_spec, context, tensorstore::OpenMode::create,
                        tensorstore::ReadWriteMode::read_write)
          .result());

  // Resolve is unnecessary as the tensorstore volumes are unlikely to change
  // bounds, however it causes the spec to include the actual bounds when
  // output, below.
  input = ResolveBounds(input).value();
  output = ResolveBounds(output).value();

  // Validate the ranks of the various tensorstores.
  // Specifically we require the following:
  // input shape(x, y, z, t)
  // output shape(x, y, z, t, q)
  // quantiles shape(q)
  TENSORSTORE_RETURN_IF_ERROR(ValidateRun(input, output, quantiles, radius));

  auto shape = output.domain().shape();
  bool is_constrained = false;
  for (int i = 0; i < 4; i++) {
    if (shape[i] != input.domain().shape()[i]) {
      is_constrained = true;
      break;
    }
  }

  // Constrain the input to the output size & translate all values to 0-origin.
  TENSORSTORE_ASSIGN_OR_RETURN(auto translated_output,
                               output | AllDims().TranslateTo(0));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto constrained_input,
      input | AllDims().TranslateTo(0) |
          Dims(0, 1, 2, 3)
              .HalfOpenInterval(0, {shape[0], shape[1], shape[2], shape[3]}));

  std::cout << "input spec: " << input.spec().value() << std::endl;

  if (is_constrained) {
    std::cout << "constrained input: " << constrained_input.spec().value()
              << std::endl;
  }

  std::cout << "output spec: " << output.spec().value() << std::endl;

  // staging_xt = [].shape(x, t, q)
  auto staging_xtq = tensorstore::AllocateArray(
      {shape[0], shape[3], static_cast<Index>(quantiles.size())},
      tensorstore::c_order, tensorstore::default_init, input.dtype());

  size_t write_failed_count = 0;
  std::list<tensorstore::WriteFutures> pending_writes;

  // Select YZ views.
  for (Index y = 0; y < shape[1]; ++y) {
    for (Index z = 0; z < shape[2]; ++z) {
      // tile_xt = input[:, y, z, :], shape(x, t)
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto tile_xt,
          tensorstore::Read(constrained_input | Dims(1, 2).IndexSlice({y, z}))
              .result());

      // Process each XT tile.
      for (Index t = 0; t < shape[3]; ++t) {
        auto [start, end] = GetWindow(t, radius, shape[3]);

        // staging_xq = staging_xt[:, t, :], shape(x, q)
        TENSORSTORE_ASSIGN_OR_RETURN(  //
            auto staging_xq, staging_xtq | Dims(1).IndexSlice(t),
            MaybeAnnotateStatus(_, "staging_slice "));

        // tile_slice = tile_xt[:, start:end], shape(x, t')
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto tile_slice,
            tile_xt | Dims(1).HalfOpenInterval(start, end).TranslateTo(0),
            MaybeAnnotateStatus(_, "staging_slice"));

        TENSORSTORE_RETURN_IF_ERROR(
            ComputeQuantiles(tile_slice, quantiles, staging_xq));
      }

      // write output[:, y, z, :, :]
      pending_writes.emplace_back(tensorstore::Write(
          staging_xtq, translated_output | Dims(1, 2).IndexSlice({y, z})));

      // cleanup any committed futures.
      for (auto it = pending_writes.begin(); it != pending_writes.end();) {
        if (it->commit_future.ready()) {
          if (!it->commit_future.status().ok()) {
            write_failed_count++;
            std::cout << it->commit_future.status();
          }
          it = pending_writes.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  // Wait for all remaining futures to complete.
  for (auto& front : pending_writes) {
    if (!front.commit_future.status().ok()) {
      write_failed_count++;
      std::cout << front.commit_future.status() << std::endl;
    }
  }

  return (write_failed_count == 0)
             ? absl::OkStatus()
             : absl::UnknownError("At least one write failed, see output");
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
                      {"dataType", "uint16"},
                      {"blockSize", {256, 1, 1, 100}},
                      {"dimensions", {1024, 1, 1, 100}},
                  }},
             })
      .value();
}

tensorstore::Spec DefaultOutputSpec() {
  return tensorstore::Spec::FromJson(
             {
                 {"create", true},
                 {"open", true},
                 {"driver", "n5"},
                 {"kvstore", {{"driver", "memory"}}},
                 {"path", "output"},
                 {"metadata",
                  {
                      {"compression", {{"type", "raw"}}},
                      {"dataType", "uint16"},
                      {"blockSize", {256, 1, 1, 100, 3}},
                      {"dimensions", {1024, 1, 1, 100, 3}},
                  }},
             })
      .value();
}

struct Quantiles {
  Quantiles(std::vector<double> q) : quantiles(q) {}
  std::vector<double> quantiles;
};
std::string AbslUnparseFlag(Quantiles out) {
  return absl::StrJoin(out.quantiles, ",");
}
bool AbslParseFlag(std::string_view in, Quantiles* out, std::string* error) {
  out->quantiles.clear();
  if (in.empty()) {
    *error = "quantiles must not be empty";
    return false;
  }
  for (std::string_view x : absl::StrSplit(in, ',', absl::AllowEmpty())) {
    double v;
    if (!absl::SimpleAtod(x, &v)) {
      *error = "failed to parse double: ";
      *error += x;
      return false;
    }
    out->quantiles.push_back(v);
  }
  return true;
}

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, input_spec,
          DefaultInputSpec(), "tensorstore JSON input specification");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Spec>, output_spec,
          DefaultOutputSpec(), "tensorstore JSON output specification");

ABSL_FLAG(Quantiles, quantiles, std::vector<double>({.1, .5, .9}), "Quantiles");

ABSL_FLAG(size_t, radius, 10, "Radius");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);  // InitTensorstore

  std::cout << "Flags: " << std::endl;
  std::cout << "  --input_spec="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_input_spec)) << std::endl;
  std::cout << "  --output_spec="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_output_spec)) << std::endl;
  std::cout << "  --quantiles="
            << AbslUnparseFlag(absl::GetFlag(FLAGS_quantiles)) << std::endl;
  std::cout << "  --radius=" << absl::GetFlag(FLAGS_radius) << std::endl;

  auto status = Run(absl::GetFlag(FLAGS_input_spec).value,
                    absl::GetFlag(FLAGS_output_spec).value,
                    absl::GetFlag(FLAGS_quantiles).quantiles,
                    absl::GetFlag(FLAGS_radius));

  if (!status.ok()) {
    std::cout << "FAIL " << status << std::endl;
  } else {
    std::cout << "PASS";
  }
  return status.ok() ? 0 : 1;
}
