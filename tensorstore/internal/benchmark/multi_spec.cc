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

#include "tensorstore/internal/benchmark/multi_spec.h"

#include <stddef.h>

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_benchmark {

std::vector<tensorstore::Spec> ReadSpecsFromFile(
    const std::string& txt_file_path) {
  if (txt_file_path.empty()) {
    return {};
  }
  std::string spec_data;
  auto read_status =
      riegeli::ReadAll(riegeli::FdReader(txt_file_path), spec_data);
  if (!read_status.ok() || spec_data.empty()) {
    ABSL_LOG(INFO) << "Failed to read " << txt_file_path << ": " << read_status;
    return {};
  }

  std::vector<tensorstore::Spec> specs;
  {
    std::string json_data = spec_data;
    ::nlohmann::json j = ::nlohmann::json::parse(json_data, nullptr, false);
    auto json_status = internal_json_binding::DefaultBinder<>(
        std::true_type{}, internal_json_binding::NoOptions{}, &specs, &j);
    if (json_status.ok()) {
      return specs;
    }
    ABSL_LOG(ERROR) << "Failed to parse " << txt_file_path << " "
                    << json_data.size() << " bytes as a json: " << json_status;
  }

  auto try_parse_spec = [&specs](std::string_view line) {
    ::nlohmann::json j =
        ::nlohmann::json::parse(std::string(line), nullptr, false);
    auto spec = tensorstore::Spec::FromJson(j);
    if (spec.ok()) {
      specs.push_back(std::move(spec).value());
    } else {
      ABSL_LOG(ERROR) << "Failed to parse " << line << ": " << spec.status();
    }
  };

  std::string_view spec_data_view = spec_data;
  size_t i = 0;
  size_t d = 0;
  size_t start = 0;
  while (i < spec_data_view.size()) {
    auto c = spec_data_view[i++];
    if (c == '{') {
      if (d == 0) start = i;
      d += 1;
    } else if (c == '}') {
      d -= 1;
      if (d == 0) {
        start = i;
        try_parse_spec(spec_data_view.substr(start, i - start));
      }
    }
  }
  if (start < i) {
    try_parse_spec(spec_data_view.substr(start, i - start));
  }
  if (specs.empty()) {
    ABSL_LOG(ERROR) << "Failed to parse any specs from " << txt_file_path;
  }
  return specs;
}

}  // namespace internal_benchmark
}  // namespace tensorstore
