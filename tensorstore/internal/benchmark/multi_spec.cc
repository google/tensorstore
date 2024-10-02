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
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/absl_log.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep
#include "tensorstore/json_serialization_options_base.h"

namespace tensorstore {
namespace internal_benchmark {

struct ShardConfig {
  std::vector<ShardVariable> variables;

  constexpr static auto default_json_binder =
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        namespace jb = tensorstore::internal_json_binding;
        using Self = ShardConfig;
        return jb::Projection<&Self::variables>() /**/
            (is_loading, options, obj, j);
      };
};

std::vector<ShardVariable> ReadFromFileOrFlag(std::string flag_value) {
  std::string json_data;
  auto read_status = riegeli::ReadAll(riegeli::FdReader(flag_value), json_data);
  if (!read_status.ok()) {
    ABSL_LOG(INFO) << read_status;
    json_data = flag_value;
  }
  ::nlohmann::json j = ::nlohmann::json::parse(json_data, nullptr, false);
  if (j.is_discarded()) {
    ABSL_LOG(INFO) << "json is discarded: " << json_data;
    return {};
  }
  ShardConfig config;
  read_status = internal_json_binding::DefaultBinder<>(
      std::true_type{}, internal_json_binding::NoOptions{}, &config, &j);
  if (!read_status.ok()) {
    ABSL_LOG(INFO) << read_status;
    return {};
  }

  return std::move(config).variables;
}

}  // namespace internal_benchmark
}  // namespace tensorstore
