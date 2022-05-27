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

#ifndef TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_METHOD_JSON_BINDER_H_
#define TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_METHOD_JSON_BINDER_H_

#include <string_view>

#include "tensorstore/downsample_method.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_downsample {

constexpr inline auto DownsampleMethodJsonBinder =
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      namespace jb = internal_json_binding;
      return jb::Enum<DownsampleMethod, std::string_view>({
          {DownsampleMethod::kStride, "stride"},
          {DownsampleMethod::kMean, "mean"},
          {DownsampleMethod::kMin, "min"},
          {DownsampleMethod::kMax, "max"},
          {DownsampleMethod::kMedian, "median"},
          {DownsampleMethod::kMode, "mode"},
      })(is_loading, options, obj, j);
    };

}  // namespace internal_downsample
namespace internal_json_binding {

template <>
constexpr inline auto DefaultBinder<DownsampleMethod> =
    internal_downsample::DownsampleMethodJsonBinder;

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_DOWNSAMPLE_DOWNSAMPLE_METHOD_JSON_BINDER_H_
