// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_TIFF_COMPRESSOR_H_
#define TENSORSTORE_DRIVER_TIFF_COMPRESSOR_H_

// Include the base class required by the JsonSpecifiedCompressor registry
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_binding/bindable.h"  // For binder macro

namespace tensorstore {
namespace internal_tiff {

class Compressor : public internal::JsonSpecifiedCompressor::Ptr {
 public:
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(
      Compressor, internal::JsonSpecifiedCompressor::FromJsonOptions,
      internal::JsonSpecifiedCompressor::ToJsonOptions)
};
}  // namespace internal_tiff
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_TIFF_COMPRESSOR_H_
