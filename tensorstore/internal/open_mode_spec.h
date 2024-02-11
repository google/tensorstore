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

#ifndef TENSORSTORE_INTERNAL_OPEN_MODE_SPEC_H_
#define TENSORSTORE_INTERNAL_OPEN_MODE_SPEC_H_

/// \file
///
/// Defines `OpenModeSpec` base class for TensorStore driver SpecT types that
/// support `OpenMode`.

#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"

namespace tensorstore {
namespace internal {

/// Specifies an open mode for use as the base class of a driver SpecT type.
struct OpenModeSpec {
  bool open = false;
  bool create = false;
  bool delete_existing = false;
  bool assume_metadata = false;
  bool assume_cached_metadata = false;

  OpenMode open_mode() const {
    return (open ? OpenMode::open : OpenMode{}) |
           (create ? OpenMode::create : OpenMode{}) |
           (delete_existing ? OpenMode::delete_existing : OpenMode{}) |
           (assume_metadata ? OpenMode::assume_metadata : OpenMode{}) |
           (assume_cached_metadata ? OpenMode::assume_cached_metadata
                                   : OpenMode{});
  }

  // For compatibility with `ContextBindingTraits`.
  static constexpr auto ApplyMembers = [](auto& x, auto f) {
    return f(x.open, x.create, x.delete_existing, x.assume_metadata,
             x.assume_cached_metadata);
  };

  /// Applies the specified options.
  absl::Status ApplyOptions(const SpecOptions& options);

  /// Returns an error if `read_write_mode` is invalid with `*this`.
  absl::Status Validate(ReadWriteMode read_write_mode) const;
};

TENSORSTORE_DECLARE_JSON_BINDER(OpenModeSpecJsonBinder, OpenModeSpec,
                                internal_json_binding::NoOptions,
                                IncludeDefaults, ::nlohmann::json::object_t);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OPEN_MODE_SPEC_H_
