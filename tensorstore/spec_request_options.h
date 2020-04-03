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

#ifndef TENSORSTORE_SPEC_REQUEST_OPTIONS_H_
#define TENSORSTORE_SPEC_REQUEST_OPTIONS_H_

#include <optional>

#include "tensorstore/open_mode.h"
#include "tensorstore/staleness_bound.h"

namespace tensorstore {

class MinimalSpec {
 public:
  constexpr explicit MinimalSpec(bool minimal_spec = true)
      : minimal_spec_(minimal_spec) {}
  bool minimal_spec() const { return minimal_spec_; }

 private:
  bool minimal_spec_;
};

/// Options for `tensorstore::TensorStore::spec`.
class SpecRequestOptions : public MinimalSpec {
 public:
  SpecRequestOptions(MinimalSpec minimal_spec = MinimalSpec{false},
                     std::optional<OpenMode> open_mode = {},
                     std::optional<StalenessBounds> staleness = {})
      : MinimalSpec(minimal_spec), open_mode(open_mode), staleness(staleness) {}
  SpecRequestOptions(MinimalSpec minimal_spec, OpenMode open_mode,
                     std::optional<StalenessBounds> staleness = {})
      : MinimalSpec(minimal_spec), open_mode(open_mode), staleness(staleness) {}
  SpecRequestOptions(OpenMode open_mode,
                     std::optional<StalenessBounds> staleness = {})
      : MinimalSpec(false), open_mode(open_mode), staleness(staleness) {}
  SpecRequestOptions(std::optional<OpenMode> open_mode,
                     std::optional<StalenessBounds> staleness = {})
      : MinimalSpec(false), open_mode(open_mode), staleness(staleness) {}
  SpecRequestOptions(MinimalSpec minimal_spec,
                     std::optional<StalenessBounds> staleness)
      : MinimalSpec(minimal_spec), staleness(staleness) {}
  SpecRequestOptions(std::optional<StalenessBounds> staleness)
      : MinimalSpec(false), open_mode(), staleness(staleness) {}

  std::optional<OpenMode> open_mode;
  std::optional<StalenessBounds> staleness;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_SPEC_REQUEST_OPTIONS_H_
