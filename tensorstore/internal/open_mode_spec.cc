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

#include "tensorstore/internal/open_mode_spec.h"

#include <optional>

#include "tensorstore/internal/json.h"
#include "tensorstore/open_mode.h"

namespace tensorstore {
namespace internal {

namespace jb = tensorstore::internal::json_binding;
namespace {
struct MaybeOpenCreate {
  std::optional<bool> open;
  std::optional<bool> create;
};
}  // namespace

TENSORSTORE_DEFINE_JSON_BINDER(
    OpenModeSpecJsonBinder,
    jb::Sequence(
        jb::GetterSetter(
            [](auto& obj) {
              return MaybeOpenCreate{(obj.open == true && obj.create == true)
                                         ? std::optional<bool>(true)
                                         : std::optional<bool>(),
                                     (obj.create == true)
                                         ? std::optional<bool>(true)
                                         : std::optional<bool>()};
            },
            [](auto& obj, const auto& x) {
              obj.open = x.open || x.create ? x.open.value_or(false) : true;
              obj.create = x.create.value_or(false);
            },
            jb::Sequence(jb::Member("open",
                                    jb::Projection(&MaybeOpenCreate::open)),
                         jb::Member("create",
                                    jb::Projection(&MaybeOpenCreate::create)))),
        jb::Member("delete_existing",
                   jb::Projection(&OpenModeSpec::delete_existing,
                                  jb::DefaultValue([](bool* v) {
                                    *v = false;
                                  }))),
        jb::Member("allow_metadata_mismatch",
                   jb::Projection(&OpenModeSpec::allow_metadata_mismatch,
                                  jb::DefaultValue([](bool* v) {
                                    *v = false;
                                  })))));

absl::Status OpenModeSpec::ConvertSpec(const SpecRequestOptions& options) {
  if (options.open_mode) {
    const OpenMode open_mode = *options.open_mode;
    open = (open_mode & OpenMode::open) == OpenMode::open;
    create = (open_mode & OpenMode::create) == OpenMode::create;
    allow_metadata_mismatch = (open_mode & OpenMode::allow_option_mismatch) ==
                              OpenMode::allow_option_mismatch;
    delete_existing =
        (open_mode & OpenMode::delete_existing) == OpenMode::delete_existing;
  }
  return absl::Status();
}

absl::Status OpenModeSpec::Validate(ReadWriteMode read_write_mode) const {
  if (delete_existing && !create) {
    return absl::InvalidArgumentError(
        "Cannot specify an open mode of `delete_existing` "
        "without `create`");
  }

  if (delete_existing && open) {
    return absl::InvalidArgumentError(
        "Cannot specify an open mode of `delete_existing` "
        "with `open`");
  }

  if (create && (read_write_mode != ReadWriteMode::dynamic &&
                 !(read_write_mode & ReadWriteMode::write))) {
    return absl::InvalidArgumentError(
        "Cannot specify an open mode of `create` "
        "without `write`");
  }
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorstore
