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

#ifndef TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_
#define TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_

#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/json_binding/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_file_kvstore {

/// When set, the "file" kvstore ensures durability for local file writes
/// (e.g. by calling ::fsync).
///
/// In cases where durability is not required, setting this to ``false`` may
/// make write operations faster.
struct FileIoSyncResource
    : public internal::ContextResourceTraits<FileIoSyncResource> {
  constexpr static bool config_only = true;
  static constexpr char id[] = "file_io_sync";
  using Spec = bool;
  using Resource = Spec;
  static Spec Default() { return true; }
  static constexpr auto JsonBinder() {
    return internal_json_binding::DefaultBinder<>;
  }
  static Result<Resource> Create(
      Spec v, internal::ContextResourceCreationContext context) {
    return v;
  }
  static Spec GetSpec(Resource v, const internal::ContextSpecBuilder& builder) {
    return v;
  }
};

/// When set, allows choosing how the "file" kvstore uses file locking, which
/// ensures that only one process is writing to a kvstore key at a time.
struct FileIoLockingResource
    : public internal::ContextResourceTraits<FileIoLockingResource> {
  constexpr static bool config_only = true;
  static constexpr char id[] = "file_io_locking";

  enum class LockingMode : unsigned char {
    /// Use os advisory locks such as fcntl(F_SETLK) to lock files.
    os,

    /// Use lockfiles, that is, files opened with O_CREAT | O_EXCL.
    lockfile,

    /// Do not use locking.
    none,

    /// Writes are non-atomic, they do not use locking or file renaming.
    non_atomic,
  };

  struct Spec {
    LockingMode mode;
    absl::Duration acquire_timeout;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.mode, x.acquire_timeout);
    };
  };

  using Resource = Spec;
  static Spec Default() { return Spec{LockingMode::os, absl::Seconds(60)}; }
  static constexpr auto JsonBinder() {
    namespace jb = internal_json_binding;

    return jb::Object(
        jb::Member("mode", jb::Projection<&Spec::mode>(
                               jb::DefaultValue<jb::kNeverIncludeDefaults>(
                                   [](auto* obj) { *obj = Default().mode; },
                                   jb::Enum<LockingMode, std::string_view>({
                                       {LockingMode::os, "os"},
                                       {LockingMode::lockfile, "lockfile"},
                                       {LockingMode::none, "none"},
                                       {LockingMode::non_atomic, "non_atomic"},
                                   })))),
        jb::Member(
            "acquire_timeout",
            jb::Projection<&Spec::acquire_timeout>(
                jb::DefaultValue<jb::kNeverIncludeDefaults>(
                    [](auto* obj) { *obj = Default().acquire_timeout; })))
        /**/);
  }

  static Result<Resource> Create(
      Spec v, internal::ContextResourceCreationContext context) {
    return v;
  }

  static Spec GetSpec(Resource v, const internal::ContextSpecBuilder& builder) {
    return v;
  }
};

/// When set, allows choosing how the "file" kvstore uses file io.
struct FileIoModeResource
    : public internal::ContextResourceTraits<FileIoModeResource> {
  constexpr static bool config_only = true;
  static constexpr char id[] = "file_io_mode";

  enum class IoMode : unsigned char {
    /// Use default io.
    kDefault = 0,

    /// Use memmap io.
    kMemmap,

    /// Use direct io.
    kDirect,
  };

  struct Spec {
    IoMode mode;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.mode);
    };
  };

  using Resource = Spec;
  static Spec Default() { return Spec{IoMode::kDefault}; }
  static constexpr auto JsonBinder() {
    namespace jb = internal_json_binding;

    return jb::Object(jb::Member(
        "mode",
        jb::Projection<&Spec::mode>(jb::DefaultValue<jb::kNeverIncludeDefaults>(
            [](auto* obj) { *obj = Default().mode; },
            jb::Enum<IoMode, std::string_view>({
                {IoMode::kDefault, "default"},
                {IoMode::kMemmap, "memmap"},
                {IoMode::kDirect, "direct"},
            }))))
                      /**/);
  }

  static Result<Resource> Create(
      Spec v, internal::ContextResourceCreationContext context) {
    return v;
  }

  static Spec GetSpec(Resource v, const internal::ContextSpecBuilder& builder) {
    return v;
  }
};

}  // namespace internal_file_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_FILE_FILE_RESOURCE_H_
