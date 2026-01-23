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

#ifndef TENSORSTORE_UTIL_STATUS_BUILDER_H_
#define TENSORSTORE_UTIL_STATUS_BUILDER_H_

#include <stddef.h>

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status_impl.h"

// Forward declaration of Result<T> to avoid including result.h directly.
namespace tensorstore {
template <typename T>
class Result;
template <typename T>
class Future;
}  // namespace tensorstore

namespace tensorstore {
namespace internal {

// StatusBuilder constructs absl::Status objects using a builder pattern,
// allowing overriding the status code, formatted messages, and adding payloads.
//
// Example:
//   absl::Status foo(int i) {
//     if (i < 0) {
//       return StatusBuilder(absl::StatusCode::kInvalidArgument)
//                 .Format("i=%d", i);
//     }
//     return absl::OkStatus();
//   }
class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  // Creates a StatusBuilder with the given status code.
  StatusBuilder(absl::StatusCode code,
                SourceLocation loc = ::tensorstore::SourceLocation::current())
      : status_(internal_status::StatusWithSourceLocation(code, "", loc)),
        loc_(loc) {}

  // Creates a StatusBuilder from an existing status. If the status is not OK,
  // the source location may be added to it.
  explicit StatusBuilder(
      absl::Status status,
      SourceLocation loc = ::tensorstore::SourceLocation::current())
      : status_(std::move(status)), loc_(loc) {
    if (!status_.ok()) {
      internal_status::MaybeAddSourceLocationImpl(status_, loc_);
    }
  }

  StatusBuilder(StatusBuilder&& other) = default;
  StatusBuilder& operator=(StatusBuilder&& other) = default;
  StatusBuilder(const StatusBuilder& other) = default;
  StatusBuilder& operator=(const StatusBuilder& other) = default;

  bool ok() const { return status_.ok(); }
  absl::StatusCode code() const {
    return has_rep() ? rep_.code : status_.code();
  }

  // Returns payload for the given type URL, or nullopt if not found.
  std::optional<absl::Cord> GetPayload(std::string_view type_url) const {
    return status_.GetPayload(type_url);
  }

  // Builds a message that will be added to the status.
  // If an existing message exists, the final message will be constructed
  // by prepending (the default) or appending, depending on the mode of
  // SetPrepend() / SetAppend(), to the existing message along with a
  // separator (': ').
  template <typename... Args>
  StatusBuilder& Format(const absl::FormatSpec<Args...>& format,
                        const Args&... args) {
    if (status_.ok()) return *this;
    EnsureRep();
    absl::StrAppendFormat(&rep_.message, format, args...);
    return *this;
  }

  // Mutates the builder so that the final additional message is prepended to
  // the original error message in the status.
  //
  // NOTE: Multiple calls to `SetPrepend` and `SetAppend` just adjust the
  // behavior of the final join of the original status with the extra message.
  //
  // Returns `*this` to allow method chaining.
  StatusBuilder& SetPrepend() {
    if (status_.ok()) return *this;
    EnsureRep();
    rep_.append = false;
    return *this;
  }

  // Mutates the builder so that the final additional message is appended to the
  // original error message in the status.
  //
  // NOTE: Multiple calls to `SetPrepend` and `SetAppend` just adjust the
  // behavior of the final join of the original status with the extra message.
  //
  // Returns `*this` to allow method chaining.
  StatusBuilder& SetAppend() {
    if (status_.ok()) return *this;
    EnsureRep();
    rep_.append = true;
    return *this;
  }

  // Sets a payload on the status using `status.SetPayload`.
  StatusBuilder& SetPayload(std::string_view type_url, absl::Cord payload) {
    if (!status_.ok()) status_.SetPayload(type_url, std::move(payload));
    return *this;
  }
  StatusBuilder& SetPayload(std::string_view type_url,
                            std::string_view payload) {
    if (!status_.ok()) status_.SetPayload(type_url, absl::Cord(payload));
    return *this;
  }

  // Adds value to status using `status.SetPayload`.
  // Iterates through payloads like `prefix` and `prefix[N]`, and if value
  // is not found, adds to the status payload.
  StatusBuilder& AddStatusPayload(std::string_view type_url,
                                  absl::Cord payload);

  // Overrides the absl::StatusCode on the error. `code` must not be
  // `absl::StatusCode::kOk`.
  StatusBuilder& SetCode(absl::StatusCode code) &;
  StatusBuilder&& SetCode(absl::StatusCode code) &&;

  // Construct the absl::Status from the current state of the StatusBuilder.
  ABSL_MUST_USE_RESULT absl::Status BuildStatus() const& {
    if (do_build_status()) {
      return BuildStatusImpl();
    }
    return status_;
  }
  ABSL_MUST_USE_RESULT absl::Status BuildStatus() && {
    if (do_build_status()) {
      return BuildStatusImpl();
    }
    return std::move(status_);
  }

  operator absl::Status() const& { return BuildStatus(); }
  operator absl::Status() && { return BuildStatus(); }

  template <typename T>
  operator tensorstore::Result<T>() const& {
    return BuildStatus();
  }
  template <typename T>
  operator tensorstore::Result<T>() && {
    return BuildStatus();
  }
  template <typename T>
  operator tensorstore::Future<T>() const& {
    return BuildStatus();
  }
  template <typename T>
  operator tensorstore::Future<T>() && {
    return BuildStatus();
  }

  // Calls `adaptor` on this status builder to apply policies, type conversions,
  // and/or side effects on the StatusBuilder. Returns the value returned by
  // `adaptor`, which may be any type including `void`.
  template <typename Adaptor>
  auto With(
      Adaptor&& adaptor) & -> decltype(std::forward<Adaptor>(adaptor)(*this)) {
    return std::forward<Adaptor>(adaptor)(*this);
  }
  template <typename Adaptor>
  auto With(Adaptor&& adaptor) && -> decltype(std::forward<Adaptor>(adaptor)(
      std::move(*this))) {
    return std::forward<Adaptor>(adaptor)(std::move(*this));
  }

 private:
  // Construct rep_ and initialize it properly.
  void EnsureRep() {
    if (rep_.code == absl::StatusCode::kOk) {
      rep_.code = status_.code();
    }
  }

  bool has_rep() const { return rep_.code != absl::StatusCode::kOk; }

  bool do_build_status() const {
    return rep_.code != absl::StatusCode::kOk &&
           (rep_.code != status_.code() || !rep_.message.empty());
  }

  absl::Status BuildStatusImpl() const;

  // Additional state needed on some error cases.
  struct Rep {
    std::string message;
    absl::StatusCode code = absl::StatusCode::kOk;
    bool append = false;  // Append mode.
  };

  // The status that the result will be based on.
  absl::Status status_;
  tensorstore::SourceLocation loc_;
  Rep rep_{};
};

inline StatusBuilder& StatusBuilder::SetCode(absl::StatusCode code) & {
  assert(code != absl::StatusCode::kOk);
  if (status_.ok()) {
    status_ = internal_status::StatusWithSourceLocation(code, "", loc_);
  } else {
    rep_.code = code;
  }
  return *this;
}

inline StatusBuilder&& StatusBuilder::SetCode(absl::StatusCode code) && {
  assert(code != absl::StatusCode::kOk);
  if (status_.ok()) {
    status_ = internal_status::StatusWithSourceLocation(code, "", loc_);
  } else {
    rep_.code = code;
  }
  return std::move(*this);
}

}  // namespace internal

// Returns the `absl::Status` for the StatusBuilder.
//
// \returns `status_builder.BuildStatus()`
inline absl::Status GetStatus(internal::StatusBuilder&& status_builder) {
  return std::move(status_builder).BuildStatus();
}
inline absl::Status GetStatus(const internal::StatusBuilder& status_builder) {
  return status_builder.BuildStatus();
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STATUS_BUILDER_H_
