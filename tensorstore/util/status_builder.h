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
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/types/source_location.h"

namespace tensorstore {

/// `StatusBuilder` is a builder object for constructing `absl::Status` objects,
/// with methods to override the status code, augment the error message, and
/// attach payloads.
///
/// A `StatusBuilder` is implicitly convertible to `absl::Status`, and the
/// `absl::Status` can be explicitly constructed using the method
/// `StatusBuilder::BuildStatus()`.
///
/// Example::
///
///   absl::Status foo(int i) {
///     if (i < 0) {
///       return StatusBuilder(absl::StatusCode::kInvalidArgument)
///                 .Format("i=%d", i);
///     }
///     return absl::OkStatus();
///   }
///
/// `StatusBuilder` is primarily used in the `TENSORSTORE_RETURN_IF_ERROR` and
/// `TENSORSTORE_ASSIGN_OR_RETURN` macros.
///
/// \ingroup error handling
class ABSL_MUST_USE_RESULT StatusBuilder {
 public:
  /// Creates a `StatusBuilder` from an existing status. If the `status` is not
  /// `absl::StatusCode::kOk`, the source location may be added to it.
  ///
  /// \id status
  explicit StatusBuilder(
      absl::Status status,
      absl::SourceLocation loc = absl::SourceLocation::current())
      : status_(std::move(status)), loc_(loc) {
    if (!status_.ok()) {
      MaybeAddSourceLocation(status_, loc_);
    }
  }

  /// Creates a `StatusBuilder` with the given status `code` and an empty
  /// message.
  ///
  /// \id statuscode
  explicit StatusBuilder(
      absl::StatusCode code,
      absl::SourceLocation loc = absl::SourceLocation::current())
      : status_(code, "", loc), loc_(loc) {}

  StatusBuilder(StatusBuilder&& other) = default;
  StatusBuilder& operator=(StatusBuilder&& other) = default;
  StatusBuilder(const StatusBuilder& other) = default;
  StatusBuilder& operator=(const StatusBuilder& other) = default;

  /// Returns whether the current status code is `absl::StatusCode::kOk`.
  bool ok() const { return status_.ok(); }

  /// Returns the status code of the underlying status, or the overridden code
  /// if one has been set.
  absl::StatusCode code() const {
    return has_rep() ? rep_.code : status_.code();
  }

  /// Overrides the absl::StatusCode on the error. `code` must not be
  /// `absl::StatusCode::kOk`.
  StatusBuilder& SetCode(absl::StatusCode code) &;
  StatusBuilder&& SetCode(absl::StatusCode code) &&;

  /// Adds a formatted message to the status.
  ///
  /// The formatted message is prepended (by default) or appended (if
  /// `SetAppend()` has been called) to the existing status message, using ": "
  /// as a separator.
  template <typename... Args>
  StatusBuilder& Format(const absl::FormatSpec<Args...>& format,
                        const Args&... args) {
    if (status_.ok()) return *this;
    EnsureRep();
    absl::StrAppendFormat(&rep_.message, format, args...);
    return *this;
  }

  /// Mutates the builder so that any formatted message is prepended to the
  /// status.
  ///
  /// NOTE: Multiple calls to `SetPrepend` and `SetAppend` just adjust the
  /// behavior of the final join of the original status with the extra message.
  StatusBuilder& SetPrepend() {
    if (status_.ok()) return *this;
    EnsureRep();
    rep_.append = false;
    return *this;
  }

  /// Mutates the builder so that any formatted message is appended to the
  /// status.
  ///
  /// NOTE: Multiple calls to `SetPrepend` and `SetAppend` just adjust the
  /// behavior of the final join of the original status with the extra message.
  StatusBuilder& SetAppend() {
    if (status_.ok()) return *this;
    EnsureRep();
    rep_.append = true;
    return *this;
  }

  /// Returns payload for the given type URL, or nullopt if not found.
  std::optional<absl::Cord> GetPayload(std::string_view type_url) const {
    return status_.GetPayload(type_url);
  }

  /// Sets a payload on the status equivalent to `absl::Status::SetPayload`.
  StatusBuilder& SetPayload(std::string_view type_url, absl::Cord payload) {
    if (!status_.ok()) status_.SetPayload(type_url, std::move(payload));
    return *this;
  }
  StatusBuilder& SetPayload(std::string_view type_url,
                            std::string_view payload) {
    if (!status_.ok()) status_.SetPayload(type_url, absl::Cord(payload));
    return *this;
  }
  template <typename T,
            std::enable_if_t<std::is_same_v<T, std::string>, int> = 0>
  StatusBuilder& SetPayload(std::string_view type_url, T&& payload) {
    if (!status_.ok())
      status_.SetPayload(type_url, absl::Cord(std::forward<T>(payload)));
    return *this;
  }

  /// Adds a payload value to status.
  ///
  /// If a payload with the same type URL already exists, then a new prefix will
  /// be generated of the form "type_url[N]".
  StatusBuilder& AddStatusPayload(std::string_view type_url,
                                  absl::Cord payload);

  /// Applies a custom `adaptor` functor to the StatusBuilder.
  ///
  /// The `adaptor` functor can be any callable that takes a `StatusBuilder` or
  /// an `absl::Status` as an argument. The primary use cases are applying
  /// policies, type conversions, and/or side effects on the StatusBuilder, or
  /// converting the returned value of the `TENSORSTORE_RETURN_IF_ERROR` macro
  /// to another type.
  ///
  /// Example::
  ///
  ///  return StatusBuilder(absl::StatusCode::kInvalidArgument)
  ///       .Format("i=%d", i)
  ///       .With([](absl::Status status) {
  ///         LOG(ERROR) << "Error in FunctionWithLogging: " << status;
  ///         return status;
  ///       });
  ///
  /// \returns The value returned by `adaptor`, which may be any type or `void`.
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

  /// Constructs an `absl::Status` from the current state.
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

  /// Converts to `absl::Status`.
  operator absl::Status() const& { return BuildStatus(); }
  operator absl::Status() && { return BuildStatus(); }

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

  static void MaybeAddSourceLocation(absl::Status& status,
                                     absl::SourceLocation loc);

  // Additional state needed on some error cases.
  struct Rep {
    std::string message;
    absl::StatusCode code = absl::StatusCode::kOk;
    bool append = false;  // Append mode.
  };

  // The status that the result will be based on.
  absl::Status status_;
  absl::SourceLocation loc_;
  Rep rep_{};
};

inline StatusBuilder& StatusBuilder::SetCode(absl::StatusCode code) & {
  assert(code != absl::StatusCode::kOk);
  if (status_.ok()) {
    status_ = absl::Status(code, "", loc_);
  } else {
    rep_.code = code;
  }
  return *this;
}

inline StatusBuilder&& StatusBuilder::SetCode(absl::StatusCode code) && {
  assert(code != absl::StatusCode::kOk);
  if (status_.ok()) {
    status_ = absl::Status(code, "", loc_);
  } else {
    rep_.code = code;
  }
  return std::move(*this);
}

/// Returns the `absl::Status` for the StatusBuilder.
///
/// \returns `status_builder.BuildStatus()`
/// \relates StatusBuilder
/// \id statusbuilder
inline absl::Status GetStatus(StatusBuilder&& status_builder) {
  return std::move(status_builder).BuildStatus();
}
inline absl::Status GetStatus(const StatusBuilder& status_builder) {
  return status_builder.BuildStatus();
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STATUS_BUILDER_H_
