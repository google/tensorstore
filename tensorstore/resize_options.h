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

#ifndef TENSORSTORE_RESIZE_OPTIONS_H_
#define TENSORSTORE_RESIZE_OPTIONS_H_

#include <iosfwd>
#include <type_traits>

#include "absl/meta/type_traits.h"
#include "tensorstore/batch.h"

namespace tensorstore {

/// Bitvector specifying options for resolving TensorStore bounds.
///
/// Currently, there is on a single optional flag, but more may be added in the
/// future.
///
/// \relates ResolveBounds
enum class ResolveBoundsMode {
  /// Requests that any resizable (implicit) bounds of the underlying
  /// TensorStore are fixed (treated as explicit).  If this flag is not
  /// specified, implicit bounds are propagated to existing implicit bounds of
  /// the input domain but do not constrain existing explicit bounds of the
  /// input domain.
  fix_resizable_bounds = 1,
};

/// \relates ResolveBoundsMode
constexpr ResolveBoundsMode fix_resizable_bounds =
    ResolveBoundsMode::fix_resizable_bounds;

/// Computes the intersection of two mode sets.
///
/// \relates ResolveBoundsMode
/// \id ResolveBoundsMode
constexpr inline ResolveBoundsMode operator&(ResolveBoundsMode a,
                                             ResolveBoundsMode b) {
  return static_cast<ResolveBoundsMode>(static_cast<int>(a) &
                                        static_cast<int>(b));
}

/// Computes the union of two mode sets.
///
/// \relates ResolveBoundsMode
/// \id ResolveBoundsMode
constexpr inline ResolveBoundsMode operator|(ResolveBoundsMode a,
                                             ResolveBoundsMode b) {
  return static_cast<ResolveBoundsMode>(static_cast<int>(a) |
                                        static_cast<int>(b));
}

/// Checks if any mode has been set.
///
/// \relates ResolveBoundsMode
/// \id ResolveBoundsMode
constexpr inline bool operator!(ResolveBoundsMode a) {
  return !static_cast<int>(a);
}

/// Prints a debugging string representation to an `std::ostream`.
///
/// \relates ResolveBoundsMode
/// \id ResolveBoundsMode
std::ostream& operator<<(std::ostream& os, ResolveBoundsMode mode);

/// Specifies options for TensorStore ResolveBounds operations.
///
/// \relates ResolveBounds
struct ResolveBoundsOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  /// Combines any number of supported options.
  template <typename... T, typename = std::enable_if_t<
                               (IsOption<absl::remove_cvref_t<T>> && ...)>>
  ResolveBoundsOptions(T&&... option) {
    (Set(std::forward<T>(option)), ...);
  }

  void Set(ResolveBoundsMode value) { this->mode = value; }

  void Set(Batch value) { this->batch = std::move(value); }

  ResolveBoundsMode mode = ResolveBoundsMode{};

  /// Optional batch.
  Batch batch{no_batch};
};

template <>
constexpr inline bool ResolveBoundsOptions::IsOption<ResolveBoundsMode> = true;

template <>
constexpr inline bool ResolveBoundsOptions::IsOption<Batch> = true;

template <>
constexpr inline bool ResolveBoundsOptions::IsOption<Batch::View> = true;

/// Bitvector specifying resize options.
///
/// \relates Resize
enum class ResizeMode {
  /// Requests that, if applicable, the resize operation affect only the
  /// metadata but not delete data chunks that are outside of the new bounds.
  resize_metadata_only = 1,

  /// Requests that the resize be permitted even if other bounds tied to the
  /// specified bounds must also be resized.  This option should be used with
  /// caution.
  resize_tied_bounds = 2,

  /// Fail if any bounds would be reduced.
  expand_only = 4,

  /// Fail if any bounds would be increased.
  shrink_only = 8,

  // TODO(jbms): If the TensorStore is chunked and the resize is not
  // chunk-aligned, by default the driver may leave extra data in boundary
  // chunks.  Consider adding an option to force these chunks to be rewritten
  // with the fill value in all out-of-bounds positions.
};

/// \relates ResizeMode
constexpr ResizeMode resize_metadata_only = ResizeMode::resize_metadata_only;
constexpr ResizeMode resize_tied_bounds = ResizeMode::resize_tied_bounds;
constexpr ResizeMode expand_only = ResizeMode::expand_only;
constexpr ResizeMode shrink_only = ResizeMode::shrink_only;

/// Computes the intersection of two mode sets.
///
/// \relates ResizeMode
/// \id ResizeMode
constexpr inline ResizeMode operator&(ResizeMode a, ResizeMode b) {
  return static_cast<ResizeMode>(static_cast<int>(a) & static_cast<int>(b));
}

/// Computes the union of two mode sets.
///
/// \relates ResizeMode
/// \id ResizeMode
constexpr inline ResizeMode operator|(ResizeMode a, ResizeMode b) {
  return static_cast<ResizeMode>(static_cast<int>(a) | static_cast<int>(b));
}

/// Checks if any mode has been set.
///
/// \relates ResizeMode
/// \id ResizeMode
constexpr inline bool operator!(ResizeMode a) { return !static_cast<int>(a); }

/// Prints a debugging string representation to an `std::ostream`.
///
/// \relates ResizeMode
/// \id ResizeMode
std::ostream& operator<<(std::ostream& os, ResizeMode mode);

/// Specifies options for resize operations.
///
/// \relates Resize
struct ResizeOptions {
  template <typename T>
  constexpr static inline bool IsOption = false;

  /// Combines any number of supported options.
  template <typename... T, typename = std::enable_if_t<
                               (IsOption<absl::remove_cvref_t<T>> && ...)>>
  ResizeOptions(T&&... option) {
    (Set(std::forward<T>(option)), ...);
  }

  void Set(ResizeMode value) { this->mode = value; }

  /// Specifies the resize mode.
  ResizeMode mode = ResizeMode{};

  // TOOD: Add delete progress callback
};

template <>
constexpr inline bool ResizeOptions::IsOption<ResizeMode> = true;

}  // namespace tensorstore

#endif  // TENSORSTORE_RESIZE_OPTIONS_H_
