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

#ifndef TENSORSTORE_STATIC_CAST_H_
#define TENSORSTORE_STATIC_CAST_H_

/// \file
/// Checked and unchecked casting framework.
///
/// Many types in TensorStore permit values such as ranks and data types to be
/// specified either at compile time or at run time.  Normal implicit and
/// explicit conversions may be used to convert from a compile-time constraint
/// to a run-time constraint (since this conversion is always valid), but cannot
/// be used for the reverse direction since such conversions may fail.
///
/// This file defines a `StaticCast` function, based on a `StaticCastTraits`
/// class template intended to be specialized for types compatible with the
/// `StaticCast` function, which provides a uniform interface for both checked
/// and unchecked casts.
///
/// A checked cast, invoked as `tensorstore::StaticCast<Target>(source)`,
/// returns a `Result<Target>` value which indicates an error if the cast fails.
///
/// An unchecked cast, invoked as
/// `tensorstore::StaticCast<Target, tensorstore::unchecked>(source)`, normally
/// returns a bare `Target` value.  When debug assertions are enabled, a failed
/// cast leads to the program terminating with an assertion failure.  Otherwise,
/// a failed cast leads to undefined behavior.  As a special case optimization,
/// if `Target` is equal to `remove_cvref_t<decltype(source)>`, no cast is done
/// and the `source` value is simply returned by perfect-forwarded reference.
///
/// The facilities defined here also make it easy to define other more
/// specialized cast functions, like `StaticRankCast` (defined in rank.h) and
/// `StaticDataTypeCast` (defined in data_type.h), that also support both
/// checked and unchecked casts.

#include <type_traits>

#include "absl/base/macros.h"
#include "absl/meta/type_traits.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Enum type used as a template parameter to casts to choose between an
/// unchecked cast (that returns the bare value `T`) and a checked cast that
/// returns `Result<T>`.
enum class CastChecking { checked = 0, unchecked = 1 };

/// Tag type used as the first parameter of some constructors to request an
/// unchecked conversion (e.g. from dynamic rank to a compile-time rank).
///
/// Most types in the TensorStore library follow the convention of providing
/// constructors with an initial `unchecked_t` parameter in order to perform
/// unchecked conversion.
///
/// For types `T` that follow this convention, a specialization of
/// `StaticCastTraits<T>` is defined that inherits from
/// `DefaultStaticCastTraits<T>` in order to make the `StaticCast` function use
/// such constructors.
///
/// For builtin types and other existing types that cannot be modified, this
/// convention can't be used and instead the `StaticCastTraits` specialization
/// must define a custom `Construct` method.
struct unchecked_t {
  explicit constexpr unchecked_t() = default;
  constexpr operator CastChecking() const { return CastChecking::unchecked; }
};

/// Tag value to indicate that an unchecked cast is requested.
///
/// This tag value serves a dual purpose:
///
/// - The primary purpose is as the second template argument to `StaticCast`
///   (and other cast functions that follow the same convention) in order to
///   indicate an unchecked conversion.  This relies on the implicit conversion
///   to `CastChecking::unchecked`.
///
/// - It can also be used directly to call constructors with an `unchecked_t`
///   parameter.
constexpr unchecked_t unchecked{};

/// Traits class that is specialized for unqualified source and target types `T`
/// compatible with `StaticCast`.
///
/// Specializations of `StaticCastTraits<T>` must define the members documented
/// by `DefaultStaticCastTraits`, and may publicly inherit from
/// `DefaultStaticCastTraits<T>` to use the default implementation of
/// `Construct`.
template <typename T>
struct StaticCastTraits;

/// Base class from which specializations of `StaticCastTraits` may inherit in
/// order to obtain the default `Construct` behavior.
template <typename T>
struct DefaultStaticCastTraits {
  /// Constructs a value of type `T` from `source`.
  ///
  /// The return type must be `T`.
  ///
  /// Should be disabled via SFINAE when the cast is not supported.
  ///
  /// The `DefaultStaticCastTraits` implementation returns
  /// `T(unchecked, std::forward<SourceRef>(source))`.
  template <typename SourceRef>
  static absl::enable_if_t<
      std::is_constructible<T, unchecked_t, SourceRef>::value, T>
  Construct(SourceRef&& source) {
    return T(unchecked, std::forward<SourceRef>(source));
  }

  /// Returns `true` if a value of type `T` can be constructed from `source`.
  ///
  /// This should check whether any runtime-specified values of `source` (such
  /// as rank or data type) are compatible with the compile-time constraints
  /// imposed by the target type `T`.
  ///
  /// The `StaticCast` function will only ever call this function for source
  /// types `SourceRef&&` accepted by `Construct`.
  template <typename SourceRef>
  static bool IsCompatible(SourceRef&& source) = delete;

  /// Returns a type convertible to `absl::string_view` that describes the type
  /// `T` and any compile-time constraints it imposes for use in cast error
  /// messages.
  static std::string Describe() = delete;

  /// Returns a type convertible to `absl::string_view` that describes the
  /// compile-time and run-time constraints specified by `value` for use in cast
  /// error messages.
  static std::string Describe(const T& value) = delete;
};

/// Alias that evaluates to the `StaticCastTraits` type to use for an
/// optionally-qualified type `T`.
template <typename T>
using CastTraitsType = StaticCastTraits<internal::remove_cvref_t<T>>;

namespace internal_cast {

/// Implementation of actual cast operation.
///
/// Specializations define a nested `ResultType<SourceRef, Target>` alias and a
/// static `StaticCast` member function that performs the cast.
///
/// \tparam Checking Specifies whether the cast is checked or unchecked.
/// \tparam IsNoOp Specifies whether `Checking == unchecked` and the `Target`
///     type is equal to `remove_cvref_t<SourceRef>`.
template <CastChecking Checking, bool IsNoOp>
struct CastImpl;

/// Returns an `absl::StatusCode::kInvalidArgument` error that includes the
/// source and target descriptions.
Status CastError(absl::string_view source_description,
                 absl::string_view target_description);

/// Specialization for `Checking == checked`.
template <>
struct CastImpl<CastChecking::checked, false> {
  template <typename SourceRef, typename Target>
  using ResultType = Result<Target>;

  template <typename Target, typename SourceRef>
  static Result<Target> StaticCast(SourceRef&& source) {
    if (!StaticCastTraits<Target>::IsCompatible(source)) {
      return CastError(CastTraitsType<SourceRef>::Describe(source),
                       StaticCastTraits<Target>::Describe());
    }
    return StaticCastTraits<Target>::Construct(std::forward<SourceRef>(source));
  }
};

/// Specialization for `Checking == unchecked` and `IsNoOp == false`.
template <>
struct CastImpl<CastChecking::unchecked, false> {
  template <typename SourceRef, typename Target>
  using ResultType = Target;

  template <typename Target, typename SourceRef>
  static Target StaticCast(SourceRef&& source) {
    ABSL_ASSERT(StaticCastTraits<Target>::IsCompatible(source) &&
                "StaticCast is not valid");
    return StaticCastTraits<Target>::Construct(std::forward<SourceRef>(source));
  }
};

/// Specialization for `Checking == unchecked` and `IsNoOp == true`.
///
/// This just returns a perfect-forwarded reference to the source value without
/// using `StaticCastTraits` at all.
template <>
struct CastImpl<CastChecking::unchecked, true> {
  template <typename SourceRef, typename Target>
  using ResultType = SourceRef&&;

  template <typename Target, typename SourceRef>
  static SourceRef&& StaticCast(SourceRef&& source) {
    return std::forward<SourceRef>(source);
  }
};

template <typename Target, typename SourceRef, CastChecking Checking,
          bool IsSame =
              std::is_same<Target, internal::remove_cvref_t<SourceRef>>::value>
using CastImplType =
    internal_cast::CastImpl<Checking,
                            (Checking == CastChecking::unchecked && IsSame)>;

template <typename Target, typename SourceRef, typename ReturnType = Target>
struct IsCastConstructible : public std::false_type {};

template <typename Target, typename SourceRef>
struct IsCastConstructible<Target, SourceRef,
                           decltype(StaticCastTraits<Target>::Construct(
                               std::declval<SourceRef>()))>
    : public std::true_type {};

}  // namespace internal_cast

/// `bool`-valued metafunction that evaluates to `true` if a value of type
/// `SourceRef&&` can be converted to a type of `Target` using `StaticCast`.
template <typename Target, typename SourceRef>
using IsCastConstructible =
    internal_cast::IsCastConstructible<Target, SourceRef>;

/// Evaluates to the result of casting a value of type `SourceRef&&` to `Target`
/// with a checking mode of `Checking`.
///
/// \requires `IsCastConstructible<Target, SourceRef>`.
template <typename Target, typename SourceRef,
          CastChecking Checking = CastChecking::unchecked>
using SupportedCastResultType = absl::enable_if_t<
    IsCastConstructible<Target, SourceRef>::value,
    typename internal_cast::CastImplType<
        Target, SourceRef, Checking>::template ResultType<SourceRef, Target>>;

/// Casts `source` to the specified `Target` type using `StaticCastTraits`.
///
/// `StaticCastTraits` must be specialized for both `Target` and
/// `remove_cvref_t<SourceRef>`.  Supported types include `DataType`,
/// `StaticRank`, `Box`, `Array`, `StridedLayout`, `IndexTransform`, `Spec`,
/// `TensorStore`.
///
/// Example:
///
///     tensorstore::Box<> b = ...;
///     BoxView<5> b_view = tensorstore::StaticCast<tensorstore::BoxView<5>,
///                                           tensorstore::unchecked>(b);
///     Result<BoxView<5>> b_view_result =
///         tensorstore::StaticCast<tensorstore::BoxView<5>>(b);
///
/// \tparam Target Target type.  Typically, both `SourceRef` and `Target` are
///     instances of the same class template, but differ in one or more template
///     arguments.  For example, the source type may be
///     `ArrayView<void, dynamic_rank>` with a target type of
///     `ArrayView<int, 3>`.
/// \tparam Checking Specifies whether to validate the cast at run time.  If
///     equal to `CastChecking::checked` (the default), the return type is
///     `Result<Target>` and an error `Status` is returned if the cast is not
///     valid.  If equal to `CastChecking::unchecked`, the return type is a bare
///     `Target` cast is validated in debug mode only via an `assert`.  A value
///     of `CastChecking::unchecked` may be more conveniently specified using
///     the special tag value `tensorstore::unchecked`.
/// \param source Source value.
/// \requires `IsCastConstructible<Target, SourceRef>`
template <typename Target, CastChecking Checking = CastChecking::checked,
          typename SourceRef>
SupportedCastResultType<Target, SourceRef, Checking> StaticCast(
    SourceRef&& source) {
  return internal_cast::CastImplType<Target, SourceRef, Checking>::
      template StaticCast<Target>(std::forward<SourceRef>(source));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_STATIC_CAST_H_
