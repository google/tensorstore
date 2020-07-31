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

#ifndef TENSORSTORE_INDEX_SPACE_TRANSFORMED_ARRAY_H_
#define TENSORSTORE_INDEX_SPACE_TRANSFORMED_ARRAY_H_

/// \file
/// A transformed array is a view through an index transform of a strided
/// in-memory array.

#include <memory>
#include <type_traits>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transformed_array_impl.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

template <typename ElementTagType, DimensionIndex Rank = dynamic_rank,
          ContainerKind LayoutCKind = container>
class TransformedArray;

template <typename ElementTagType, DimensionIndex Rank = dynamic_rank,
          ContainerKind LayoutCKind = container>
class NormalizedTransformedArray;

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `TransformedArray`.
template <typename T>
struct IsTransformedArray : public std::false_type {};

template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct IsTransformedArray<TransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `NormalizedTransformedArray`.
template <typename T>
struct IsNormalizedTransformedArray : public std::false_type {};

template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct IsNormalizedTransformedArray<
    NormalizedTransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

/// Bool-valued metafunction that evaluates to `true` if `T` satisfies
/// `IsArray`, `IsTransformedArray`, or `IsNormalizedTransformedArray`.
template <typename T>
struct IsTransformedArrayLike : public IsArray<T> {};

template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct IsTransformedArrayLike<
    TransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct IsTransformedArrayLike<
    NormalizedTransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

/// View through an index transform of an in-memory array.
///
/// Example of making a transformed array directly:
///
///    // Transform that extracts the diagonal.
///    IndexTransform<> t = IndexTransformBuilder<>(1, 2)
///                             .output_single_input_dimension(0, 0)
///                             .output_single_input_dimension(1, 0)
///                             .Finalize()
///                             .value();
///    auto source_array = MakeArray<int>({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
///    auto dest_array = AllocateArray<int>({3});
///    auto transformed_array = TransformedArray(source_array, t);
///    // Copy the diagonal of source_array to dest_array.
///    IterateOverTransformedArrays([](const int* x, int* y) { *y = *x; },
///                                 /*constraints=*/{}, transformed_array,
///                                 dest_array);
///    // dest_array equals {1, 5, 9}.
///
/// Example of making a transformed array using DimExpression:
///
///     auto transformed_array = ChainResult(
///       tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}}),
///       tensorstore::Dims(0).TranslateTo(10),
///       tensorstore::Dims(0, 1).IndexVectorArraySlice(
///         tensorstore::MakeArray<Index>({10, 1}, {11, 1}, {11, 2}))
///       tensorstore::Dims(0).Label("a")).value();
///
///
/// A transformed array uses one of three possible representations (determined
/// at run time):
///
///   1. An element pointer `element_pointer()` accessed directly using an index
///      transform `transform()` (the byte offset of each element is obtained by
///      summing the output indices computed by the index transform).  This is
///      the "normalized" representation (and is the only representation
///      supported by the simpler and more efficient
///      `NormalizedTransformedArray` class template).
///
///   2. A strided "base" array `base_array()` (of arbitrary rank, using either
///      a zero-origin or offset-origin layout) transformed by an index
///      transform `transform()`.  This representation is not considered
///      "normalized", and it is possible that the range of the `transform()`
///      may not be contained within the domain of the `base_array()`.  Such
///      incompatibilities lead to errors being returned (never undefined
///      behavior) when the transformed array is used.
///
///   3. A strided "untransformed" array `untransformed_array()` without an
///      index transform (using either a zero-origin or offset-origin layout).
///      This representation is also not considered "normalized", although
///      unlike representation 2 this representation is always valid.
///
/// \tparam ElementTagType Must satisfy `IsElementTag`.  Either `T` or
///     `Shared<T>`, where `T` satisfies `IsElementType<T>`.
/// \tparam Rank The static rank of the transformed array.  May be
///     `dynamic_rank` to allow the rank to be determined at run time.
/// \tparam LayoutCKind Either `container` or `view`.  If equal to `container`,
///     the transformed array owns the contained index transform and/or strided
///     layout.  If equal to `view`, an unowned reference to an index transform
///     and/or strided layout is stored.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
class TransformedArray {
  using Access = internal_index_space::TransformedArrayAccess;
  using LayoutStorage = Access::LayoutStorage<Rank, LayoutCKind>;
  static_assert(IsElementTag<ElementTagType>::value,
                "ElementTagType must be an ElementTag type.");
  static_assert(Rank == dynamic_rank || Rank >= 0,
                "Rank must be dynamic_rank or >= 0.");

 public:
  using ElementTag = ElementTagType;
  using ElementPointer = tensorstore::ElementPointer<ElementTag>;
  using Pointer = typename ElementPointer::Pointer;
  using Transform = typename LayoutStorage::Transform;
  using Element = typename ElementPointer::Element;
  using DataType = StaticOrDynamicDataTypeOf<Element>;
  constexpr static DimensionIndex static_rank = Transform::static_input_rank;
  constexpr static ContainerKind layout_container_kind = LayoutCKind;
  using RankType = StaticOrDynamicRank<static_rank>;

  template <ArrayOriginKind OriginKind>
  using UntransformedArray =
      Array<ElementTagType, Rank, OriginKind, LayoutCKind>;
  template <ArrayOriginKind OriginKind>
  using BaseArray =
      Array<ElementTagType, dynamic_rank, OriginKind, LayoutCKind>;

  template <ContainerKind CKind>
  using RebindContainerKind = TransformedArray<ElementTagType, Rank, CKind>;

  /// Alias that evaluates to the `NormalizedTransformedArray` type with the
  /// same `ElementTag` but with the `static_input_rank` of `OtherTransform`,
  /// and a layout container kind of `container`.
  ///
  /// \tparam OtherTransform The new transform type.  Must be an instance of
  ///     `IndexTransform` or `IndexTransformView`.
  template <typename OtherTransform = Transform>
  using RebindTransform =
      NormalizedTransformedArray<ElementTagType,
                                 OtherTransform::static_input_rank, container>;

  /// Constructs an invalid transformed array with a rank of `RankType()`.
  TransformedArray() = default;

  /// Constructs a transformed array that holds the specified "untransformed"
  /// array.
  ///
  /// The domain of the index transform is equal to the domain of the array.  An
  /// index vector `v` into the transformed array corresponds to `array(v)`.
  ///
  /// \requires `A` satisfies `IsArray`
  /// \requires `UntransformedArray<A::array_origin_kind>` is constructible from
  /// `A`. \remarks This constructor is explicit if, and only if, the conversion
  /// from
  ///     `A` to `UntransformedArray<A::array_origin_kind>` is explicit.
  template <
      typename A,
      std::enable_if_t<(IsArray<internal::remove_cvref_t<A>>::value &&
                        std::is_convertible<
                            A, UntransformedArray<internal::remove_cvref_t<
                                   A>::array_origin_kind>>::value)>* = nullptr>
  TransformedArray(A&& array) noexcept
      : TransformedArray(Access::construct_array_tag{},
                         std::forward<A>(array)) {}

  /// Overload that handles the explicit conversion case.
  template <
      typename A,
      std::enable_if_t<(IsArray<internal::remove_cvref_t<A>>::value &&
                        std::is_constructible<UntransformedArray<offset_origin>,
                                              A&&>::value &&
                        !std::is_convertible<
                            A, UntransformedArray<internal::remove_cvref_t<
                                   A>::array_origin_kind>>::value)>* = nullptr>
  explicit TransformedArray(A&& array) noexcept
      : TransformedArray(Access::construct_array_tag{},
                         std::forward<A>(array)) {}

  /// Unchecked conversion from an existing `Array`.
  ///
  /// \requires `A` is an instance of `Array` with a `StaticCast`-compatible
  ///     `ElementPointer` and `static_rank`.
  /// \pre `array.data_type()` is compatible with `Element`.
  /// \pre `array.rank()` is compatible with `Rank`.
  template <
      typename A,
      std::enable_if_t<(IsArray<internal::remove_cvref_t<A>>::value &&
                        IsCastConstructible<UntransformedArray<offset_origin>,
                                            A&&>::value)>* = nullptr>
  explicit TransformedArray(unchecked_t, A&& array) noexcept
      : TransformedArray(Access::construct_array_tag{},
                         std::forward<A>(array)) {}

  /// Constructs a transformed array from an element pointer and an index
  /// transform.
  ///
  /// The domain of the transformed array is equal to the input domain of the
  /// index transform.  An index vector `v` into the transformed array
  /// corresponds to the element at a byte offset of `sum(transform(v))` from
  /// `element_pointer`.
  ///
  /// \requires `ElementPointer` is constructible from `P`.
  /// \requires `Transform` is constructible from `T`.
  template <typename P, typename T,
            std::enable_if_t<internal::IsPairImplicitlyConvertible<
                P, T, ElementPointer, Transform>::value>* = nullptr>
  TransformedArray(P&& element_pointer, T&& transform) noexcept
      : TransformedArray(Access::construct_element_pointer_tag{},
                         std::forward<P>(element_pointer),
                         std::forward<T>(transform)) {}

  /// Constructs a transformed array from a "base" array and an index
  /// transform.
  ///
  /// The domain of the transformed array is equal to the domain of the index
  /// transform.  An index vector `v` into the transformed array corresponds to
  /// `array(transform(v))`, but all accesses are checked to ensure that
  /// `transform(v)` is within the domain of `array`.
  ///
  /// \requires `A` satisfies `IsArray`.
  /// \requires `T` satisfies `IsIndexTransform`.
  /// \requires `A::static_rank == T::static_output_rank`.
  /// \requires `BaseArray<A::array_origin_kind>` is constructible from `A`.
  /// \requires `Transform` is constructible from `T`.
  /// \remarks This constructor is explicit if, and only if, the conversion from
  ///     `A` to `BaseArray<A::array_origin_kind>` and/or the conversion from
  ///     `T` to `Transform` is explicit.
  template <
      typename A, typename T,
      std::enable_if_t<
          (IsArray<internal::remove_cvref_t<A>>::value &&
           IsIndexTransform<internal::remove_cvref_t<T>>::value &&
           (internal::remove_cvref_t<T>::static_output_rank ==
            internal::remove_cvref_t<A>::static_rank) &&
           internal::IsPairImplicitlyConvertible<
               A, T, BaseArray<internal::remove_cvref_t<A>::array_origin_kind>,
               Transform>::value)>* = nullptr>
  TransformedArray(A&& array, T&& transform)
      : TransformedArray(Access::construct_base_array_transform_tag{},
                         std::forward<A>(array), std::forward<T>(transform)) {}

  /// Overload that handles the explicit conversion case.
  template <
      typename A, typename T,
      std::enable_if_t<
          (IsArray<internal::remove_cvref_t<A>>::value &&
           IsIndexTransform<internal::remove_cvref_t<T>>::value &&
           (internal::remove_cvref_t<T>::static_output_rank ==
            internal::remove_cvref_t<A>::static_rank) &&
           internal::IsPairOnlyExplicitlyConvertible<
               A, T, BaseArray<internal::remove_cvref_t<A>::array_origin_kind>,
               Transform>::value)>* = nullptr>
  explicit TransformedArray(A&& array, T&& transform)
      : TransformedArray(Access::construct_base_array_transform_tag{},
                         std::forward<A>(array), std::forward<T>(transform)) {}

  /// Copy or move constructs from another transformed array.
  ///
  /// \requires `ElementPointer` is constructible from `Other::ElementPointer`.
  /// \requires `Transform` is constructible from `Other::Transform`.
  /// \remarks This constructor is explicit if, and only if, the conversion from
  ///     `Other::ElementPointer` to `ElementPointer` and/or the conversion from
  ///     `Other::UntransformedArray<...>` to `UntransformedArray<...>` is
  ///     explicit.
  template <
      typename Other,
      std::enable_if_t<
          (IsTransformedArray<internal::remove_cvref_t<Other>>::value &&
           internal::IsPairImplicitlyConvertible<
               typename internal::remove_cvref_t<Other>::ElementPointer,
               typename internal::remove_cvref_t<
                   Other>::template UntransformedArray<offset_origin>,
               ElementPointer, UntransformedArray<offset_origin>>::value)>* =
          nullptr>
  TransformedArray(Other&& other) noexcept
      : TransformedArray(Access::construct_tag{}, std::forward<Other>(other)) {}

  /// Overload that handles the explicit conversion case.
  template <
      typename Other,
      std::enable_if_t<
          (IsTransformedArray<internal::remove_cvref_t<Other>>::value &&
           internal::IsPairOnlyExplicitlyConvertible<
               typename internal::remove_cvref_t<Other>::ElementPointer,
               typename internal::remove_cvref_t<
                   Other>::template UntransformedArray<offset_origin>,
               ElementPointer, UntransformedArray<offset_origin>>::value)>* =
          nullptr>
  explicit TransformedArray(Other&& other) noexcept
      : TransformedArray(Access::construct_tag{}, std::forward<Other>(other)) {}

  /// Copy or move constructs from another normalized transformed array.
  ///
  /// \requires `ElementPointer` is constructible from `Other::ElementPointer`.
  /// \requires `Transform` is constructible from `Other::Transform`.
  template <
      typename Other,
      std::enable_if_t<
          (IsNormalizedTransformedArray<
               internal::remove_cvref_t<Other>>::value &&
           std::is_convertible<
               typename internal::remove_cvref_t<Other>::ElementPointer,
               ElementPointer>::value &&
           IsRankImplicitlyConvertible(
               internal::remove_cvref_t<Other>::static_rank, Rank))>* = nullptr>
  TransformedArray(Other&& other) noexcept
      : TransformedArray(Access::construct_tag{}, std::forward<Other>(other)) {}

  /// Unchecked conversion from an existing `TransformedArray` or
  /// `NormalizedTransformedArray`.
  ///
  /// \requires `Other` is an instance of `TransformedArray` or
  ///     `NormalizedTransformedArray` with `StaticCast`-compatible
  ///     `ElementPointer` and `static_rank`.
  /// \pre `other.data_type()` is compatible with `Element`.
  /// \pre `other.rank()` is compatible with `Rank`.
  template <
      typename Other,
      std::enable_if_t<
          ((IsTransformedArray<internal::remove_cvref_t<Other>>::value ||
            IsNormalizedTransformedArray<
                internal::remove_cvref_t<Other>>::value) &&
           IsCastConstructible<ElementPointer,
                               typename internal::remove_cvref_t<
                                   Other>::ElementPointer>::value &&
           IsRankExplicitlyConvertible(
               internal::remove_cvref_t<Other>::static_rank, Rank))>* = nullptr>
  explicit TransformedArray(unchecked_t, Other&& other) noexcept
      : TransformedArray(Access::construct_tag{}, std::forward<Other>(other)) {}

  /// Copy or move assigns from another transformed array or array.
  template <typename Other, std::enable_if_t<std::is_constructible<
                                TransformedArray, Other&&>::value>* = nullptr>
  TransformedArray& operator=(Other&& other) noexcept {
    std::destroy_at(this);
    // TODO(jbms): handle exceptions
    new (this) TransformedArray(std::forward<Other>(other));
    return *this;
  }

  /// Returns the rank of the transformed array.
  ///
  /// \returns `transform().rank()` if `has_transform()`, else
  ///     `untransformed_array().rank()`.
  RankType rank() const { return layout_.rank(); }

  /// Returns the domain of the transformed array.
  ///
  /// \returns `transform().input_domain()` if `has_transform()`, else
  ///     `untransformed_array().domain()`.
  BoxView<static_rank> domain() const { return layout_.domain(); }

  /// Returns the origin vector of the transformed array.
  span<const Index, static_rank> origin() const { return domain().origin(); }

  /// Returns the shape vector of the transformed array.
  span<const Index, static_rank> shape() const { return domain().shape(); }

  /// Returns the dimension label vector.
  ///
  /// If this transformed array is represented without an index transform, this
  /// returns a vector of empty strings.
  span<const std::string, static_rank> labels() const {
    return layout_.labels();
  }

  /// Returns the element representation.
  DataType data_type() const { return element_pointer_.data_type(); }

  /// Returns the base element pointer.
  const ElementPointer& element_pointer() const& { return element_pointer_; }
  ElementPointer& element_pointer() & { return element_pointer_; }
  ElementPointer&& element_pointer() && { return std::move(element_pointer_); }

  /// Returns `true` if the transformed array is represented using an index
  /// transform.
  ///
  /// If `false`, the transformed array is represented using an "untransformed"
  /// array.
  bool has_transform() const { return static_cast<bool>(layout_.transform_); }

  /// Returns the index transform used to represent the transformed array.
  ///
  /// If `has_transform() == false`, returns an invalid index transform for
  /// which `valid()` returns `false`.
  IndexTransformView<Rank, dynamic_rank> transform() const {
    return layout_.transform();
  }

  /// Returns `true` if this transformed array is represented using an
  /// "untransformed" array.
  bool has_untransformed_array() const {
    return layout_.has_untransformed_array();
  }

  /// Returns the "untransformed" array used to represent the transformed array.
  ///
  /// \dchecks `has_untransformed_array()`.
  /// \returns `{ element_pointer(), untransformed_strided_layout() }`.
  OffsetArrayView<Element, Rank> untransformed_array() const {
    return {element_pointer_, untransformed_strided_layout()};
  }

  /// Returns the "untransformed" array strided layout used to represent this
  /// transformed array.
  ///
  /// \dchecks `has_untransformed_array()`
  StridedLayoutView<Rank, offset_origin> untransformed_strided_layout() const {
    return layout_.untransformed_strided_layout();
  }

  /// Returns the "base" array used to represent the transformed array.
  ///
  /// 1. If the transformed array is represented using a base array, this
  ///    returns a reference to it.
  ///
  /// 2. If the transformed array is represented using an element pointer and an
  ///    index transform (but not an explicit base array), this returns an
  ///    array with the stored `element_pointer()` and an unbounded layout of
  ///    rank equal to `transform().output_rank()` and an all-1 `byte_strides`
  ///    vector.
  ///
  /// 3. If the transformed array is represented using an "untransformed" array
  ///    (and therefore no index transform or "base" array), this returns
  ///    `untransformed_array()`.
  ///
  /// \remarks Unlike the `untransformed_array()` accessor, this accessor has no
  ///     preconditions and is always valid to call.
  OffsetArrayView<Element> base_array() const {
    return {element_pointer_, layout_.base_or_untransformed_strided_layout()};
  }

  /// Returns `true` if this transformed array is represented using an explicit
  /// "base" array.
  ///
  /// \remarks This returns `false` if the transformed array is represented
  ///     using an `element_pointer()` and a `transform()` but no explicit
  ///     "base" array.
  bool has_base_array() const { return layout_.has_base_array(); }

  /// Returns the "base" array strided layout used to represent this
  /// transformed array.
  ///
  /// \dchecks `has_base_array()`
  /// \remarks Unlike the `base_array()` accessor, this accessor is
  ///     preconditioned on `has_base_array() == true`.
  StridedLayoutView<dynamic_rank, offset_origin> base_strided_layout() const {
    return layout_.base_strided_layout();
  }

  /// Materializes the transformed array as a strided array.
  ///
  /// Refer to the documentation for `TransformArray`.  Depending on
  /// `constraints` and whether the transform uses index arrays, the returned
  /// array may be newly allocated or point to a sub-region of the existing
  /// array.  In the latter case, the returned array is only valid as long as
  /// the existing array despite being stored as a `SharedArray`.
  ///
  /// \tparam OriginKind Specifies whether to retain the origin offset.
  /// \param constraints If `constraints == std::nullopt`, the returned array
  ///     may refer to `base_array()`.
  template <ArrayOriginKind OriginKind = offset_origin>
  Result<SharedArray<const Element, Rank, OriginKind>> Materialize(
      TransformArrayConstraints constraints = skip_repeated_elements) const {
    return TransformArray<OriginKind>(UnownedToShared(base_array()),
                                      transform(), constraints);
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// StatusOr<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  PipelineResultType<const TransformedArray&, Func> operator|(
      Func&& func) const& {
    return static_cast<Func&&>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<TransformedArray&&, Func> operator|(Func&& func) && {
    return static_cast<Func&&>(func)(std::move(*this));
  }

 private:
  friend class internal_index_space::TransformedArrayAccess;

  template <typename A>
  explicit TransformedArray(Access::construct_array_tag, A&& array)
      : element_pointer_(unchecked, std::forward<A>(array).element_pointer()),
        layout_(Access::construct_array_tag{},
                std::forward<A>(array).layout()) {}

  template <typename P, typename T>
  explicit TransformedArray(Access::construct_element_pointer_tag,
                            P&& element_pointer, T&& transform) noexcept
      : element_pointer_(unchecked, std::forward<P>(element_pointer)),
        layout_(Access::construct_element_pointer_tag{},
                std::forward<T>(transform)) {}

  template <typename A, typename T>
  explicit TransformedArray(Access::construct_base_array_transform_tag,
                            A&& array, T&& transform) noexcept
      : element_pointer_(unchecked, std::forward<A>(array).element_pointer()),
        layout_(Access::construct_base_array_transform_tag{},
                std::forward<A>(array).layout(), std::forward<T>(transform)) {}

  template <typename Other,
            std::enable_if_t<IsTransformedArray<
                internal::remove_cvref_t<Other>>::value>* = nullptr>
  explicit TransformedArray(Access::construct_tag, Other&& other)
      : element_pointer_(unchecked,
                         std::forward<Other>(other).element_pointer()),
        layout_(Access::construct_tag{},
                Access::layout(std::forward<Other>(other))) {}

  template <typename Other,
            std::enable_if_t<IsNormalizedTransformedArray<
                internal::remove_cvref_t<Other>>::value>* = nullptr>
  explicit TransformedArray(Access::construct_tag, Other&& other)
      : element_pointer_(unchecked,
                         std::forward<Other>(other).element_pointer()),
        layout_(Access::construct_element_pointer_tag{},
                std::forward<Other>(other).transform()) {}

  explicit TransformedArray(Access::construct_tag,
                            ElementPointer element_pointer,
                            LayoutStorage layout)
      : element_pointer_(std::move(element_pointer)),
        layout_(std::move(layout)) {}

  ElementPointer element_pointer_;
  LayoutStorage layout_;
};

/// Converts a `TransformedArray` with a non-`Shared` element pointer to
/// `TransformedArray` with a `Shared` element pointer that does not manage
/// ownership.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
///
/// This is useful for passing a `TransformedArray` with non-`Shared` element
/// pointer to a function that requires a `Shared` element pointer, when the
/// caller can ensure that the array data remains valid as long as required by
/// the callee.
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 TransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(TransformedArray<Element, Rank, LayoutCKind> array) {
  using internal_index_space::TransformedArrayAccess;
  return TransformedArrayAccess::Construct<
      TransformedArray<Shared<Element>, Rank, LayoutCKind>>(
      TransformedArrayAccess::construct_tag{},
      UnownedToShared(array.element_pointer()),
      std::move(TransformedArrayAccess::layout(array)));
}

/// Converts a `TransformedArray` with a non-`Shared` element pointer to a
/// `TransformedArray` with a `Shared` element pointer that shares the ownership
/// of the specified `owned` pointer, in the same way as the `std::shared_ptr`
/// aliasing constructor.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
template <typename T, typename Element, DimensionIndex Rank,
          ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 TransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(const std::shared_ptr<T>& owned,
                TransformedArray<Element, Rank, LayoutCKind> array) {
  using internal_index_space::TransformedArrayAccess;
  return TransformedArrayAccess::Construct<
      TransformedArray<Shared<Element>, Rank, LayoutCKind>>(
      TransformedArrayAccess::construct_tag{},
      UnownedToShared(owned, array.element_pointer()),
      std::move(TransformedArrayAccess::layout(array)));
}

/// No-op overload for an existing `Shared` element type.
///
/// The returned array shares ownership with `array`.
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
TransformedArray<Shared<Element>, Rank, LayoutCKind> UnownedToShared(
    TransformedArray<Shared<Element>, Rank, LayoutCKind> array) {
  return array;
}

/// NormalizedTransformedArray behaves like `TransformedArray` but always uses
/// the "normalized" representation of an element pointer pair with an index
/// transform.
///
/// It smaller and more efficient than `TransformedArray` due to not supporting
/// multiple representation types.
///
/// Typically, `TransformedArray` is used as a function parameter type in public
/// APIs because it can be implicitly constructed from any of the 3 supported
/// representations (untransformed array, array + transform, and element pointer
/// + transform).  Within the implementation of such a function, a
/// `NormalizedTransformedArray` is created from the `TransformedArray` argument
/// and then the `NormalizedTransformedArray` representation is used internally.
///
/// For example:
///
///     void ProcessArrayHelper(
///         NormalizedTransformedArray<void, dynamic_rank, view>
///             array, int arg) {
///       // ...
///     }
///
///     Status ProcessArray(TransformedArrayView<void, 2> array) {
///       TENSORSTORE_ASSIGN_OR_RETURN(
///           NormalizedTransformedArray<void, 2> normalized,
///           MakeNormalizedTransformedArray(array));
///       // Use `normalized.domain()` (which may differ from `array.domain()`
///       // due to implicit bounds having been resolved),
///       // `normalized.transform()`, `normalized.element_pointer()`.
///       ProcessArrayHelper(normalized, 1);
///       ProcessArrayHelper(normalized, 5);
///       // ...
///     }
///
/// In the above example, `ProcessArrayHelper` uses
/// `layout_container_kind = view` to avoid copying the transform.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
class NormalizedTransformedArray {
  static_assert(IsElementTag<ElementTagType>::value,
                "ElementTagType must be an ElementTag type.");
  static_assert(Rank == dynamic_rank || Rank >= 0,
                "Rank must be dynamic_rank or >= 0.");

 public:
  using ElementTag = ElementTagType;
  using ElementPointer = tensorstore::ElementPointer<ElementTag>;
  using Pointer = typename ElementPointer::Pointer;
  using Transform = IndexTransform<Rank, dynamic_rank, LayoutCKind>;
  using Element = typename ElementPointer::Element;
  using DataType = StaticOrDynamicDataTypeOf<Element>;
  constexpr static DimensionIndex static_rank = Transform::static_input_rank;
  constexpr static ContainerKind layout_container_kind = LayoutCKind;
  using RankType = StaticOrDynamicRank<static_rank>;

  template <ContainerKind CKind>
  using RebindContainerKind =
      NormalizedTransformedArray<ElementTagType, Rank, CKind>;

  /// Alias that evaluates to the `NormalizedTransformedArray` type with the
  /// same `ElementTag` but with the `static_input_rank` of `OtherTransform`,
  /// and a layout container kind of `container`.
  ///
  /// \tparam OtherTransform The new transform type.  Must be an instance of
  ///     `IndexTransform` or `IndexTransformView`.
  template <typename OtherTransform>
  using RebindTransform =
      NormalizedTransformedArray<ElementTagType,
                                 OtherTransform::static_input_rank, container>;

  /// Constructs an invalid transformed array.
  NormalizedTransformedArray() = default;

  /// Constructs a normalized transformed array from an element pointer and an
  /// index transform.
  ///
  /// The domain of the transformed array is equal to the input domain of the
  /// index transform.  An index vector `v` into the transformed array
  /// corresponds to the element at a byte offset of `sum(transform(v))` from
  /// `element_pointer`.
  ///
  /// \requires `ElementPointer` is constructible from `P`.
  /// \requires `Transform` is constructible from `T`.
  template <typename P, typename T,
            std::enable_if_t<internal::IsPairImplicitlyConvertible<
                P, T, ElementPointer, Transform>::value>* = nullptr>
  NormalizedTransformedArray(P&& element_pointer, T&& transform) noexcept
      : element_pointer_(std::forward<P>(element_pointer)),
        transform_(std::forward<T>(transform)) {}

  /// Copy or move constructs from another normalized transformed array.
  ///
  /// \requires `ElementPointer` is constructible from `Other::ElementPointer`.
  /// \requires `Transform` is constructible from `Other::Transform`.
  template <typename Other,
            std::enable_if_t<
                (IsNormalizedTransformedArray<
                     internal::remove_cvref_t<Other>>::value &&
                 internal::IsPairImplicitlyConvertible<
                     typename internal::remove_cvref_t<Other>::ElementPointer,
                     typename internal::remove_cvref_t<Other>::Transform,
                     ElementPointer, Transform>::value)>* = nullptr>
  NormalizedTransformedArray(Other&& other) noexcept
      : element_pointer_(std::forward<Other>(other).element_pointer()),
        transform_(std::forward<Other>(other).transform()) {}

  /// Unchecked conversion from an existing NormalizedTransformedArray.
  ///
  /// \requires `ElementPointer` is `StaticCast` constructible from
  ///     `Other::ElementPointer`.
  /// \requires `Transform` is `StaticCast` constructible from
  /// `Other::Transform`.
  template <
      typename Other,
      std::enable_if_t<
          (IsNormalizedTransformedArray<
               internal::remove_cvref_t<Other>>::value &&
           IsCastConstructible<ElementPointer,
                               typename internal::remove_cvref_t<
                                   Other>::ElementPointer>::value &&
           IsCastConstructible<Transform, typename internal::remove_cvref_t<
                                              Other>::Transform>::value)>* =
          nullptr>
  explicit NormalizedTransformedArray(unchecked_t, Other&& other) noexcept
      : element_pointer_(unchecked,
                         std::forward<Other>(other).element_pointer()),
        transform_(unchecked, std::forward<Other>(other).transform()) {}

  /// Copy or move assigns from another normalized transformed array.
  template <typename Other,
            std::enable_if_t<std::is_constructible<NormalizedTransformedArray,
                                                   Other&&>::value>* = nullptr>
  NormalizedTransformedArray& operator=(Other&& other) noexcept {
    element_pointer_ = std::forward<Other>(other).element_pointer();
    transform_ = std::forward<Other>(other).transform();
    return *this;
  }

  /// Returns the rank of the transformed array.
  ///
  /// \returns `transform().input_rank()`.
  RankType rank() const { return transform_.input_rank(); }

  /// Returns the domain of the transformed array.
  ///
  /// \returns `transform().input_domain()`.
  IndexDomainView<static_rank> domain() const { return transform_.domain(); }

  /// Returns the origin vector of the transformed array.
  span<const Index, static_rank> origin() const { return domain().origin(); }

  /// Returns the shape vector of the transformed array.
  span<const Index, static_rank> shape() const { return domain().shape(); }

  /// Returns the dimension label vector.
  span<const std::string, static_rank> labels() const {
    return transform_.input_labels();
  }

  /// Returns the element representation.
  DataType data_type() const { return element_pointer_.data_type(); }

  /// Returns the base element pointer.
  const ElementPointer& element_pointer() const& { return element_pointer_; }
  ElementPointer& element_pointer() & { return element_pointer_; }
  ElementPointer&& element_pointer() && { return std::move(element_pointer_); }

  /// Returns the transform.
  const Transform& transform() const& { return transform_; }
  Transform& transform() & { return transform_; }
  Transform&& transform() && { return std::move(transform_); }

  /// Returns a fake "base array" such that this transformed array is equivalent
  /// to applying `transform()` to `base_array()`.
  ///
  /// \returns An array with an `element_pointer` equal to
  ///     `this->element_pointer()` and a layout of rank
  ///     `transform().output_rank()` with unbounded domain and `byte_strides`
  ///     of `1`.
  ArrayView<ElementTag, dynamic_rank, offset_origin> base_array() const {
    return {element_pointer(),
            internal_index_space::GetUnboundedLayout(transform_.output_rank())};
  }

  /// Materializes the transformed array as a strided array.
  ///
  /// Refer to the documentation for `TransformArray`.  Depending on
  /// `constraints` and whether the transform uses index arrays, the returned
  /// array may be newly allocated or point to a sub-region of the existing
  /// array.  In the latter case, the returned array is only valid as long as
  /// the existing array despite being stored as a `SharedArray`.
  ///
  /// \tparam OriginKind Specifies whether to retain the origin offset.
  /// \param constraints If `constraints.allocate_constraint() == may_allocate`,
  ///     the returned array may refer to `element_pointer`.
  template <ArrayOriginKind OriginKind = offset_origin>
  Result<SharedArray<const Element, Rank, OriginKind>> Materialize(
      TransformArrayConstraints constraints = skip_repeated_elements) const {
    return TransformArray<OriginKind>(UnownedToShared(base_array()),
                                      transform(), constraints);
  }

  /// "Pipeline" operator.
  ///
  /// In the expression  `x | y`, if
  ///   * y is a function having signature `Result<U>(T)`
  ///
  /// Then operator| applies y to the value of x, returning a
  /// Result<U>. See tensorstore::Result operator| for examples.
  template <typename Func>
  PipelineResultType<const NormalizedTransformedArray&, Func> operator|(
      Func&& func) const& {
    return static_cast<Func&&>(func)(*this);
  }
  template <typename Func>
  PipelineResultType<NormalizedTransformedArray&&, Func> operator|(
      Func&& func) && {
    return static_cast<Func&&>(func)(std::move(*this));
  }

 private:
  ElementPointer element_pointer_;
  Transform transform_;
};

/// Converts a `NormalizedTransformedArray` with a non-`Shared` element pointer
/// to `NormalizedTransformedArray` with a `Shared` element pointer that does
/// not manage ownership.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
///
/// This is useful for passing a `NormalizedTransformedArray` with non-`Shared`
/// element pointer to a function that requires a `Shared` element pointer, when
/// the caller can ensure that the array data remains valid as long as required
/// by the callee.
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(NormalizedTransformedArray<Element, Rank, LayoutCKind> array) {
  return NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind>(
      UnownedToShared(array.element_pointer()), std::move(array.transform()));
}

/// Converts a `NormalizedTransformedArray` with a non-`Shared` element pointer
/// to a `NormalizedTransformedArray` with a `Shared` element pointer that
/// shares the ownership of the specified `owned` pointer, in the same way as
/// the `std::shared_ptr` aliasing constructor.
///
/// The caller is responsible for ensuring that the returned array is not used
/// after the element data to which it points becomes invalid.
template <typename T, typename Element, DimensionIndex Rank,
          ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>::value,
                 NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(const std::shared_ptr<T>& owned,
                NormalizedTransformedArray<Element, Rank, LayoutCKind> array) {
  return NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind>(
      UnownedToShared(owned, array.element_pointer()),
      std::move(array.transform()));
}

/// No-op overload for an existing `Shared` element type.
///
/// The returned array shares ownership with `array`.
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind> UnownedToShared(
    NormalizedTransformedArray<Shared<Element>, Rank, LayoutCKind> array) {
  return array;
}

/// Specializations of `StaticCastTraits` for `TransformedArray` and
/// `NormalizedTransformedArray`, which enables `StaticCast`,
/// `StaticDataTypeCast`, `ConstDataTypeCast`, and `StaticRankCast`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutContainerKind>
struct StaticCastTraits<
    NormalizedTransformedArray<ElementTagType, Rank, LayoutContainerKind>>
    : public internal_index_space::TransformedArrayCastTraits<
          NormalizedTransformedArray, ElementTagType, Rank,
          LayoutContainerKind> {};
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutContainerKind>
struct StaticCastTraits<
    TransformedArray<ElementTagType, Rank, LayoutContainerKind>>
    : public internal_index_space::TransformedArrayCastTraits<
          TransformedArray, ElementTagType, Rank, LayoutContainerKind> {};

/// Specializes the HasBoxDomain metafunction for TransformedArray.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct HasBoxDomain<TransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

/// Implements the HasBoxDomain concept for `TransformedArray`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
BoxView<Rank> GetBoxDomainOf(
    const TransformedArray<ElementTagType, Rank, LayoutCKind>& array) {
  return array.domain();
}

/// Specializes the HasBoxDomain metafunction for `NormalizedTransformedArray`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
struct HasBoxDomain<
    NormalizedTransformedArray<ElementTagType, Rank, LayoutCKind>>
    : public std::true_type {};

/// Implements the HasBoxDomain concept for `NormalizedTransformedArray`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
BoxView<Rank> GetBoxDomainOf(
    const NormalizedTransformedArray<ElementTagType, Rank, LayoutCKind>&
        array) {
  return array.domain().box();
}

template <typename Element, DimensionIndex Rank = dynamic_rank>
using TransformedSharedArray =
    TransformedArray<Shared<Element>, Rank, container>;

template <typename Element, DimensionIndex Rank = dynamic_rank>
using TransformedArrayView = TransformedArray<Element, Rank, view>;

template <typename Element, DimensionIndex Rank = dynamic_rank>
using TransformedSharedArrayView =
    TransformedArray<Shared<Element>, Rank, view>;

/// Alias that evaluates to the transformed array type corresponding to a
/// strided array type.
///
/// The resultant transformed array type has the same element pointer type,
/// rank, and layout container kind as `A`.
///
/// \requires `A` satisfies `IsArray`.
template <typename A>
using TransformedArrayTypeFromArray =
    std::enable_if_t<IsArray<A>::value,
                     TransformedArray<typename A::ElementTag, A::static_rank,
                                      A::layout_container_kind>>;

template <typename ElementTag, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
TransformedArray(Array<ElementTag, Rank, OriginKind, LayoutCKind> array)
    -> TransformedArray<ElementTag, NormalizeRankSpec(Rank), LayoutCKind>;

template <typename ElementTag, DimensionIndex Rank, ContainerKind LayoutCKind>
TransformedArray(
    NormalizedTransformedArray<ElementTag, Rank, LayoutCKind> array)
    -> TransformedArray<ElementTag, Rank, LayoutCKind>;

template <typename ElementTag, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind, DimensionIndex InputRank>
TransformedArray(
    Array<ElementTag, Rank, OriginKind, LayoutCKind> array,
    IndexTransform<InputRank, NormalizeRankSpec(Rank), LayoutCKind> transform)
    -> TransformedArray<ElementTag, InputRank, LayoutCKind>;

/// Returns an equivalent normalized transformed array.
///
/// \requires `A` satisfies `IsArray`.
template <typename A>
std::enable_if_t<
    IsArray<internal::remove_cvref_t<A>>::value,
    NormalizedTransformedArray<typename internal::remove_cvref_t<A>::ElementTag,
                               internal::remove_cvref_t<A>::static_rank>>
MakeNormalizedTransformedArray(A&& array) {
  return {
      std::forward<A>(array).element_pointer(),
      internal_index_space::TransformAccess::Make<
          IndexTransform<internal::remove_cvref_t<A>::static_rank>>(
          internal_index_space::MakeTransformFromStridedLayout(array.layout())),
  };
}

/// Returns an equivalent normalized transformed array.
///
/// \requires `A` satisfies `IsTransformedArray`.
template <typename A>
std::enable_if_t<IsTransformedArray<internal::remove_cvref_t<A>>::value,
                 Result<NormalizedTransformedArray<
                     typename internal::remove_cvref_t<A>::ElementTag,
                     internal::remove_cvref_t<A>::static_rank>>>
MakeNormalizedTransformedArray(A&& array) {
  return internal_index_space::TransformedArrayAccess::NormalizeTransform(
      std::forward<A>(array));
}

/// No-op overload that handles the case of an argument that is already a
/// normalized transformed array.
template <typename A>
std::enable_if_t<
    IsNormalizedTransformedArray<internal::remove_cvref_t<A>>::value, A&&>
MakeNormalizedTransformedArray(A&& array) {
  return std::forward<A>(array);
}

/// Alias that evaluates to the transformed array type corresponding to the
/// normalized combination of a strided array type and an index transform.
///
/// The resultant transformed array has the element pointer type of `A`, the
/// `static_rank` of `T`, and a `layout_container_kind` of `container`.
///
/// \requires `A` satifies `IsArray`.
/// \requires `T` satisfies `IsIndexTransform`.
/// \requires `A::static_rank == T::static_output_rank`.
template <typename A, typename T>
using NormalizedTransformedArrayTypeFromArrayAndTransform = std::enable_if_t<
    (IsArray<A>::value && IsIndexTransform<T>::value &&
     A::static_rank == T::static_output_rank),
    NormalizedTransformedArray<typename A::ElementTag, T::static_input_rank,
                               container>>;

/// Returns an index transform composed from a strided layout and an existing
/// index transform.  The domain of `layout` is propagated to `transform` using
/// `PropagateBounds`.
///
/// The lower and upper bounds of the returned transform are explicit.
///
/// \requires `L` satisfies `IsStridedLayout`.
/// \requires `T` satisfies `IsIndexTransform`.
/// \requires `L::static_rank == T::static_output_rank`.
/// \returns The composed IndexTransform on success, or an error from
///     `PropagateBounds` on failure.
/// \error `absl::StatusCode::kInvalidArgument` if `layout.rank()` does not
///     equal `transform.output_rank()`.
template <typename L, typename T>
inline std::enable_if_t<
    (IsStridedLayout<L>::value &&
     IsIndexTransform<internal::remove_cvref_t<T>>::value),
    Result<IndexTransform<internal::remove_cvref_t<T>::static_input_rank,
                          L::static_rank>>>
ComposeLayoutAndTransform(const L& layout, T&& transform) {
  using TX = internal::remove_cvref_t<T>;
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(auto transform_ptr,
                               MakeTransformFromStridedLayoutAndTransform(
                                   layout, TransformAccess::rep_ptr<container>(
                                               std::forward<T>(transform))));
  return TransformAccess::Make<
      IndexTransform<TX::static_input_rank, L::static_rank>>(
      std::move(transform_ptr));
}

/// Returns an equivalent transformed array with an owned layout where
/// `has_transform() == true` and `has_base_array() == false`.  The domain of
/// `array` is propagated to `transform` using `PropagateBounds`.
///
/// \requires `A` satisfies `IsArray`.
/// \requires `T` satisfies `IsIndexTransform`.
/// \requires `A::static_rank == T::static_output_rank`.
/// \error `absl::StatusCode::kInvalidArgument` if `array.rank()` is not equal
/// to
///     `transform.output_rank()`.
template <typename A, typename T>
inline Result<NormalizedTransformedArrayTypeFromArrayAndTransform<
    internal::remove_cvref_t<A>, internal::remove_cvref_t<T>>>
MakeNormalizedTransformedArray(A&& array, T&& transform) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto composed_transform,
      ComposeLayoutAndTransform(array.layout(), std::forward<T>(transform)));
  return {std::in_place, std::forward<A>(array).element_pointer(),
          std::move(composed_transform)};
}

/// Returns a copy of a transformed array as a strided array.
///
/// This behaves similarly to the MakeCopy function defined in array.h for Array
/// instances.
///
/// \requires `A` satisfies `IsTransformedArray` or
///     `IsNormalizedTransformedArray`.
/// \param transformed_array The transformed array to copy.
/// \param constraints The constraints on the layout of the returned array.
template <ArrayOriginKind OriginKind = offset_origin, typename A>
inline std::enable_if_t<
    (IsTransformedArray<A>::value || IsNormalizedTransformedArray<A>::value),
    Result<SharedOffsetArray<std::remove_const_t<typename A::Element>,
                             A::static_rank>>>
MakeCopy(const A& transformed_array, IterationConstraints constraints = {
                                         c_order, include_repeated_elements}) {
  return MakeCopy<OriginKind>(transformed_array.base_array(),
                              transformed_array.transform(), constraints);
}

namespace internal_index_space {
Status CopyTransformedArrayImpl(TransformedArrayView<const void> source,
                                TransformedArrayView<void> dest);
}

/// Copies from one transformed array to another, possibly converting the data
/// type.
///
/// \param source The source transformed array.
/// \param dest The destination transformed array.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `source` and `dest` do not
/// have
///     the same rank.
/// \error `absl::StatusCode::kOutOfRange` if `source` and `dest` do not have
/// compatible
///     domains.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
/// out-of-bounds
///     index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
/// computing output
///     indices.
/// \error `absl::StatusCode::kInvalidArgument` if `source.data_type()` is not
/// equal to, or
///     cannot be converted to, `dest.data_type()`
template <typename SourceResult, typename DestResult>
std::enable_if_t<
    (IsTransformedArrayLike<UnwrapResultType<SourceResult>>::value &&
     IsTransformedArrayLike<UnwrapResultType<DestResult>>::value),
    Status>
CopyTransformedArray(const SourceResult& source, const DestResult& dest) {
  using Source = UnwrapResultType<SourceResult>;
  using Dest = UnwrapResultType<DestResult>;
  static_assert(
      IsRankExplicitlyConvertible(Dest::static_rank, Source::static_rank),
      "Arrays must have compatible ranks.");
  static_assert(!std::is_const<typename Dest::Element>::value,
                "Dest array must have a non-const element type.");
  TENSORSTORE_RETURN_IF_ERROR(
      GetFirstErrorStatus(GetStatus(source), GetStatus(dest)));
  return internal_index_space::CopyTransformedArrayImpl(UnwrapResult(source),
                                                        UnwrapResult(dest));
}

/// Applies a function that operates on an IndexTransform to a transformed
/// array.  The result is always a `NormalizedTransformedArray` with a
/// `layout_container_type` of `container`.
///
/// This definition allows DimExpression objects to be used with transformed
/// arrays.
template <typename Expr, typename T>
internal_index_space::EnableIfTransformedArrayMapTransformResultType<
    (IsTransformedArray<internal::remove_cvref_t<T>>::value ||
     IsNormalizedTransformedArray<internal::remove_cvref_t<T>>::value),
    internal::remove_cvref_t<T>, Expr>
ApplyIndexTransform(Expr&& expr, T&& t) {
  return internal_index_space::TransformedArrayAccess::MapTransform(
      /*normalized=*/IsNormalizedTransformedArray<
          internal::remove_cvref_t<T>>{},
      std::forward<T>(t), std::forward<Expr>(expr));
}

/// Applies a function that operates on an IndexTransform to a strided
/// (non-transformed) array.  The result is always a
/// `NormalizedTransformedArray` with a `layout_container_type` of `container`.
///
/// This definition allows DimExpression objects to be used with strided arrays.
template <typename Expr, typename T>
internal_index_space::EnableIfTransformedArrayMapTransformResultType<
    IsArray<internal::remove_cvref_t<T>>::value,
    TransformedArrayTypeFromArray<internal::remove_cvref_t<T>>, Expr>
ApplyIndexTransform(Expr&& expr, T&& t) {
  return internal_index_space::TransformedArrayAccess::MapTransform(
      /*normalized=*/std::true_type{},
      MakeNormalizedTransformedArray(std::forward<T>(t)),
      std::forward<Expr>(expr));
}

namespace internal {

/// Function object implementation of Materialize()
template <ArrayOriginKind OriginKind>
struct MaterializeFn {
  TransformArrayConstraints constraints;

  template <typename A>
  inline std::enable_if_t<
      (IsTransformedArray<A>::value || IsNormalizedTransformedArray<A>::value),
      decltype(std::declval<A>().template Materialize<OriginKind>())>
  operator()(const A& array) const {
    return array.template Materialize<OriginKind>(constraints);
  }
};

}  // namespace internal

/// Materializes the transformed array as a strided array.
///
/// Refer to the documentation for `TransformArray`.  Depending on
/// `constraints` and whether the transform uses index arrays, the returned
/// array may be newly allocated or point to a sub-region of the existing
/// array.  In the latter case, the returned array is only valid as long as
/// the existing array despite being stored as a `SharedArray`.
///
/// Example:
///    auto materialized_array = array | AllDims().Diagonal() | Materialize();
///
/// \tparam OriginKind Specifies whether to retain the origin offset.
/// \param constraints If `constraints == std::nullopt`, the returned array
///     may refer to `base_array()`.
template <ArrayOriginKind OriginKind = offset_origin>
inline internal::MaterializeFn<OriginKind> Materialize(
    TransformArrayConstraints constraints = skip_repeated_elements) {
  return internal::MaterializeFn<OriginKind>{constraints};
}

namespace internal {

/// Internal untyped interface to tensorstore::IterateOverTransformedArrays.
template <std::size_t Arity>
Result<ArrayIterateResult> IterateOverTransformedArrays(
    ElementwiseClosure<Arity, Status*> closure, Status* status,
    IterationConstraints constraints,
    span<const TransformedArrayView<const void>, Arity> transformed_arrays);

}  // namespace internal

/// Jointly iterates over one or more transformed arrays with compatible
/// domains.
///
/// For each index vector `input_indices` in the domain of the transformed
/// arrays, invokes
/// `func(&TransformedArray(array).array()(output_indices)...)`. where for
/// each `array`, `output_indices` is the output index vector corresponding to
/// `input_indices`.
///
/// \requires `sizeof...(Array) > 0`
/// \requires Every `Array` type satisfies
///     `IsTransformedArrayLike<UnwrapResultType<Array>>`.
/// \param func The element-wise function.  Must return `void` or a type
///     explicitly convertible to `bool` when invoked as
///     `func(Array::Element*..., Status*)`.  Iteration stops if it returns
///     `false`.
/// \param status Status pointer to pass to `func`.
/// \param iteration_order Specifies constraints on the iteration order, and
///     whether repeated elements may be skipped.  If
///     `constraints.can_skip_repeated_elements()`, the element-wise function
///     may be invoked only once for multiple `input_indices` vectors that yield
///     the same tuple of element pointers.  If
///     `constraints.has_order_constraint()`, `func` is invoked in the order
///     given by `constraints.order_constraint_value()`.  Otherwise, iteration
///     is not guaranteed to occur in any particular order; an efficient
///     iteration order is determined automatically.
/// \param array The transformed arrays over which to iterate, which must all
///     have compatible input domains.
/// \returns An `ArrayIterateResult` object specifying whether iteration
///     completed and the number of elements successfully processed.
/// \error `absl::StatusCode::kInvalidArgument` if the transformed arrays do not
/// all have
///     the same rank.
/// \error `absl::StatusCode::kOutOfRange` if the transformed arrays do not have
/// compatible
///     domains.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
/// out-of-bounds
///     index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
/// computing output
///     indices.
template <typename Func, typename... Array>
std::enable_if_t<
    ((IsTransformedArrayLike<UnwrapResultType<Array>>::value && ...) &&
     std::is_constructible_v<
         bool,
         internal::Void::WrappedType<std::invoke_result_t<
             Func&, typename UnwrapResultType<Array>::Element*..., Status*>>>),
    Result<ArrayIterateResult>>
IterateOverTransformedArrays(Func&& func, Status* status,
                             IterationConstraints constraints,
                             const Array&... array) {
  static_assert(
      AreStaticRanksCompatible(UnwrapResultType<Array>::static_rank...),
      "Arrays must have compatible static ranks.");
  TENSORSTORE_RETURN_IF_ERROR(GetFirstErrorStatus(GetStatus(array)...));
  return internal::IterateOverTransformedArrays<sizeof...(Array)>(
      internal::SimpleElementwiseFunction<
          std::remove_reference_t<Func>(
              typename UnwrapResultType<Array>::Element...),
          Status*>::Closure(&func),
      status, constraints,
      span<const TransformedArrayView<const void>, sizeof...(Array)>(
          {TransformedArray(UnwrapResult(array))...}));
}

/// Same as above, except that `func` is called without an extra `Status`
/// pointer.
template <typename Func, typename... Array>
std::enable_if_t<
    ((IsTransformedArrayLike<UnwrapResultType<Array>>::value && ...) &&
     std::is_constructible_v<
         bool, internal::Void::WrappedType<std::invoke_result_t<
                   Func&, typename UnwrapResultType<Array>::Element*...>>>),
    Result<ArrayIterateResult>>
IterateOverTransformedArrays(Func&& func, IterationConstraints constraints,
                             const Array&... array) {
  static_assert(
      AreStaticRanksCompatible(UnwrapResultType<Array>::static_rank...),
      "Arrays must have compatible static ranks.");
  TENSORSTORE_RETURN_IF_ERROR(GetFirstErrorStatus(GetStatus(array)...));
  const auto func_wrapper =
      [&func](typename UnwrapResultType<Array>::Element*... ptr, Status*) {
        return func(ptr...);
      };
  return internal::IterateOverTransformedArrays<sizeof...(Array)>(
      internal::SimpleElementwiseFunction<
          decltype(func_wrapper)(typename UnwrapResultType<Array>::Element...),
          Status*>::Closure(&func_wrapper),
      /*status=*/nullptr, constraints,
      span<const TransformedArrayView<const void>, sizeof...(Array)>(
          {TransformedArrayView<const void>(UnwrapResult(array))...}));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_TRANSFORMED_ARRAY_H_
