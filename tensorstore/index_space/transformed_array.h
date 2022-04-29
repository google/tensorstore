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
#include <string>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

template <typename ElementTagType, DimensionIndex Rank = dynamic_rank,
          ContainerKind LayoutCKind = container>
class TransformedArray;

/// Alias for a `TransformedArray` where the `IndexTransform` is
/// stored by unowned reference.
///
/// \relates TransformedArray
template <typename ElementTagType, DimensionIndex Rank = dynamic_rank>
using TransformedArrayView = TransformedArray<ElementTagType, Rank>;

/// Alias for a `TransformedArray` where the data pointer is stored as
/// an `std::shared_ptr`.
///
/// \relates TransformedArray
template <typename Element, DimensionIndex Rank = dynamic_rank,
          ContainerKind LayoutCKind = container>
using TransformedSharedArray =
    TransformedArray<Shared<Element>, Rank, LayoutCKind>;

/// Alias for a `TransformedArray` where the data pointer is stored as
/// an `std::shared_ptr` and the `IndexTransform` is stored by unowned
/// reference.
///
/// \relates TransformedArray
template <typename Element, DimensionIndex Rank = dynamic_rank>
using TransformedSharedArrayView =
    TransformedArray<Shared<Element>, Rank, view>;

/// Bool-valued metafunction that evaluates to `true` if `T` is an instance of
/// `TransformedArray`.
///
/// \relates TransformedArray
template <typename T>
constexpr inline bool IsTransformedArray = false;

template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
constexpr inline bool
    IsTransformedArray<TransformedArray<ElementTagType, Rank, LayoutCKind>> =
        true;

/// Bool-valued metafunction that evaluates to `true` if `T` satisfies
/// `IsArray` or `IsTransformedArray`.
///
/// \relates TransformedArray
template <typename T>
constexpr inline bool IsTransformedArrayLike =
    IsArray<T> || IsTransformedArray<T>;

namespace internal_index_space {
TransformRep::Ptr<> MakeTransformFromStridedLayout(
    StridedLayoutView<dynamic_rank, offset_origin> layout);
Result<TransformRep::Ptr<>> MakeTransformFromStridedLayoutAndTransform(
    StridedLayoutView<dynamic_rank, offset_origin> layout,
    TransformRep::Ptr<> transform);

/// Returns a layout of rank `rank` with a domain of `Box(rank)` and a
/// `byte_strides` vector of `GetConstantVector<Index, 1>(output_rank)`.
StridedLayoutView<dynamic_rank, offset_origin> GetUnboundedLayout(
    DimensionIndex rank);

/// Type alias that evaluates to the result of calling MapTransform with a given
/// function type.
template <typename A, typename Func>
using TransformedArrayMapTransformResultType = FlatMapResultType<
    A::template RebindTransform,
    internal::remove_cvref_t<std::invoke_result_t<
        Func, const typename internal::remove_cvref_t<A>::Transform&>>>;

/// Returns a new `Result`-wrapped `TransformedArray` where the index transform
/// has been mapped by the specified function.
template <typename A, typename Func>
static TransformedArrayMapTransformResultType<internal::remove_cvref_t<A>, Func>
TransformedArrayMapTransform(A&& a, Func&& func) {
  using ResultType =
      TransformedArrayMapTransformResultType<internal::remove_cvref_t<A>, Func>;
  using AX = internal::remove_cvref_t<A>;
  using MappedTransform = UnwrapResultType<
      std::invoke_result_t<Func, const typename AX::Transform&>>;
  return MapResult(
      [&](MappedTransform transform) {
        return typename ResultType::value_type{
            std::forward<A>(a).element_pointer(), std::move(transform)};
      },
      std::forward<Func>(func)(std::forward<A>(a).transform()));
}

// Used to implement `EnableIfTransformedArrayMapTransformResultType` below.
template <bool Condition>
struct ConditionalTransformedArrayMapTransformResultType {
  template <typename A, typename Func>
  using type = TransformedArrayMapTransformResultType<A, Func>;
};

template <>
struct ConditionalTransformedArrayMapTransformResultType<false> {};

/// Equivalent to:
///
///     std::enable_if_t<
///         Condition,
///         TransformedArrayMapTransformResultType<A, Func>>
///
/// except that `TransformedArrayMapTransformResultType<A, Func>` is not
/// evaluated if `Condition` is `false` (this avoids the potential for SFINAE
/// loops).
template <bool Condition, typename A, typename Func>
using EnableIfTransformedArrayMapTransformResultType =
    typename ConditionalTransformedArrayMapTransformResultType<
        Condition>::template type<A, Func>;

std::string DescribeTransformedArrayForCast(DataType dtype,
                                            DimensionIndex rank);

}  // namespace internal_index_space

/// View through an index transform of an in-memory array.
///
/// Example of making a transformed array directly::
///
///    // Transform that extracts the diagonal.
///    IndexTransform<> t = IndexTransformBuilder<>(1, 2)
///                             .output_single_input_dimension(0, 0)
///                             .output_single_input_dimension(1, 0)
///                             .Finalize()
///                             .value();
///    auto source_array = MakeArray<int>({1, 2, 3}, {4, 5, 6}, {7, 8, 9});
///    auto dest_array = AllocateArray<int>({3});
///    TENSORSTORE_ASSIGN_OR_RETURN(auto transformed_array, source_array | t);
///
///    // Copy the diagonal of source_array to dest_array.
///    IterateOverTransformedArrays([](const int* x, int* y) { *y = *x; },
///                                 /*constraints=*/{}, transformed_array,
///                                 dest_array);
///    // dest_array equals {1, 5, 9}.
///
/// Example of making a transformed array using `DimExpression`::
///
///     TENSORSTORE_ASSIGN_OR_RETURN(
///       auto transformed_array,
///       tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}}) |
///       tensorstore::Dims(0).TranslateTo(10) |
///       tensorstore::Dims(0, 1).IndexVectorArraySlice(
///         tensorstore::MakeArray<Index>({10, 1}, {11, 1}, {11, 2})) |
///       tensorstore::Dims(0).Label("a"));
///
/// Logically, a `TransformedArray` is represented by an
/// `ElementPointer<ElementTagType>` and an `IndexTransform<Rank, LayoutCKind>`.
/// The original `StridedLayout` of the array is represented implicitly as part
/// of the `IndexTransform`.
///
/// \tparam ElementTagType Must satisfy `IsElementTag`.  Either ``T`` or
///     ``Shared<T>``, where ``T`` satisfies ``IsElementType<T>``.
/// \tparam Rank The static rank of the transformed array.  May be
///     `dynamic_rank` to allow the rank to be determined at run time.
/// \tparam LayoutCKind Either `container` or `view`.  If equal to `container`,
///     the transformed array owns the index transform.  If equal to `view`, an
///     unowned reference to the index transform is stored.
/// \ingroup array
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
class TransformedArray {
  static_assert(IsElementTag<ElementTagType>,
                "ElementTagType must be an ElementTag type.");
  static_assert(Rank == dynamic_rank || Rank >= 0,
                "Rank must be dynamic_rank or >= 0.");

 public:
  /// Element tag type of the array.
  using ElementTag = ElementTagType;

  /// Element pointer type.
  using ElementPointer = tensorstore::ElementPointer<ElementTag>;

  /// Data pointer type.
  using Pointer = typename ElementPointer::Pointer;

  /// Index transform type.
  using Transform = IndexTransform<Rank, dynamic_rank, LayoutCKind>;

  /// Element type.
  using Element = typename ElementPointer::Element;

  /// Data type representation.
  using DataType = dtype_t<Element>;

  /// Compile-time rank constraint, or `dynamic_rank` if the rank is determined
  /// at run time.
  constexpr static DimensionIndex static_rank = Transform::static_input_rank;

  /// Layout container kind.
  constexpr static ContainerKind layout_container_kind = LayoutCKind;

  /// Static or dynamic rank representation type.
  using RankType = StaticOrDynamicRank<static_rank>;

  template <ContainerKind CKind>
  using RebindContainerKind = TransformedArray<ElementTagType, Rank, CKind>;

  /// Alias that evaluates to the `TransformedArray` type with the same
  /// `ElementTag` but with the `IndexTransform::static_input_rank` of
  /// `OtherTransform`, and a layout container kind of `container`.
  ///
  /// \tparam OtherTransform The new transform type.  Must be an instance of
  ///     `IndexTransform` or `IndexTransformView`.
  template <typename OtherTransform>
  // NONITPICK: OtherTransform::static_input_rank
  using RebindTransform =
      TransformedArray<ElementTagType, OtherTransform::static_input_rank>;

  /// Constructs an invalid transformed array.
  ///
  /// \id default
  TransformedArray() = default;

  /// Constructs a normalized transformed array from an element pointer and an
  /// index transform.
  ///
  /// The domain of the transformed array is equal to the input domain of the
  /// index transform.  An index vector ``v`` into the transformed array
  /// corresponds to the element at a byte offset of ``sum(transform(v))`` from
  /// `element_pointer`.
  ///
  /// \requires `ElementPointer` is constructible from `P`.
  /// \requires `Transform` is constructible from `T`.
  /// \id element_pointer, transform
  template <typename P, typename T,
            std::enable_if_t<internal::IsPairImplicitlyConvertible<
                P, T, ElementPointer, Transform>>* = nullptr>
  TransformedArray(P&& element_pointer, T&& transform) noexcept
      : element_pointer_(std::forward<P>(element_pointer)),
        transform_(std::forward<T>(transform)) {}

  /// Constructs a transformed array from a regular strided `Array`.
  ///
  /// \id array
  template <typename A, ContainerKind SfinaeC = LayoutCKind,
            typename = std::enable_if_t<
                (SfinaeC == container && IsArray<internal::remove_cvref_t<A>> &&
                 std::is_convertible_v<
                     typename internal::remove_cvref_t<A>::ElementPointer,
                     ElementPointer> &&
                 RankConstraint::Implies(
                     internal::remove_cvref_t<A>::static_rank, Rank))>>
  // NONITPICK: std::remove_cvref_t<A>::ElementPointer
  // NONITPICK: std::remove_cvref_t<A>::static_rank
  TransformedArray(A&& array)
      : element_pointer_(std::forward<A>(array).element_pointer()),
        transform_(internal_index_space::TransformAccess::Make<Transform>(
            internal_index_space::MakeTransformFromStridedLayout(
                array.layout()))) {}

  /// Copy or move constructs from another normalized transformed array.
  ///
  /// \requires `ElementPointer` is constructible from
  ///     ``Other::ElementPointer``.
  /// \requires `Transform` is constructible from ``Other::Transform``.
  /// \id convert
  template <typename Other,
            std::enable_if_t<
                (IsTransformedArray<internal::remove_cvref_t<Other>> &&
                 internal::IsPairImplicitlyConvertible<
                     typename internal::remove_cvref_t<Other>::ElementPointer,
                     typename internal::remove_cvref_t<Other>::Transform,
                     ElementPointer, Transform>)>* = nullptr>
  // NONITPICK: std::remove_cvref_t<Other>::ElementPointer
  // NONITPICK: std::remove_cvref_t<Other>::Transform
  TransformedArray(Other&& other) noexcept
      : element_pointer_(std::forward<Other>(other).element_pointer()),
        transform_(std::forward<Other>(other).transform()) {}

  /// Unchecked conversion from an existing `TransformedArray`.
  ///
  /// \id unchecked
  template <typename Other,
            std::enable_if_t<(
                IsTransformedArray<internal::remove_cvref_t<Other>> &&
                IsStaticCastConstructible<
                    ElementPointer,
                    typename internal::remove_cvref_t<Other>::ElementPointer> &&
                IsStaticCastConstructible<Transform,
                                          typename internal::remove_cvref_t<
                                              Other>::Transform>)>* = nullptr>
  // NONITPICK: std::remove_cvref_t<Other>::ElementPointer
  // NONITPICK: std::remove_cvref_t<Other>::Transform
  explicit TransformedArray(unchecked_t, Other&& other) noexcept
      : element_pointer_(unchecked,
                         std::forward<Other>(other).element_pointer()),
        transform_(unchecked, std::forward<Other>(other).transform()) {}

  /// Unchecked conversion from an existing `Array`.
  ///
  /// \id unchecked, array
  template <
      typename A, ContainerKind SfinaeC = LayoutCKind,
      std::enable_if_t<
          (SfinaeC == container && IsArray<internal::remove_cvref_t<A>> &&
           IsStaticCastConstructible<
               ElementPointer,
               typename internal::remove_cvref_t<A>::ElementPointer> &&
           RankConstraint::EqualOrUnspecified(
               internal::remove_cvref_t<A>::static_rank, Rank))>* = nullptr>
  // NONITPICK: std::remove_cvref_t<A>::ElementPointer
  // NONITPICK: std::remove_cvref_t<A>::static_rank
  explicit TransformedArray(unchecked_t, A&& array) noexcept
      : element_pointer_(unchecked, std::forward<A>(array).element_pointer()),
        transform_(unchecked,
                   internal_index_space::TransformAccess::Make<Transform>(
                       internal_index_space::MakeTransformFromStridedLayout(
                           array.layout()))) {}

  /// Copy or move assigns from another normalized transformed array.
  ///
  /// \id convert
  template <typename Other,
            std::enable_if_t<
                (IsTransformedArray<internal::remove_cvref_t<Other>> &&
                 internal::IsPairImplicitlyConvertible<
                     typename internal::remove_cvref_t<Other>::ElementPointer,
                     typename internal::remove_cvref_t<Other>::Transform,
                     ElementPointer, Transform>)>* = nullptr>
  TransformedArray& operator=(Other&& other) noexcept {
    element_pointer_ = std::forward<Other>(other).element_pointer();
    transform_ = std::forward<Other>(other).transform();
    return *this;
  }

  /// Copy or move assigns from another `Array`.
  ///
  /// \id array
  template <typename A, ContainerKind SfinaeC = LayoutCKind,
            typename = std::enable_if_t<
                (SfinaeC == container && IsArray<internal::remove_cvref_t<A>> &&
                 std::is_assignable_v<
                     ElementPointer,
                     typename internal::remove_cvref_t<A>::ElementPointer> &&
                 RankConstraint::Implies(
                     internal::remove_cvref_t<A>::static_rank, Rank))>>
  // NONITPICK: std::remove_cvref_t<A>::ElementPointer
  // NONITPICK: std::remove_cvref_t<A>::static_rank
  TransformedArray& operator=(A&& array) noexcept {
    element_pointer_ = std::forward<A>(array).element_pointer();
    transform_ = internal_index_space::TransformAccess::Make<Transform>(
        internal_index_space::MakeTransformFromStridedLayout(array.layout()));
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
  DataType dtype() const { return element_pointer_.dtype(); }

  /// Returns the base element pointer.
  const ElementPointer& element_pointer() const& { return element_pointer_; }
  ElementPointer& element_pointer() & { return element_pointer_; }
  ElementPointer&& element_pointer() && { return std::move(element_pointer_); }

  /// Returns a raw pointer to the first element of the array.
  Element* data() const { return element_pointer_.data(); }

  /// Returns the transform.
  const Transform& transform() const& { return transform_; }
  Transform& transform() & { return transform_; }
  Transform&& transform() && { return std::move(transform_); }

  /// Returns a fake "base array" such that this transformed array is equivalent
  /// to applying `transform()` to `base_array()`.
  ///
  /// \returns An array with an `Array::element_pointer` equal to
  ///     `element_pointer()` and a layout of rank `transform().output_rank()`
  ///     with unbounded domain and `Array::byte_strides` of `1`.
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
  /// In the expression ``x | y``, if ``y`` is a function having signature
  /// ``Result<U>(T)``, then `operator|` applies ``y`` to the value of ``x``,
  /// returning a ``Result<U>``.
  ///
  /// See `tensorstore::Result::operator|` for examples.
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
  ElementPointer element_pointer_;
  Transform transform_;
};

/// Converts an arbitrary `TransformedArray` to a `TransformedSharedArray`.
///
/// .. warning::
///
///    The caller is responsible for ensuring that the returned array is not
///    used after the element data to which it points becomes invalid.
///
/// \param owned If specified, the returned array shares ownership with the
///     `owned` pointer, in the same way as the `std::shared_ptr` aliasing
///     constructor.  Cannot be specified if `array` is already a
///     `TransformedSharedArray`.
/// \param array Existing array to return.  If `array` is already a
///     `TransformedSharedArray`, it is simply returned as is, i.e. the returned
///     array shares ownership with `array`.
/// \relates TransformedArray
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>,
                 TransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(TransformedArray<Element, Rank, LayoutCKind> array) {
  return TransformedArray<Shared<Element>, Rank, LayoutCKind>(
      UnownedToShared(array.element_pointer()), std::move(array.transform()));
}
template <typename T, typename Element, DimensionIndex Rank,
          ContainerKind LayoutCKind>
std::enable_if_t<!IsShared<Element>,
                 TransformedArray<Shared<Element>, Rank, LayoutCKind>>
UnownedToShared(const std::shared_ptr<T>& owned,
                TransformedArray<Element, Rank, LayoutCKind> array) {
  return TransformedArray<Shared<Element>, Rank, LayoutCKind>(
      UnownedToShared(owned, array.element_pointer()),
      std::move(array.transform()));
}
template <typename Element, DimensionIndex Rank, ContainerKind LayoutCKind>
TransformedArray<Shared<Element>, Rank, LayoutCKind> UnownedToShared(
    TransformedArray<Shared<Element>, Rank, LayoutCKind> array) {
  return array;
}

// Specialization of `StaticCastTraits` for `TransformedArray`, which
// enables `StaticCast`, `StaticDataTypeCast`, `ConstDataTypeCast`, and
// `StaticRankCast`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutContainerKind>
struct StaticCastTraits<
    TransformedArray<ElementTagType, Rank, LayoutContainerKind>>
    : public DefaultStaticCastTraits<
          TransformedArray<ElementTagType, Rank, LayoutContainerKind>> {
  using type = TransformedArray<ElementTagType, Rank, LayoutContainerKind>;

  template <typename TargetElement>
  using RebindDataType = TransformedArray<
      typename ElementTagTraits<ElementTagType>::template rebind<TargetElement>,
      Rank, LayoutContainerKind>;

  template <DimensionIndex TargetRank>
  using RebindRank =
      TransformedArray<ElementTagType, TargetRank, LayoutContainerKind>;

  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return RankConstraint::EqualOrUnspecified(other.rank(), Rank) &&
           IsPossiblySameDataType(other.dtype(), typename type::DataType());
  }

  static std::string Describe() {
    return internal_index_space::DescribeTransformedArrayForCast(
        typename type::DataType(), Rank);
  }

  static std::string Describe(const type& value) {
    return internal_index_space::DescribeTransformedArrayForCast(value.dtype(),
                                                                 value.rank());
  }
};

// Specializes the HasBoxDomain metafunction for `TransformedArray`.
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
constexpr inline bool
    HasBoxDomain<TransformedArray<ElementTagType, Rank, LayoutCKind>> = true;

/// Implements the `HasBoxDomain` concept for `TransformedArray`.
///
/// \relates TransformedArray
/// \id TransformedArray
template <typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutCKind>
BoxView<Rank> GetBoxDomainOf(
    const TransformedArray<ElementTagType, Rank, LayoutCKind>& array) {
  return array.domain().box();
}

/// Alias that evaluates to the transformed array type corresponding to a
/// strided array type.
///
/// The resultant transformed array type has the same element pointer type,
/// rank, and layout container kind as `A`.
///
/// \relates TransformedArray
template <typename A>
// NONITPICK: A::ElementTag
// NONITPICK: A::static_rank
// NONITPICK: A::layout_container_kind
using TransformedArrayTypeFromArray =
    std::enable_if_t<IsArray<A>,
                     TransformedArray<typename A::ElementTag, A::static_rank,
                                      A::layout_container_kind>>;

template <typename ElementTag, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind LayoutCKind>
TransformedArray(Array<ElementTag, Rank, OriginKind, LayoutCKind> array)
    -> TransformedArray<ElementTag, RankConstraint::FromInlineRank(Rank)>;

/// Alias that evaluates to the transformed array type corresponding to the
/// normalized combination of a strided array type and an index transform.
///
/// The resultant transformed array has the element pointer type of `A`, the
/// `static_rank` of `T`, and a `layout_container_kind` of `container`.
///
/// \relates TransformedArray
template <typename A, typename T>
// NONITPICK: A::static_rank
// NONITPICK: T::static_input_rank
// NONITPICK: T::static_output_rank
// NONITPICK: A::ElementTag
using TransformedArrayTypeFromArrayAndTransform = std::enable_if_t<
    (IsArray<A> && IsIndexTransform<T> &&
     A::static_rank == T::static_output_rank),
    TransformedArray<typename A::ElementTag, T::static_input_rank, container>>;

/// Returns an index transform composed from a strided layout and an existing
/// index transform.
///
/// The domain of `layout` is propagated to `transform` using `PropagateBounds`.
///
/// The lower and upper bounds of the returned transform are explicit.
///
/// \returns The composed `IndexTransform` on success, or an error from
///     `PropagateBounds` on failure.
/// \error `absl::StatusCode::kInvalidArgument` if `layout.rank()` does not
///     equal `transform.output_rank()`.
/// \relates TransformedArray
template <DimensionIndex R, ArrayOriginKind O, ContainerKind AC, typename T>
// NONITPICK: std::remove_cvref_t<T>::static_input_rank
inline std::enable_if_t<
    (IsIndexTransform<internal::remove_cvref_t<T>>),
    Result<IndexTransform<internal::remove_cvref_t<T>::static_input_rank,
                          RankConstraint::FromInlineRank(R)>>>
ComposeLayoutAndTransform(const StridedLayout<R, O, AC>& layout,
                          T&& transform) {
  static_assert(RankConstraint::FromInlineRank(R) ==
                internal::remove_cvref_t<T>::static_output_rank);
  using TX = internal::remove_cvref_t<T>;
  using internal_index_space::TransformAccess;
  TENSORSTORE_ASSIGN_OR_RETURN(auto transform_ptr,
                               MakeTransformFromStridedLayoutAndTransform(
                                   layout, TransformAccess::rep_ptr<container>(
                                               std::forward<T>(transform))));
  return TransformAccess::Make<
      IndexTransform<TX::static_input_rank, RankConstraint::FromInlineRank(R)>>(
      std::move(transform_ptr));
}

/// Returns a transformed array representing `transform` applied to `array`.
///
/// The domain of `array` is propagated to `transform` using `PropagateBounds`.
///
/// \requires `A` satisfies `IsArray`.
/// \requires `T` satisfies `IsIndexTransform`.
/// \requires ``A::static_rank == T::static_output_rank``.
/// \error `absl::StatusCode::kInvalidArgument` if `array.rank()` is not equal
///     to `transform.output_rank()`.
/// \relates TransformedArray
template <typename A, typename T>
inline Result<TransformedArrayTypeFromArrayAndTransform<
    internal::remove_cvref_t<A>, internal::remove_cvref_t<T>>>
MakeTransformedArray(A&& array, T&& transform) {
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
/// \tparam OriginKind Origin kind of the returned array.  If equal to
///     `offset_origin` (the default), the returned array has the same origin as
///     `transformed_array`.  If equal to `zero_origin`, the origin of each
///     dimension is translated to zero.
/// \param transformed_array The transformed array to copy.
/// \param constraints The constraints on the layout of the returned array.
/// \id transformed_array
/// \relates TransformedArray
template <ArrayOriginKind OriginKind = offset_origin, typename A>
// NONITPICK: A::Element
// NONITPICK: A::static_rank
inline std::enable_if_t<
    IsTransformedArray<A>,
    Result<SharedArray<std::remove_const_t<typename A::Element>, A::static_rank,
                       OriginKind>>>
MakeCopy(const A& transformed_array, IterationConstraints constraints = {
                                         c_order, include_repeated_elements}) {
  return MakeCopy<OriginKind>(transformed_array.base_array(),
                              transformed_array.transform(), constraints);
}

namespace internal_index_space {
absl::Status CopyTransformedArrayImpl(TransformedArrayView<const void> source,
                                      TransformedArrayView<void> dest);
}

/// Copies from one transformed array to another, possibly converting the data
/// type.
///
/// \param source The source transformed array.
/// \param dest The destination transformed array.
/// \returns `absl::Status()` on success.
/// \error `absl::StatusCode::kInvalidArgument` if `source` and `dest` do not
///     have the same rank.
/// \error `absl::StatusCode::kOutOfRange` if `source` and `dest` do not have
///     compatible domains.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing output indices.
/// \error `absl::StatusCode::kInvalidArgument` if `source.dtype()` is not equal
///     to, or cannot be converted to, `dest.dtype()`
/// \relates TransformedArray
template <typename SourceResult, typename DestResult>
std::enable_if_t<(IsTransformedArrayLike<UnwrapResultType<SourceResult>> &&
                  IsTransformedArrayLike<UnwrapResultType<DestResult>>),
                 absl::Status>
CopyTransformedArray(const SourceResult& source, const DestResult& dest) {
  using Source = UnwrapResultType<SourceResult>;
  using Dest = UnwrapResultType<DestResult>;
  static_assert(RankConstraint::EqualOrUnspecified(Dest::static_rank,
                                                   Source::static_rank),
                "Arrays must have compatible ranks.");
  static_assert(!std::is_const_v<typename Dest::Element>,
                "Dest array must have a non-const element type.");
  if constexpr (IsResult<SourceResult>) {
    if (!source.ok()) return source.status();
  }
  if constexpr (IsResult<DestResult>) {
    if (!dest.ok()) return dest.status();
  }
  return internal_index_space::CopyTransformedArrayImpl(UnwrapResult(source),
                                                        UnwrapResult(dest));
}

/// Applies a function that operates on an `IndexTransform` to a transformed
/// array.  The result is always a `TransformedArray`.
///
/// This allows `DimExpression` objects to be used with transformed arrays.
///
/// \relates TransformedArray
/// \id TransformedArray
template <typename Expr, typename T>
internal_index_space::EnableIfTransformedArrayMapTransformResultType<
    IsTransformedArray<internal::remove_cvref_t<T>>,
    internal::remove_cvref_t<T>, Expr>
ApplyIndexTransform(Expr&& expr, T&& t) {
  return internal_index_space::TransformedArrayMapTransform(
      std::forward<T>(t), std::forward<Expr>(expr));
}

/// Applies a function that operates on an IndexTransform to a strided
/// (non-transformed) array.  The result is always a `TransformedArray`.
///
/// This allows `DimExpression` objects to be used with regular strided arrays.
///
/// \relates Array
/// \id Array
template <typename Expr, typename T>
internal_index_space::EnableIfTransformedArrayMapTransformResultType<
    IsArray<internal::remove_cvref_t<T>>,
    TransformedArrayTypeFromArray<internal::remove_cvref_t<T>>, Expr>
ApplyIndexTransform(Expr&& expr, T&& t) {
  return internal_index_space::TransformedArrayMapTransform(
      TransformedArray(std::forward<T>(t)), std::forward<Expr>(expr));
}

namespace internal {

/// Function object implementation of Materialize()
template <ArrayOriginKind OriginKind>
struct MaterializeFn {
  TransformArrayConstraints constraints;

  template <typename A>
  inline std::enable_if_t<
      (IsTransformedArray<A> || IsTransformedArray<A>),
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
/// Example::
///
///    auto materialized_array = array | AllDims().Diagonal() | Materialize();
///
/// \tparam OriginKind Specifies whether to retain the origin offset.
/// \param constraints If `constraints == std::nullopt`, the returned array may
///     refer to the existing array data.
/// \relates TransformedArray
template <ArrayOriginKind OriginKind = offset_origin>
inline internal::MaterializeFn<OriginKind> Materialize(
    TransformArrayConstraints constraints = skip_repeated_elements) {
  return internal::MaterializeFn<OriginKind>{constraints};
}

namespace internal {

/// Internal untyped interface to tensorstore::IterateOverTransformedArrays.
template <std::size_t Arity>
Result<ArrayIterateResult> IterateOverTransformedArrays(
    ElementwiseClosure<Arity, absl::Status*> closure, absl::Status* status,
    IterationConstraints constraints,
    span<const TransformedArrayView<const void>, Arity> transformed_arrays);

}  // namespace internal

/// Jointly iterates over one or more transformed arrays with compatible
/// domains.
///
/// For each index vector ``input_indices`` in the domain of the transformed
/// arrays, invokes::
///
///     func(&TransformedArray(arrays).array()(output_indices)...)
///
/// where for each of the `arrays`, ``output_indices`` is the output index
/// vector corresponding to ``input_indices``.
///
/// \requires `sizeof...(Arrays) > 0`
/// \param func The element-wise function.  Must return `void` or `bool` when
///     invoked with ``(Array::Element*...)``, or as
///     ``(Array::Element*..., absl::Status*)`` if `status` is specified.
///     Iteration stops if the return value of `func` is `false`.
/// \param status The `absl::Status` pointer to pass through the `func`.
/// \param constraints Specifies constraints on the iteration order, and whether
///     repeated elements may be skipped.  If
///     `constraints.can_skip_repeated_elements()`, the element-wise function
///     may be invoked only once for multiple ``input_indices`` vectors that
///     yield the same tuple of element pointers.  If
///     `constraints.has_order_constraint()`, `func` is invoked in the order
///     given by `constraints.order_constraint_value()`.  Otherwise, iteration
///     is not guaranteed to occur in any particular order; an efficient
///     iteration order is determined automatically.
/// \param arrays The transformed arrays over which to iterate, which must all
///     have compatible input domains.
/// \returns An `ArrayIterateResult` object specifying whether iteration
///     completed and the number of elements successfully processed.
/// \error `absl::StatusCode::kInvalidArgument` if the transformed arrays do not
///     all have the same rank.
/// \error `absl::StatusCode::kOutOfRange` if the transformed arrays do not have
///     compatible domains.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing output indices.
/// \relates TransformedArray
template <typename Func, typename... Arrays>
std::enable_if_t<
    ((IsTransformedArrayLike<UnwrapResultType<Arrays>> && ...) &&
     std::is_constructible_v<
         bool, internal::Void::WrappedType<std::invoke_result_t<
                   Func&, typename UnwrapResultType<Arrays>::Element*...,
                   absl::Status*>>>),
    Result<ArrayIterateResult>>
IterateOverTransformedArrays(Func&& func, absl::Status* status,
                             IterationConstraints constraints,
                             const Arrays&... arrays) {
  static_assert(RankConstraint::EqualOrUnspecified(
                    {UnwrapResultType<Arrays>::static_rank...}),
                "Arrays must have compatible static ranks.");
  return tensorstore::MapResult(
      [&](auto&&... unwrapped_array) {
        return internal::IterateOverTransformedArrays<sizeof...(Arrays)>(
            internal::SimpleElementwiseFunction<
                std::remove_reference_t<Func>(
                    typename UnwrapResultType<Arrays>::Element...),
                absl::Status*>::Closure(&func),
            status, constraints,
            span<const TransformedArrayView<const void>, sizeof...(Arrays)>(
                {TransformedArray(unwrapped_array)...}));
      },
      arrays...);
}
template <typename Func, typename... Arrays>
std::enable_if_t<
    ((IsTransformedArrayLike<UnwrapResultType<Arrays>> && ...) &&
     std::is_constructible_v<
         bool, internal::Void::WrappedType<std::invoke_result_t<
                   Func&, typename UnwrapResultType<Arrays>::Element*...>>>),
    Result<ArrayIterateResult>>
IterateOverTransformedArrays(Func&& func, IterationConstraints constraints,
                             const Arrays&... arrays) {
  static_assert(RankConstraint::EqualOrUnspecified(
                    {UnwrapResultType<Arrays>::static_rank...}),
                "Arrays must have compatible static ranks.");
  return tensorstore::MapResult(
      [&](auto&&... unwrapped_array) {
        const auto func_wrapper =
            [&func](typename UnwrapResultType<Arrays>::Element*... ptr,
                    absl::Status*) { return func(ptr...); };
        return internal::IterateOverTransformedArrays<sizeof...(Arrays)>(
            internal::SimpleElementwiseFunction<
                decltype(func_wrapper)(
                    typename UnwrapResultType<Arrays>::Element...),
                absl::Status*>::Closure(&func_wrapper),
            /*status=*/nullptr, constraints,
            span<const TransformedArrayView<const void>, sizeof...(Arrays)>(
                {TransformedArrayView<const void>(unwrapped_array)...}));
      },
      arrays...);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_TRANSFORMED_ARRAY_H_
