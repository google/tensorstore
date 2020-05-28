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

#ifndef TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORMED_ARRAY_IMPL_H_
#define TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORMED_ARRAY_IMPL_H_

/// \file
/// Implementation details for TransformedArrayBase.

#include <memory>
#include <type_traits>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

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

/// Holds a raw pointer to a transform along with 2 additional bits used by
/// `TransformedArrayAccess::LayoutStorage` to indicate which type of
/// StridedLayout is also stored.
using RawTaggedTransformPtr = internal::TaggedPtr<TransformRep, 2>;

/// Represents a transform along with tag bits, where the transform may either
/// be stored as an unowned raw pointer (if `CKind == view`) or a
/// reference-counted pointer (if `CKind == container`).
///
/// This is used by `TransformedArrayAccess::LayoutStorage<Rank, CKind>` to
/// store the optional transform pointer and information about what type of
/// strided layout is stored.
///
/// This supports implicit conversions between different values of `CKind`.
template <ContainerKind CKind>
class TaggedTransformPtr;

/// Trivial wrapper around `RawTaggedTransformPtr` that enables implicit
/// conversion from `TaggedTransformPtr<container>` and construction from
/// `TransformRep::Ptr<>`.
template <>
class TaggedTransformPtr<view> : public RawTaggedTransformPtr {
 public:
  using RawTaggedTransformPtr::RawTaggedTransformPtr;
  TaggedTransformPtr(RawTaggedTransformPtr p) : RawTaggedTransformPtr(p) {}
  template <ContainerKind OtherCKind>
  explicit TaggedTransformPtr(const TaggedTransformPtr<OtherCKind>& other)
      : RawTaggedTransformPtr(other.get(), other.tag()) {}
  explicit TaggedTransformPtr(const TransformRep::Ptr<>& p, std::uintptr_t tag)
      : RawTaggedTransformPtr(p.get(), tag) {}

  using RawTaggedTransformPtr::get;
  using RawTaggedTransformPtr::tag;
};

/// Wrapper around `internal::IntrusivePtr` that permits implicit construction
/// from `TaggedTransformPtr<view>` and provides the same interface as
/// `TaggedTransformPtr<view>`.
template <>
class TaggedTransformPtr<container>
    : public internal::IntrusivePtr<
          TransformRep,
          TransformRep::IntrusivePtrTraits<RawTaggedTransformPtr>> {
  using Base = internal::IntrusivePtr<
      TransformRep, TransformRep::IntrusivePtrTraits<RawTaggedTransformPtr>>;

 public:
  using Base::Base;
  TaggedTransformPtr(TaggedTransformPtr<view> other)
      : Base(static_cast<RawTaggedTransformPtr>(other)) {}

  TaggedTransformPtr(TransformRep* p, std::uintptr_t tag)
      : Base(RawTaggedTransformPtr(p, tag)) {}

  TaggedTransformPtr(TransformRep::Ptr<> p, std::uintptr_t tag)
      : Base(RawTaggedTransformPtr(p.release(), tag),
             internal::adopt_object_ref) {}
  std::uintptr_t tag() const { return Base::get().tag(); }
};

/// Friend class of `TransformedArrayBase` that provides access to its private
/// members and defines common utility functions used by the implementation.
class TransformedArrayAccess {
 public:
  template <typename T, typename... U>
  static T Construct(U&&... u) {
    return T(std::forward<U>(u)...);
  }

  /// Internal tag type used to indicate construction from an existing
  /// transformed array.
  struct construct_tag {
    explicit construct_tag() = default;
  };

  /// Internal tag type used to indicate construction from an element pointer
  /// and index transform.
  struct construct_element_pointer_tag {
    explicit construct_element_pointer_tag() = default;
  };

  /// Internal tag type used to indicate construction from a strided array and
  /// index transform.
  struct construct_base_array_transform_tag {
    explicit construct_base_array_transform_tag() = default;
  };

  /// Internal tag type used to indicate construction from a strided array.
  struct construct_array_tag {
    explicit construct_array_tag() = default;
  };

  /// If bit 0 of the tag stored in a TaggedTransformPtr is set, a strided
  /// layout is used.
  constexpr static std::uintptr_t kHasStridedLayoutTagMask = 1;

  /// If bit 1 of the tag stored in a TaggedTransformPtr is set, a zero-origin
  /// (rather than offset-origin) strided layout is used.
  constexpr static std::uintptr_t kZeroOriginTagMask = 2;

  /// Returns the tag value (to be stored in the TaggedTransformPtr) that
  /// indicates that a layout with the specified `ArrayOriginKind` is stored.
  static constexpr std::uintptr_t GetTransformTag(ArrayOriginKind k) {
    return kHasStridedLayoutTagMask |
           (k == zero_origin ? kZeroOriginTagMask : 0);
  }

  /// Implementation used by `TransformTaker` when the transform will be moved
  /// out.
  ///
  /// \tparam L Instance of `LayoutStorage`.
  template <typename L>
  class TransformContainerTaker {
    static_assert(L::layout_container_kind == container, "");

   public:
    using Transform = IndexTransform<L::static_rank>;

    TransformContainerTaker(L& layout)  // NOLINT
        : layout_(layout),
          transform_(TransformAccess::Make<Transform>(TransformRep::Ptr<>(
              layout.transform_.get(), internal::adopt_object_ref))) {
      assert(transform_);
    }

    ~TransformContainerTaker() {
      if (layout_.has_any_strided_layout()) {
        // Destroy the existing "base" layout because a base layout cannot be
        // represented if the transform_ is `nullptr`.
        layout_.layouts_.base_layout.Destroy(layout_.base_layout_origin_kind());
      }
      // Release reference borrowed by `this->transform_`.
      layout_.transform_.release();
      // Construct a default "untransformed" layout in order to put the
      // LayoutStorage object in a valid state once the `transform_` is set to
      // `nullptr`.
      layout_.transform_.reset(
          RawTaggedTransformPtr(
              nullptr,
              GetTransformTag(layout_.layouts_.ConstructUntransformed()
                                  .template Construct<offset_origin>())),
          internal::adopt_object_ref);
    }

    Transform transform() && { return std::move(transform_); }

   private:
    L& layout_;
    /// Borrows the reference owned by `layout_.transform_`.  We can't set
    /// `layout.transform_` to `nullptr` until after the function that created
    /// this `TransformContainerTaker` object has finished accessing
    /// `layout_.base_strided_layout()`.
    Transform transform_;
  };

  /// Implementation used by `TransformTaker` when the transform will be copied.
  ///
  /// \tparam L Instance of `LayoutStorage`.
  template <typename L>
  class TransformViewTaker {
   public:
    using Transform = IndexTransformView<L::static_rank>;

    TransformViewTaker(const L& layout)
        : transform_(
              TransformAccess::Make<Transform>(layout.transform_.get())) {
      assert(transform_);
    }

    Transform transform() const { return transform_; }

   private:
    Transform transform_;
  };

  /// Alias to utility class to use for copying or moving the transform out of a
  /// LayoutStorage object, while still allowing the `base_strided_layout` to be
  /// accessed prior to the destruction of the `TransformTaker` object.
  ///
  /// Moving out the transform is non-trivial because whether the transform is
  /// non-null also indicates which of the two union members,
  /// `untransformed_layout_` or `base_layout_`, is active.
  ///
  /// Whether the transform will be moved out or copied depends on `L` and
  /// `TargetContainerKind`: If `L&&` is a type of the form
  /// `LayoutStorage<Rank, container>&&` and `TargetContainerKind == container`,
  /// then the transform is moved out.  Otherwise, it is copied/referenced.
  ///
  /// \tparam L A cvref-qualified LayoutStorage instance type, e.g.:
  ///     `const LayoutStorage<3, view>&` or `LayoutStorage<3, container>&&`.
  /// \tparam TargetContainerKind Specifies the container kind of the new layout
  ///     to be constructed from the existing layout.
  template <typename L, ContainerKind TargetContainerKind,
            typename LX = internal::remove_cvref_t<L>>
  using TransformTaker = std::conditional_t<
      (std::is_same<L, internal::remove_cvref_t<L>&&>::value &&
       internal::remove_cvref_t<L>::layout_container_kind == container &&
       TargetContainerKind == container),
      TransformContainerTaker<LX>, TransformViewTaker<LX>>;

  /// Returns an IndexTransform that combines both `layout.transform()` and
  /// `layout.base_strided_layout()`.
  ///
  /// If `L&&` is a type of the form
  /// ``LayoutStorage<Rank ,container>&&` then the transform will be
  /// moved out and modified.  Otherwise, it will be copied.
  ///
  /// \dchecks `layout.has_base_array()`.
  template <typename L>
  static Result<IndexTransform<internal::remove_cvref_t<L>::static_rank>>
  TakeTransformWithBaseLayout(L&& layout) {
    assert(layout.has_any_strided_layout() && layout.transform_);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto transform_ptr,
        MakeTransformFromStridedLayoutAndTransform(
            layout.base_strided_layout(),
            TransformAccess::rep_ptr<container>(
                TransformTaker<L&&, container>(layout).transform())));
    return TransformAccess::Make<
        IndexTransform<internal::remove_cvref_t<L>::static_rank, dynamic_rank>>(
        std::move(transform_ptr));
  }

  /// Type alias that evaluates to the result of calling MapTransform with a
  /// given function type.
  template <typename A, typename Func>
  using MapTransformResultType = FlatMapResultType<
      A::template RebindTransform,
      internal::remove_cvref_t<std::invoke_result_t<
          Func, const typename internal::remove_cvref_t<A>::Transform&>>>;

  /// Returns a new Result-wrapped NormalizedTransformedArray where the index
  /// transform has been mapped by the specified function.
  ///
  /// The first argument is used for tag dispatching and is `std::true_type` if
  /// `A` is a `NormalizedTransformedArray`, or `std::false_type` if `A` is a
  /// `TransformedArray`.
  ///
  /// This overload handles the case of `A` being a `TransformedArray`.
  template <typename A, typename Func>
  static MapTransformResultType<internal::remove_cvref_t<A>, Func> MapTransform(
      std::false_type normalized, A&& a, Func&& func) {
    using ResultType =
        MapTransformResultType<internal::remove_cvref_t<A>, Func>;
    using AX = internal::remove_cvref_t<A>;
    using MappedTransform = UnwrapResultType<
        std::invoke_result_t<Func, const typename AX::Transform&>>;
    const auto transform_to_transformed_array = [&](MappedTransform transform) {
      return typename ResultType::value_type{
          std::forward<A>(a).element_pointer_, std::move(transform)};
    };
    if (a.layout_.transform_) {
      if (a.layout_.has_any_strided_layout()) {
        return ChainResult(
            TakeTransformWithBaseLayout(std::forward<A>(a).layout_),
            std::forward<Func>(func), transform_to_transformed_array);
      } else {
        return ChainResult(
            std::forward<Func>(func)(
                TransformTaker<
                    internal::CopyQualifiers<A&&, typename AX::LayoutStorage>,
                    container>(a.layout_)
                    .transform()),
            transform_to_transformed_array);
      }
    } else {
      assert(a.layout_.has_any_strided_layout());
      return ChainResult(std::forward<Func>(func)(
                             TransformAccess::Make<
                                 IndexTransform<AX::static_rank, dynamic_rank>>(
                                 MakeTransformFromStridedLayout(
                                     a.untransformed_strided_layout()))),
                         transform_to_transformed_array);
    }
  }

  /// Overload that handles the case of `A` being a
  /// `NormalizedTransformedArray`.
  template <typename A, typename Func>
  static MapTransformResultType<internal::remove_cvref_t<A>, Func> MapTransform(
      std::true_type normalized, A&& a, Func&& func) {
    using ResultType =
        MapTransformResultType<internal::remove_cvref_t<A>, Func>;
    using AX = internal::remove_cvref_t<A>;
    using MappedTransform = UnwrapResultType<
        std::invoke_result_t<Func, const typename AX::Transform&>>;
    return ChainResult(std::forward<Func>(func)(std::forward<A>(a).transform()),
                       [&](MappedTransform transform) {
                         return typename ResultType::value_type{
                             std::forward<A>(a).element_pointer(),
                             std::move(transform)};
                       });
  }

  template <typename NewTransform>
  struct ConvertFunc {
    template <typename X>
    NewTransform operator()(X&& x) const {
      return NewTransform(std::forward<X>(x));
    }
  };

  /// Returns a transformed array normalized to the (element_pointer, transform)
  /// representation.
  ///
  /// This is used to implement `MakeNormalizedTransformedArray` for the case of
  /// a `TransformedArray` argument.
  ///
  /// \tparam A Instance of `TransformedArray` (must not be an instance of
  ///     `NormalizedTransformedArray`).
  template <typename A>
  static Result<
      typename internal::remove_cvref_t<A>::template RebindTransform<>>
  NormalizeTransform(A&& a) {
    return MapTransform(/*normalized=*/std::false_type{}, std::forward<A>(a),
                        ConvertFunc<typename internal::remove_cvref_t<
                            A>::template RebindTransform<>::Transform>{});
  }

  /// Union of `MemberType<zero_origin>` and `MemberType<offset_origin>`.
  ///
  /// This is used by LayoutStorage to store the untransformed and base layouts.
  /// The user of this class is responsible for keeping track of the active
  /// member.
  ///
  /// \tparam MemberType Type alias that maps an `ArrayOriginKind` to the
  ///     corresponding member type to store.
  /// \tparam MapLayoutOriginKind Type alias that maps an `ArrayOriginKind` to
  ///     an `std::integral_constant<ArrayOriginKind, ...>` value.  This should
  ///     either be an identity transform, in which case both members are used,
  ///     or always map to `offset_origin`, in which case only the
  ///     `offset_origin_` member is used.
  template <template <ArrayOriginKind> class MemberType,
            template <ArrayOriginKind> class MapLayoutOriginKind>
  union OriginDependentUnion {
    OriginDependentUnion() noexcept {}
    ~OriginDependentUnion() {}

    MemberType<zero_origin>& get(
        std::integral_constant<ArrayOriginKind, zero_origin>) {
      return zero_origin_;
    }
    MemberType<offset_origin>& get(
        std::integral_constant<ArrayOriginKind, offset_origin>) {
      return offset_origin_;
    }

    /// Constructs a layout of type `MapLayoutOriginKind<SourceKind>`.
    ///
    /// \returns The kind of layout stored, equal to
    ///     `MapLayoutOriginKind<SourceKind>:;value`.
    template <ArrayOriginKind SourceKind, typename... Args>
    ArrayOriginKind Construct(Args&&... args) noexcept {
      using K = MapLayoutOriginKind<SourceKind>;
      new (&get(K{})) MemberType<K::value>(static_cast<Args&&>(args)...);
      return K::value;
    }

    /// Copy or move constructs a contained layout from `other`, which must
    /// store a layout of kind `source_kind`.
    ///
    /// \returns The kind of layout stored.
    template <typename Other>
    ArrayOriginKind CopyConstruct(Other&& other,
                                  ArrayOriginKind other_kind) noexcept {
      if (other_kind == zero_origin) {
        return Construct<zero_origin>(unchecked,
                                      std::forward<Other>(other).zero_origin_);
      } else {
        return Construct<offset_origin>(
            unchecked, std::forward<Other>(other).offset_origin_);
      }
    }

    /// Destroys the stored strided layout, which must be of kind `stored_kind`.
    ///
    /// Every call to `Construct` or `CopyConstruct` must be followed by a call
    /// to `Destroy` with the returned layout kind.
    void Destroy(ArrayOriginKind stored_kind) noexcept {
      if (stored_kind == zero_origin) {
        using K = MapLayoutOriginKind<zero_origin>;
        std::destroy_at(&get(K{}));
      } else {
        using K = MapLayoutOriginKind<offset_origin>;
        std::destroy_at(&get(K{}));
      }
    }

    MemberType<zero_origin> zero_origin_;
    MemberType<offset_origin> offset_origin_;
  };

  /// Accessor for the private `layout_` member of `TransformedArray`.
  template <typename T>
  static auto layout(T&& x) -> decltype((std::declval<T>().layout_)) {
    return std::forward<T>(x).layout_;
  }

  /// Manages the storage of the transformed array layout representation.
  ///
  /// The layout is represented using a `transform_` pointer to a TransformRep
  /// (or `nullptr` if the layout is not represented using a transform), and a
  /// tagged union in one of 5 possible states:
  ///
  /// 1. `untransformed_layout_` (only allowed if `transform_ == nullptr`),
  ///     represented as a tagged union of:
  ///
  ///    1a[tag=0b01] `StridedLayout<Rank, offset_origin, LayoutCKind>`
  ///
  ///    1b[tag=0b11] `StridedLayout<Rank, zero_origin, LayoutCKind>` (only used
  ///        if `Rank == dynamic_rank` and `LayoutCKind == container`)
  ///
  /// 2. `base_layout_` (only allowed if `transform_ != nullptr`),
  ///    represented as a tagged union of:
  ///
  ///     2a[tag=0b01] `StridedLayout<dynamic_rank, offset_origin, LayoutCKind>`
  ///     2b[tag=0b11] `StridedLayout<dynamic_rank, zero_origin, LayoutCKind>
  ///         (only used if `LayoutCKind == container`).
  ///
  /// 3. [tag=0b00] No strided layout (only allowed if `transform_ != nullptr`).
  ///
  /// In the case of a dynamic rank and a `layout_container_kind` of
  /// `container`, both `offset_origin` and `zero_origin` layouts are supported
  /// in order to avoid the need to convert `zero_origin` layouts to an
  /// `offset_origin` representation, which would require a heap allocation.
  /// However, in the case of a `view` container kind or a static rank (for
  /// `untransformed_layout_` only), conversion to an `offset_origin`
  /// representation is cheap and therefore we do not support the `zero_origin`
  /// representation in order to reduce the amount of branching required in the
  /// generated code.
  ///
  /// The discriminant for this tagged union representation of the
  /// `untransformed_layout_` or `base_layout_` is stored in the low 2 bits of
  /// the `transform_` pointer (using the `TaggedPtr` class template).  The
  /// stored tag values for each state are indicated above.  Bit 0 indicates if
  /// a strided layout is stored at all, while bit 1 indicates that a
  /// `zero_origin` representation is used.  Note that identical tag values are
  /// used for 1a/2a and 1b/2b; these states are distinguished based on whether
  /// `transform_` is `nullptr`.
  template <DimensionIndex Rank, ContainerKind LayoutCKind>
  struct LayoutStorage {
    constexpr static DimensionIndex static_rank = Rank;
    constexpr static ContainerKind layout_container_kind = LayoutCKind;
    using RankType = StaticOrDynamicRank<Rank>;
    static constexpr bool kUseZeroOriginUntransformedLayout =
        LayoutCKind != view && Rank == dynamic_rank;
    static constexpr bool kUseZeroOriginBaseLayout = LayoutCKind != view;

    using Transform = IndexTransform<Rank, dynamic_rank, LayoutCKind>;

    // All constructors are marked noexcept to terminate in case of bad_alloc,
    // which is currently not handled correctly.  This also avoids link errors
    // in case of mixed -fexceptions/-fno-exceptions builds.

    LayoutStorage() noexcept : transform_(RawTaggedTransformPtr(nullptr, 1)) {
      layouts_.ConstructUntransformed().template Construct<offset_origin>();
    }

    template <typename L>
    explicit LayoutStorage(construct_array_tag, L&& layout) noexcept
        : transform_(RawTaggedTransformPtr(
              nullptr,
              GetTransformTag(
                  layouts_.ConstructUntransformed()
                      .template Construct<
                          internal::remove_cvref_t<L>::array_origin_kind>(
                          unchecked, std::forward<L>(layout))))) {}

    template <typename T>
    explicit LayoutStorage(construct_element_pointer_tag,
                           T&& transform) noexcept
        : transform_(
              TransformAccess::rep_ptr<LayoutCKind>(std::forward<T>(transform)),
              0) {
      assert(transform_);
    }

    template <typename L, typename T>
    explicit LayoutStorage(construct_base_array_transform_tag, L&& layout,
                           T&& transform) noexcept
        : transform_(
              (ABSL_ASSERT(transform.output_rank() == layout.rank()),
               TransformAccess::rep_ptr<LayoutCKind>(
                   std::forward<T>(transform))),
              GetTransformTag(
                  layouts_.ConstructBase()
                      .template Construct<
                          internal::remove_cvref_t<L>::array_origin_kind>(
                          unchecked, std::forward<L>(layout)))) {}

    template <typename Other>
    explicit LayoutStorage(construct_tag, Other&& other) noexcept {
      if (other.transform_) {
        std::uintptr_t tag = 0;
        if (other.has_any_strided_layout()) {
          tag = GetTransformTag(layouts_.ConstructBase().CopyConstruct(
              std::forward<Other>(other).layouts_.base_layout,
              other.base_layout_origin_kind()));
        }
        transform_ = TaggedTransformPtr<LayoutCKind>(
            TransformAccess::rep_ptr<LayoutCKind>(
                TransformTaker<Other&&, LayoutCKind>(other).transform()),
            tag);
      } else {
        assert(other.has_any_strided_layout());
        transform_ = TaggedTransformPtr<LayoutCKind>(
            nullptr,
            GetTransformTag(layouts_.ConstructUntransformed().CopyConstruct(
                std::forward<Other>(other).layouts_.untransformed_layout,
                other.untransformed_layout_origin_kind())));
      }
    }

    LayoutStorage(const LayoutStorage& other) noexcept
        : LayoutStorage(construct_tag{}, other) {}

    LayoutStorage(LayoutStorage&& other) noexcept
        : LayoutStorage(construct_tag{}, std::move(other)) {}

    LayoutStorage& operator=(const LayoutStorage& other) noexcept {
      // TODO(jbms): handle exceptions
      std::destroy_at(this);
      new (this) LayoutStorage(other);
      return *this;
    }

    LayoutStorage& operator=(LayoutStorage&& other) noexcept {
      // TODO(jbms): handle exceptions
      std::destroy_at(this);
      new (this) LayoutStorage(std::move(other));
      return *this;
    }

    RankType rank() const {
      if (transform_) {
        return StaticRankCast<Rank, unchecked>(
            static_cast<DimensionIndex>(transform_->input_rank));
      }
      assert(has_any_strided_layout());
      if (untransformed_layout_origin_kind() == offset_origin) {
        return layouts_.untransformed_layout.offset_origin_.rank();
      } else {
        return layouts_.untransformed_layout.zero_origin_.rank();
      }
    }

    span<const std::string, Rank> labels() const {
      if (transform_) {
        return transform().input_labels();
      }
      return GetDefaultStringVector(rank());
    }

    BoxView<Rank> domain() const {
      if (transform_) {
        return StaticRankCast<Rank, unchecked>(
            transform_->input_domain(transform_->input_rank));
      }
      assert(has_any_strided_layout());
      if (untransformed_layout_origin_kind() == offset_origin) {
        return layouts_.untransformed_layout.offset_origin_.domain();
      } else {
        return layouts_.untransformed_layout.zero_origin_.domain();
      }
    }

    bool has_untransformed_array() const {
      assert(transform_ || has_any_strided_layout());
      return !transform_;
    }

    bool has_base_array() const {
      return transform_ && has_any_strided_layout();
    }

    IndexTransformView<Rank, dynamic_rank> transform() const {
      return TransformAccess::Make<IndexTransformView<Rank, dynamic_rank>>(
          transform_.get());
    }

    StridedLayoutView<Rank, offset_origin> untransformed_strided_layout()
        const {
      assert(has_untransformed_array());
      if (untransformed_layout_origin_kind() == offset_origin) {
        return layouts_.untransformed_layout.offset_origin_;
      }
      return layouts_.untransformed_layout.zero_origin_;
    }

    StridedLayoutView<dynamic_rank, offset_origin> base_strided_layout() const {
      assert(has_base_array());
      if (base_layout_origin_kind() == offset_origin) {
        return layouts_.base_layout.offset_origin_;
      }
      return layouts_.base_layout.zero_origin_;
    }

    StridedLayoutView<dynamic_rank, offset_origin>
    base_or_untransformed_strided_layout() const {
      if (transform_) {
        return has_base_array() ? base_strided_layout()
                                : GetUnboundedLayout(transform_->output_rank);
      } else {
        return untransformed_strided_layout();
      }
    }

    ~LayoutStorage() {
      if (transform_) {
        if (has_any_strided_layout()) {
          layouts_.base_layout.Destroy(base_layout_origin_kind());
        }
      } else {
        assert(has_any_strided_layout());
        layouts_.untransformed_layout.Destroy(
            untransformed_layout_origin_kind());
      }
    }

    bool has_any_strided_layout() const {
      return transform_.tag() & kHasStridedLayoutTagMask;
    }

    ArrayOriginKind untransformed_layout_origin_kind() const {
      return kUseZeroOriginUntransformedLayout &&
                     (transform_.tag() & kZeroOriginTagMask)
                 ? zero_origin
                 : offset_origin;
    }

    ArrayOriginKind base_layout_origin_kind() const {
      return kUseZeroOriginBaseLayout && (transform_.tag() & kZeroOriginTagMask)
                 ? zero_origin
                 : offset_origin;
    }

    template <ArrayOriginKind OriginKind>
    using UntransformedLayoutOriginKind =
        std::integral_constant<ArrayOriginKind,
                               (OriginKind == zero_origin &&
                                kUseZeroOriginUntransformedLayout)
                                   ? zero_origin
                                   : offset_origin>;

    template <ArrayOriginKind OriginKind>
    using BaseLayoutOriginKind =
        std::integral_constant<ArrayOriginKind, (OriginKind == zero_origin &&
                                                 kUseZeroOriginBaseLayout)
                                                    ? zero_origin
                                                    : offset_origin>;

    template <ArrayOriginKind OriginKind>
    using UntransformedLayout = StridedLayout<Rank, OriginKind, LayoutCKind>;
    template <ArrayOriginKind OriginKind>
    using BaseLayout = StridedLayout<dynamic_rank, OriginKind, LayoutCKind>;

    union Layouts {
      Layouts() noexcept {}
      ~Layouts() {}

      auto& ConstructBase() noexcept {
        new (&base_layout)
            OriginDependentUnion<BaseLayout, BaseLayoutOriginKind>;
        return base_layout;
      }

      auto& ConstructUntransformed() noexcept {
        new (&untransformed_layout)
            OriginDependentUnion<UntransformedLayout,
                                 UntransformedLayoutOriginKind>;
        return untransformed_layout;
      }

      OriginDependentUnion<UntransformedLayout, UntransformedLayoutOriginKind>
          untransformed_layout;
      OriginDependentUnion<BaseLayout, BaseLayoutOriginKind> base_layout;
    };
    Layouts layouts_;
    TaggedTransformPtr<LayoutCKind> transform_;
  };
};

// Used to implement `EnableIfTransformedArrayMapTransformResultType` below.
template <bool Condition>
struct ConditionalTransformedArrayMapTransformResultType {
  template <typename A, typename Func>
  using type = TransformedArrayAccess::MapTransformResultType<A, Func>;
};

template <>
struct ConditionalTransformedArrayMapTransformResultType<false> {};

/// Equivalent to:
///
///     std::enable_if_t<
///         Condition,
///         TransformedArrayAccess::MapTransformResultType<A, Func>>
///
/// except that `TransformedArrayAccess::MapTransformResultType<A, Func>` is not
/// evaluated if `Condition` is `false` (this avoids the potential for SFINAE
/// loops).
template <bool Condition, typename A, typename Func>
using EnableIfTransformedArrayMapTransformResultType =
    typename ConditionalTransformedArrayMapTransformResultType<
        Condition>::template type<A, Func>;

std::string DescribeTransformedArrayForCast(DataType data_type,
                                            DimensionIndex rank);

/// Base class providing common implementation of the `StaticCastTraits`
/// specializations for `TransformedArray` and `NormalizedTransformedArray`.
template <template <typename ElementTagType, DimensionIndex Rank,
                    ContainerKind LayoutContainerKind>
          class ArrayTemplate,
          typename ElementTagType, DimensionIndex Rank,
          ContainerKind LayoutContainerKind>
struct TransformedArrayCastTraits
    : public DefaultStaticCastTraits<
          ArrayTemplate<ElementTagType, Rank, LayoutContainerKind>> {
  using type = ArrayTemplate<ElementTagType, Rank, LayoutContainerKind>;

  template <typename TargetElement>
  using RebindDataType = ArrayTemplate<
      typename ElementTagTraits<ElementTagType>::template rebind<TargetElement>,
      Rank, LayoutContainerKind>;

  template <DimensionIndex TargetRank>
  using RebindRank =
      ArrayTemplate<ElementTagType, TargetRank, LayoutContainerKind>;

  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return IsRankExplicitlyConvertible(other.rank(), Rank) &&
           IsPossiblySameDataType(other.data_type(), typename type::DataType());
  }

  static std::string Describe() {
    return internal_index_space::DescribeTransformedArrayForCast(
        typename type::DataType(), Rank);
  }

  static std::string Describe(const type& value) {
    return internal_index_space::DescribeTransformedArrayForCast(
        value.data_type(), value.rank());
  }
};

}  // namespace internal_index_space
}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_INTERNAL_TRANSFORMED_ARRAY_IMPL_H_
