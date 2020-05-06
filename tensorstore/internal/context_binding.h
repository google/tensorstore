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

#ifndef TENSORSTORE_INTERNAL_CONTEXT_BINDING_H_
#define TENSORSTORE_INTERNAL_CONTEXT_BINDING_H_

/// \file Facilities for converting between `Context::ResourceSpec<T>` and
/// `Context::Resource<T>` within nested structures, using minimal boilerplate.
///
/// Components within TensorStore such as `KeyValueStore` involve a `Spec` type
/// that is loaded from a JSON representation and whose underlying
/// representation may have various `Context::ResourceSpec` members, possibly
/// contained within nested structures.  To actually open a driver based on the
/// `Spec`, `Context::Resource` objects must be obtained from all of the
/// `Context::ResourceSpec` members; typically there is a `BoundSpec` type that
/// is isomorphic to the `Spec` type but where all of the
/// `Context::ResourceSpec` members are converted to `Context::Resource`
/// members.
///
/// The templates in this file provide a way to accomplish this without the need
/// for duplicate structure definitions and repetitive code.  Typically the
/// `Spec` and `BoundSpec` types are both defined as a single `SpecT` template
/// with a nested static `ApplyMembers` polymorphic lambda as shown below.
///
/// Example usage:
///
///     template <template <typename T> class MaybeBound>
///     struct NestedT {
///       std::string value;
///       MaybeBound<Context::ResourceSpec<Baz>> baz;
///
///       constexpr static auto ApplyMembers = [](auto &x, auto f) {
///         return f(x.value, x.baz);
///       };
///     };
///
///     template <template <typename T> class MaybeBound>
///     struct SpecT {
///       int value;
///       MaybeBound<Context::ResourceSpec<Foo>> foo;
///       MaybeBound<Context::ResourceSpec<Bar>> bar;
///       NestedT<MaybeBound> nested;
///
///       constexpr static auto ApplyMembers = [](auto &x, auto f) {
///         return f(x.value, x.foo, x.bar, x.nested);
///       };
///     };
///
/// For reference, equivalent `ContextBindingTraits` definitions that don't rely
/// on `ApplyMembers`:
///
///     template <>
///     struct ContextBindingTraits<NestedT<ContextUnbound>> {
///       using Spec = NestedT<ContextUnbound>;
///       using Bound = NestedT<ContextBound>;
///       static Status Bind(const Spec* spec, Bound* bound,
///                          const Context& context) {
///         bound->value = spec->value;
///         TENSORSTORE_ASSIGN_OR_RETURN(bound->baz,
///                                      context.GetResource(spec->baz));
///         return Status();
///       }
///
///       static void Unbind(Spec* spec, const Bound* bound,
///                          const ContextSpecBuilder& builder) {
///         spec->value = bound->value;
///         spec->baz = builder.AddResource(bound->baz);
///       }
///     };
///
///     template <>
///     struct ContextBindingTraits<SpecT<ContextUnbound>> {
///       using Spec = SpecT<ContextUnbound>;
///       using Bound = SpecT<ContextBound>;
///       static Status Bind(const Spec* spec, Bound* bound,
///                          const Context& context) {
///         spec->value = bound->value;
///         TENSORSTORE_ASSIGN_OR_RETURN(bound->foo,
///                                      context.GetResource(spec->foo));
///         TENSORSTORE_ASSIGN_OR_RETURN(bound->bar,
///                                      context.GetResource(spec->bar));
///         TENSORSTORE_RETURN_IF_ERROR(
///             ContextBindingTraits<NestedT<ContextUnbound>>::Bind(
///                 &spec->nested, &bound->nested, context));
///         return Status();
///       }
///
///       static void Unbind(Spec* spec, const Bound* bound,
///                          const ContextSpecBuilder& builder) {
///         spec->value = bound->value;
///         spec->foo = builder.AddResource(bound->foo);
///         spec->bar = builder.AddResource(bound->bar);
///         ContextBindingTraits<NestedT<ContextUnbound>>::Unbind(
///             &spec->nested, &bound->nested, builder);
///       }
///     };

#include <type_traits>

#include "tensorstore/context.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

// clang: requires use when explicitly capturing a parameter pack
template <typename... T>
void SuppressMaybeUnusedWarning(T...) {}

/// Traits type that may be specialized for a given `Spec` type to define the
/// conversion to the corresponding `Bound` type.
///
/// This default definition handles the trivial case where the `Spec` and
/// `Bound` types are the same.
template <typename Spec, typename SFINAE = void>
struct ContextBindingTraits {
  /// Specifies the corresponding `Bound` type (in this case the same as
  /// `Spec`).
  using Bound = Spec;

  /// Assigns `*bound` to the "bound" representation of `*spec` using `context.
  static Status Bind(const Spec* spec, Bound* bound, const Context& context) {
    *bound = *spec;
    return Status();
  }

  /// Assigns `*spec` to the "unbound" representation of `*bound` using
  /// `builder`.
  static void Unbind(Spec* spec, const Bound* bound,
                     const ContextSpecBuilder& builder) {
    *spec = *bound;
  }
};

/// Specialization of `ContextBindingTraits` for `Context::ResourceSpec`.
template <typename Provider>
struct ContextBindingTraits<Context::ResourceSpec<Provider>> {
  using Spec = Context::ResourceSpec<Provider>;
  using Bound = Context::Resource<Provider>;
  static Status Bind(const Spec* spec, Bound* bound, const Context& context) {
    TENSORSTORE_ASSIGN_OR_RETURN(*bound, context.GetResource(*spec));
    return Status();
  }
  static void Unbind(Spec* spec, const Bound* bound,
                     const ContextSpecBuilder& builder) {
    *spec = builder.AddResource(*bound);
  }
};

/// Alias that maps a `Spec` type to the corresponding `Bound` type using
/// `ContextBindingTraits`.  This is intended for use as a template argument to
/// `SpecT` templates as in the example above.
template <typename Spec>
using ContextBound = typename ContextBindingTraits<Spec>::Bound;

/// Identity alias, intended for use as a template argument to `SpecT` templates
/// as in the example above.
template <typename Spec>
using ContextUnbound = Spec;

/// Specialization of `ContextBindingTraits` for `SpecT` class templates with
/// nested static `ApplyMembers` method.
///
/// Refer to example above.
template <template <template <typename> class MaybeBound> class SpecT>
struct ContextBindingTraits<
    SpecT<ContextUnbound>,
    std::void_t<decltype(SpecT<ContextBound>::ApplyMembers)>> {
  using Spec = SpecT<ContextUnbound>;
  using Bound = SpecT<ContextBound>;
  static Status Bind(const Spec* spec, Bound* bound, const Context& context) {
    return Spec::ApplyMembers(*spec, [&](const auto&... spec_member) {
      // Use explicit capture list to work around Clang bug
      // https://bugs.llvm.org/show_bug.cgi?id=42104
      return Bound::ApplyMembers(*bound, [&spec_member...,
                                          &context](auto&... bound_member) {
        SuppressMaybeUnusedWarning(spec_member...);
        Status status;
        (void)((status = ContextBindingTraits<
                    remove_cvref_t<decltype(spec_member)>>::Bind(&spec_member,
                                                                 &bound_member,
                                                                 context))
                   .ok() &&
               ...);
        return status;
      });
    });
  }

  static void Unbind(Spec* spec, const Bound* bound,
                     const ContextSpecBuilder& builder) {
    Spec::ApplyMembers(*spec, [&](auto&... spec_member) {
      // Use explicit capture list to work around Clang bug
      // https://bugs.llvm.org/show_bug.cgi?id=42104
      Bound::ApplyMembers(*bound, [&spec_member...,
                                   &builder](const auto&... bound_member) {
        SuppressMaybeUnusedWarning(spec_member...);
        (ContextBindingTraits<remove_cvref_t<decltype(spec_member)>>::Unbind(
             &spec_member, &bound_member, builder),
         ...);
      });
    });
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONTEXT_BINDING_H_
