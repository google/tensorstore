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

/// \file Facilities for binding and unbinding context resources within nested
///     structures, using minimal boilerplate.
///
/// Components within TensorStore such as `KeyValueStore` involve a `Spec` type
/// that is loaded from a JSON representation and whose underlying
/// representation may have various `Context::Resource` members, possibly
/// contained within nested structures.  To actually open a driver based on the
/// `Spec`, the unbound context resource specs obtained from the JSON
/// representation must be bound to actual context resources.
///
/// Example usage:
///
///     struct Nested {
///       std::string value;
///       Context::Resource<Baz> baz;
///
///       constexpr static auto ApplyMembers = [](auto &x, auto f) {
///         return f(x.baz);
///       };
///     };
///
///     struct SpecData {
///       int value;
///       Context::Resource<Foo> foo;
///       Context::Resource<Bar> bar;
///       Nested nested;
///
///       constexpr static auto ApplyMembers = [](auto &x, auto f) {
///         return f(x.foo, x.bar, x.nested);
///       };
///     };
///
/// For reference, equivalent `ContextBindingTraits` definitions that don't rely
/// on `ApplyMembers`:
///
///     template <>
///     struct ContextBindingTraits<Nested> {
///       static absl::Status Bind(Nested& spec, const Context& context) {
///         TENSORSTORE_RETURN_IF_ERROR(spec.baz.Bind(context));
///         return absl::OkStatus();
///       }
///
///       static void Unbind(Nested& spec, const ContextSpecBuilder& builder) {
///         spec.baz.Unbind(builder);
///       }
///
///       static void Strip(Nested& spec) {
///         spec.baz.Strip();
///       }
///     };
///
///     template <>
///     struct ContextBindingTraits<SpecData> {
///       static absl::Status Bind(SpecData& spec, const Context& context) {
///         TENSORSTORE_RETURN_IF_ERROR(spec.foo.Bind(context));
///         TENSORSTORE_RETURN_IF_ERROR(spec.bar.Bind(context));
///         TENSORSTORE_RETURN_IF_ERROR(
///             ContextBindingTraits<Nested>::Bind(spec.nested, context));
///         return absl::OkStatus();
///       }
///
///       static void Unbind(SpecData& spec,
///                          const ContextSpecBuilder& builder) {
///         spec.foo.Unbind(builder);
///         spec.bar.Unbind(builder);
///         ContextBindingTraits<Nested>::Unbind(spec.nested, builder);
///       }
///
///       static void Strip(SpecData& spec) {
///         spec.foo.Strip();
///         spec.bar.Strip();
///         ContextBindingTraits<Nested>::Strip(spec.nested);
///       }
///     };

#include <optional>
#include <type_traits>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/util/apply_members/apply_members.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

template <typename Spec>
struct NoOpContextBindingTraits {
  /// Resolves context resources in `spec` using `context.
  static absl::Status Bind(Spec& spec, const Context& context) {
    return absl::OkStatus();
  }

  /// Unbinds context resources in `spec`, using `builder`.
  static void Unbind(Spec& spec, const ContextSpecBuilder& builder) {}

  /// Resets context resources in `spec` to default resource specs.
  static void Strip(Spec& spec) {}
};

/// Traits type that may be specialized for a given `Spec` type to define
/// context binding/unbinding operations.
///
/// This default definition handles the trivial case where the `Spec` type
/// contains no context resources.
template <typename Spec, typename SFINAE = void>
struct ContextBindingTraits : public NoOpContextBindingTraits<Spec> {};

/// Specialization of `ContextBindingTraits` for `Context::Resource`.
template <typename Provider>
struct ContextBindingTraits<Context::Resource<Provider>> {
  using Spec = Context::Resource<Provider>;
  static absl::Status Bind(Spec& spec, const Context& context) {
    return spec.BindContext(context);
  }
  static void Unbind(Spec& spec, const ContextSpecBuilder& builder) {
    spec.UnbindContext(builder);
  }
  static void Strip(Spec& spec) { spec.StripContext(); }
};

/// Specialization of `ContextBindingTraits` for classes with nested static
/// `ApplyMembers` method.
///
/// Refer to example above.
template <class Spec>
struct ContextBindingTraits<Spec,
                            std::enable_if_t<SupportsApplyMembers<Spec>>> {
  static absl::Status Bind(Spec& spec, const Context& context) {
    return ApplyMembers<Spec>::Apply(spec, [&context](auto&&... spec_member) {
      absl::Status status;
      (void)((status = ContextBindingTraits<
                  remove_cvref_t<decltype(spec_member)>>::Bind(spec_member,
                                                               context))
                 .ok() &&
             ...);
      return status;
    });
  }

  static void Unbind(Spec& spec, const ContextSpecBuilder& builder) {
    ApplyMembers<Spec>::Apply(spec, [&builder](auto&&... spec_member) {
      (ContextBindingTraits<remove_cvref_t<decltype(spec_member)>>::Unbind(
           spec_member, builder),
       ...);
    });
  }

  static void Strip(Spec& spec) {
    ApplyMembers<Spec>::Apply(spec, [](auto&&... spec_member) {
      (ContextBindingTraits<remove_cvref_t<decltype(spec_member)>>::Strip(
           spec_member),
       ...);
    });
  }
};

/// Specialization of `ContextBindingTraits` for std::optional<T>.
template <typename Spec>
struct ContextBindingTraits<std::optional<Spec>> {
  using BaseTraits = ContextBindingTraits<Spec>;

  static absl::Status Bind(std::optional<Spec>& spec, const Context& context) {
    if (spec.has_value()) {
      TENSORSTORE_RETURN_IF_ERROR(BaseTraits::Bind(spec.value(), context));
    }
    return absl::OkStatus();
  }

  static void Unbind(std::optional<Spec>& spec,
                     const ContextSpecBuilder& builder) {
    if (spec.has_value()) {
      BaseTraits::Unbind(spec.value(), builder);
    }
  }

  static void Strip(std::optional<Spec>& spec) {
    if (spec.has_value()) {
      BaseTraits::Strip(spec.value());
    }
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONTEXT_BINDING_H_
