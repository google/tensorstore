// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_UTIL_GARBAGE_COLLECTION_GARBAGE_COLLECTION_H_
#define TENSORSTORE_UTIL_GARBAGE_COLLECTION_GARBAGE_COLLECTION_H_

/// \file
///
/// Enables the TensorStore library to interoperate with the Python garbage
/// collector.
///
/// For each C++ type that can potentially (transitively) reference a
/// garbage-collected object, garbage collection support must be implemented in
/// one of several ways:
///
/// 1. For simple structs, it is usually sufficient to implement the
///    `ApplyMembers` protocol.  All specified members must also support garbage
///    collection.
///
/// 2. For more complex types `T`, you can specialize
///    `tensorstore::garbage_collection::GarbageCollection<T>` directly.  For
///    non-templated types, the
///    `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION` and
///    `TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION` convenience macros
///    can be used as well.
///
/// 3. For types that cannot reference any garbage-collected objects, you can
///    specialize `tensorstore::garbage_collection::GarbageCollection<T>` to
///    inherit from
///    `tensorstore::garbage_collection::GarbageCollectionNotRequired`.
///    Alternatively, you can use the macro
///    `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED`.
///
/// 4. For polymorphic base classes, you should manually define a pure virtual
///    `GarbageCollectionVisit` method to be overridden by the derived classes.
///    The `tensorstore::garbage_collection::GarbageCollection` specialization
///    for the base class can be defined by inheriting from
///    `PolymorphicGarbageCollection`.

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/apply_members/apply_members.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace garbage_collection {

/// Indicates whether `T` potentially requires garbage collection.
///
/// Types are assumed to require garbage collection unless they define a
/// `GarbageCollection` specialization with a
/// `constexpr static bool required() { return false; }` method.
///
/// For types that do require garbage collection, it is sufficient to specialize
/// `GarbageCollection` and define a `Visit` method.  It is not necessary to
/// define a `constexpr static bool required() { return true; }` method.
///
/// Defaults to `true` in order to ensure a hard error for types that don't
/// explicitly opt out.
template <typename T, typename SFINAE = void>
constexpr inline bool IsGarbageCollectionRequired = true;

template <typename T>
constexpr inline bool IsGarbageCollectionRequired<
    T, std::void_t<decltype(&GarbageCollection<T>::required)>> =
    GarbageCollection<T>::required();

/// Convenience type that may be used to define a `GarbageCollection`
/// specialization for a type that cannot reference any garbage-collected
/// objects.
struct GarbageCollectionNotRequired {
  constexpr static bool required() { return false; }
};

/// Garbage collection is never required for trivial types.
template <typename T>
struct GarbageCollection<T,
                         std::enable_if_t<std::is_trivially_destructible_v<T>>>
    : public GarbageCollectionNotRequired {};

template <>
struct GarbageCollection<std::string> : public GarbageCollectionNotRequired {};

template <>
struct GarbageCollection<absl::Cord> : public GarbageCollectionNotRequired {};

template <>
struct GarbageCollection<absl::Time> : public GarbageCollectionNotRequired {};

template <>
struct GarbageCollection<absl::Duration> : public GarbageCollectionNotRequired {
};

template <>
struct GarbageCollection<absl::Status> : public GarbageCollectionNotRequired {};

/// Calls `visitor` with each potentially-garbage-collected object transitively
/// reachable from `value`.
template <typename T>
void GarbageCollectionVisit(GarbageCollectionVisitor& visitor, const T& value) {
  if constexpr (IsGarbageCollectionRequired<T>) {
    GarbageCollection<T>::Visit(visitor, value);
  }
}

/// Base class for defining visitors that that will be called with all objects
/// that potentially (directly or indirectly) reference objects managed by a
/// garbage collector.
///
/// This is not relevant to pure C++ usage of TensorStore, but is needed to
/// interoperate with the Python garbage collector.
class GarbageCollectionVisitor {
 public:
  using ErasedVisitFunction = void (*)(GarbageCollectionVisitor& visitor,
                                       const void* ptr);

  /// Visits `obj` and any objects reachable from it.
  template <typename T>
  void Indirect(const T& obj) {
    Indirect<GarbageCollection<T>>(obj);
  }

  /// Same as above, but uses a custom traits type in place of
  /// `GarbageCollection<T>`.
  ///
  /// \tparam Traits Must define a
  ///     `static void Visit(GarbageCollectionVisitor& visitor, const T& value)`
  ///     method.
  template <typename Traits, typename T>
  void Indirect(const T& obj) {
    DoIndirect(
        typeid(T),
        [](GarbageCollectionVisitor& visitor, const void* ptr) {
          Traits::Visit(visitor, *static_cast<const T*>(ptr));
        },
        static_cast<const void*>(&obj));
  }

  virtual void DoIndirect(const std::type_info& type, ErasedVisitFunction visit,
                          const void* ptr) = 0;

 protected:
  ~GarbageCollectionVisitor() = default;
};

/// Returns `true` if any argument requires garbage collection.
struct DoesAnyRequireGarbageCollection {
  template <typename... T>
  constexpr auto operator()(const T&... arg) const {
    return std::integral_constant<bool,
                                  (IsGarbageCollectionRequired<T> || ...)>{};
  }
};

/// Garbage collection for types that implement the `ApplyMembers` protocol.
template <typename T>
struct ApplyMembersGarbageCollection {
  static void Visit(GarbageCollectionVisitor& visitor, const T& value) {
    ApplyMembers<T>::Apply(value, [&visitor](auto&&... member) {
      (garbage_collection::GarbageCollectionVisit(visitor, member), ...);
    });
  }

  constexpr static bool required() {
    return decltype(ApplyMembers<T>::Apply(
        std::declval<const T&>(), DoesAnyRequireGarbageCollection{}))::value;
  }
};

template <typename T>
struct GarbageCollection<
    T, std::enable_if_t<(SupportsApplyMembers<T> &&
                         !std::is_trivially_destructible_v<T>)>>
    : public ApplyMembersGarbageCollection<T> {};

template <typename T, typename ValueType = typename T::value_type>
struct ContainerGarbageCollection {
  static void Visit(GarbageCollectionVisitor& visitor, const T& value) {
    for (const auto& element : value) {
      garbage_collection::GarbageCollectionVisit(visitor, element);
    }
  }
  constexpr static bool required() {
    return IsGarbageCollectionRequired<ValueType>;
  }
};

template <typename T,
          typename ValueType = std::remove_cv_t<typename T::value_type>>
struct OptionalGarbageCollection {
  static void Visit(GarbageCollectionVisitor& visitor, const T& value) {
    if (!value) return;
    garbage_collection::GarbageCollectionVisit(visitor, *value);
  }
  constexpr static bool required() {
    return IsGarbageCollectionRequired<ValueType>;
  }
};

template <typename Pointer>
struct IndirectPointerGarbageCollection {
  static void Visit(GarbageCollectionVisitor& visitor, const Pointer& value) {
    if (!value) return;
    visitor.Indirect(*value);
  }
  constexpr static bool required() {
    return IsGarbageCollectionRequired<
        std::remove_cv_t<typename std::pointer_traits<Pointer>::element_type>>;
  }
};

template <typename T>
struct GarbageCollection<std::shared_ptr<T>>
    : public IndirectPointerGarbageCollection<std::shared_ptr<T>> {};

template <typename T, typename Traits>
struct GarbageCollection<internal::IntrusivePtr<T, Traits>>
    : public IndirectPointerGarbageCollection<
          internal::IntrusivePtr<T, Traits>> {};

template <typename T>
struct GarbageCollection<Result<T>>
    : public OptionalGarbageCollection<Result<T>> {};

template <typename T>
struct PolymorphicGarbageCollection {
  static void Visit(GarbageCollectionVisitor& visitor, const T& value) {
    value.GarbageCollectionVisit(visitor);
  }
};

}  // namespace garbage_collection
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_GARBAGE_COLLECTION_GARBAGE_COLLECTION_H_
