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

#ifndef TENSORSTORE_SERIALIZATION_FUNCTION_H_
#define TENSORSTORE_SERIALIZATION_FUNCTION_H_

/// \file
///
/// `SerializableFunction` is a type-erased function type (like `std::function`)
/// that is serializable.
///
/// Example:
///
///     tensorstore::SerializableFunction<int(int)> f = [](int x) { return 3 +
///     x; }; EXPECT_EQ(5, f(2));
///
///     struct MyFunction {
///       int a;
///       int operator()(int b) const { return a + b; }
///       constexpr static auto ApplyMembers = [](auto &&x, auto f) {
///         return f(x.a);
///       };
///     };
///     tensorstore::SerializableFunction<int(int)> f2 = MyFunction{3};
///     EXPECT_EQ(5, f(2));
///
/// A `SerializableFunction<R(Args...)>` may be implicitly constructed from any
/// function object of type `F` that satisfies the following conditions:
///
/// - `F` is serializable or an empty type (e.g. capture-less lambda), and
///
/// - `F` is const-invocable with the call signature `R(Args...)`,
///   i.e. satisfies `std::is_invocable_r_v<R, const F&, Args...>`.
///
/// Note that the parameter types and return type need not be serializable.
///
/// Capture-less lambdas are supported (they are not default constructible, but
/// are supported as a special case).  Lambdas with captures are not supported.
/// Plain function pointers and pointer-to-members are not supported.
///
/// Capturing state
/// ---------------
///
/// Functions that require captured state can be defined in two ways:
///
/// - Implement the function as a default-constructible class/struct that is
///   serializable, e.g. by defining `ApplyMembers`.
///
/// - Use the `BindFront` or `BackBack` adapter with a capture-less lambda or
///   other serializable function.
///
///   For example:
///
///       tensorstore::SerializableFunction<int(int)> f =
///           tensorstore::serialization::BindFront([](int a, int b)
///             return a + b;
///           }, 3);
///       EXPECT_EQ(5, f(2));
///
/// Registration
/// ------------
///
/// `SerializableFunction` is primarily intended to be used in cases where
/// serialization and deserialization are performed using an identical binary,
/// possibly running on different machines.
///
/// Constructing a `SerializableFunction` object from a function object of type
/// `F` causes `F` to be registered automatically (via a global initializer) in
/// a global registry, using a unique identifier.  By default, the unique
/// identifier is assigned automatically, as `typeid(F).name()`.  That avoids
/// the need to manually assign a unique identifier, but is only guaranteed to
/// work if serialization and deserialization are performed using the same
/// binary.  For improved stability of the encoded representation, a unique
/// identifier can also be specified manually, by defining a
/// `constexpr static const char id[]` member of `F`:
///
///     struct MyFunction {
///       constexpr static const char id[] = "MyFunction";
///       int a;
///       int operator()(int b) const { return a + b; }
///       constexpr static auto ApplyMembers = [](auto &&x, auto f) {
///         return f(x.a);
///       };
///     };
///     tensorstore::SerializableFunction<int(int)> f2 = MyFunction{3};
///
/// Even when using a manually specified `id`, you must still ensure that the
/// function type `F` is registered in the binary used for deserialization, by
/// ensuring that `tensorstore::SerializableFunction` is constructed from `F`
/// somewhere in the program (it does not have to be executed at run time).
///
/// Manually defined `id` values must begin with a letter `[a-zA-Z]`.
///
/// Attempting to register two different function types `F` with the same `id`
/// for the same signature `R(Args...)` leads to fatal error at run time during
/// global initialization.  In particular, if you wish to use a manually-defined
/// `id` value for a class template, you must ensure there is a unique `id` for
/// each combination of template parameters used with the same function
/// signature `R(Args...)`.
///
/// Optionally-serializable functions
/// ---------------------------------
///
/// In some cases it may be desirable to define an interface that takes an
/// optionally-serializable function.  This is supported by the
/// `NonSerializable` wrapper type: the interface can use a
/// `SerializableFunction` parameter, and a non-serializable function `f` can be
/// passed as `NonSerializable{f}`, which can be converted to a
/// `SerializableFunction` but will fail to serialize at run time.

#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/std_tuple.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace serialization {

namespace internal_serialization {
struct RegisteredSerializableFunction;

class SerializableFunctionBase
    : public internal::AtomicReferenceCount<SerializableFunctionBase> {
 public:
  using Ptr = internal::IntrusivePtr<SerializableFunctionBase>;
  using ErasedFunction = void (*)();
  virtual ~SerializableFunctionBase();
  virtual bool Encode(EncodeSink& sink) const = 0;
  virtual ErasedFunction erased_function() const = 0;
  virtual void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const = 0;
};

/// Registers a serializable function.
void RegisterSerializableFunction(const RegisteredSerializableFunction& r);

struct RegisteredSerializableFunction {
  using ErasedFunction = SerializableFunctionBase::ErasedFunction;
  using DecodeFunction = bool (*)(DecodeSource& source,
                                  SerializableFunctionBase::Ptr& impl);
  RegisteredSerializableFunction(const std::type_info* signature,
                                 std::string_view id, DecodeFunction decode)
      : signature(signature), id(id), decode(decode) {
    RegisterSerializableFunction(*this);
  }
  RegisteredSerializableFunction(const RegisteredSerializableFunction&) =
      delete;
  RegisteredSerializableFunction& operator=(
      const RegisteredSerializableFunction&) = delete;

  /// Key under which this function is registered: includes the `signature` as
  /// well as the `id` of the function.
  using Key = std::pair<std::type_index, std::string_view>;

  Key key() const { return {*signature, id}; }

  const std::type_info* signature;
  std::string_view id;
  DecodeFunction decode;
};

/// Evaluates to `true` if `T` has an `id` member.
template <typename T, typename SFINAE = void>
constexpr inline bool HasId = false;

template <typename T>
constexpr inline bool HasId<T, std::void_t<decltype(T::id)>> = true;

/// Returns either the manually assigned name, or `typeid(T).name()`, the
/// implementation-defined mangled name of `T`.
///
/// In practice, the mangled names always begin with `_Z` (Itanium ABI) or `?`
/// (MSVC).  Since manually assigned `id` values are required to begin with a
/// letter (or `0`, for names defined internally by tensorstore), they are
/// guaranteed not to conflict with a mangled name.
template <typename T>
constexpr std::string_view GetFunctionId() {
  if constexpr (HasId<T>) {
    return T::id;
  } else {
    return typeid(T).name();
  }
}

template <typename T, typename R, typename... Arg>
class SerializableFunctionImpl : public SerializableFunctionBase {
 public:
  SerializableFunctionImpl() = default;
  SerializableFunctionImpl(T&& func) : func_(std::move(func)) {}
  static R Invoke(const SerializableFunctionBase& base_impl, Arg... arg) {
    return static_cast<const SerializableFunctionImpl&>(base_impl).func_(
        std::forward<Arg>(arg)...);
  }
  static bool Decode(DecodeSource& source,
                     SerializableFunctionBase::Ptr& impl) {
    impl.reset(new SerializableFunctionImpl);
    return serialization::Decode(
        source, static_cast<SerializableFunctionImpl&>(*impl).func_);
  }

  bool Encode(EncodeSink& sink) const final {
    return serialization::EncodeTuple(sink, registry_entry_.id, func_);
  }

  ErasedFunction erased_function() const final {
    return reinterpret_cast<ErasedFunction>(&SerializableFunctionImpl::Invoke);
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    garbage_collection::GarbageCollectionVisit(visitor, func_);
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  internal::DefaultConstructibleFunctionIfEmpty<T> func_;

  static inline const RegisteredSerializableFunction registry_entry_{
      &typeid(R(*)(Arg...)),
      GetFunctionId<T>(),
      &SerializableFunctionImpl::Decode,
  };
};

class NonSerializableFunctionBase : public SerializableFunctionBase {
  bool Encode(EncodeSink& sink) const final;
  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final;
};

template <typename T, typename R, typename... Arg>
class NonSerializableFunctionImpl : public NonSerializableFunctionBase {
 public:
  NonSerializableFunctionImpl(T&& func) : func_(std::move(func)) {}

  static R Invoke(const SerializableFunctionBase& base_impl, Arg... arg) {
    return static_cast<const NonSerializableFunctionImpl&>(base_impl).func_(
        std::forward<Arg>(arg)...);
  }

  ErasedFunction erased_function() const override {
    return reinterpret_cast<ErasedFunction>(
        &NonSerializableFunctionImpl::Invoke);
  }

 private:
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  internal::DefaultConstructibleFunctionIfEmpty<T> func_;
};

bool DecodeSerializableFunction(DecodeSource& source,
                                SerializableFunctionBase::Ptr& value,
                                const std::type_info& signature);

}  // namespace internal_serialization

/// Evaluates to `true` if `Func` is convertible to
/// `SerializableFunction<R(Arg...)>`.
///
/// Normally, a type must be default constructible in order to be serializable,
/// but `SerializableFunction` allows non-default-constructible `Func` types
/// that are either empty or satisfy `NonSerializableLike`.  This is important
/// for supporting lambdas.
template <typename R, typename Func, typename... Arg>
constexpr inline bool IsSerializableFunctionLike =
    std::is_invocable_r_v<R, const Func&, Arg...> &&
    (IsSerializable<Func> || std::is_empty_v<Func> ||
     IsNonSerializableLike<Func>);

template <typename Signature>
class SerializableFunction;

/// Type-erased function that is serializable.
template <typename R, typename... Arg>
class SerializableFunction<R(Arg...)> {
 public:
  /// Constructs a null function that must not be invoked.
  SerializableFunction() = default;

  /// Constructs from a serializable function type `Func`.
  template <typename Func,
            std::enable_if_t<(std::is_invocable_r_v<R, const Func&, Arg...> &&
                              !IsNonSerializableLike<Func> &&
                              (IsSerializable<Func> ||
                               std::is_empty_v<Func>))>* = nullptr>
  SerializableFunction(Func func)
      : impl_(new internal_serialization::SerializableFunctionImpl<Func, R,
                                                                   Arg...>(
            std::move(func))) {}

  /// Constructs a `SerializableFunction` from an arbitrary non-serializable
  /// function.
  ///
  /// In this case, the function need not be default constructible.
  template <typename Func,
            std::enable_if_t<(std::is_invocable_r_v<R, const Func&, Arg...> &&
                              IsNonSerializableLike<Func>)>* = nullptr>
  SerializableFunction(Func func)
      : impl_(new internal_serialization::NonSerializableFunctionImpl<Func, R,
                                                                      Arg...>(
            std::move(func))) {}

  /// Returns true if this is a valid function that may be invoked, false if
  /// this is a default-constructed "null" function that must not be invoked.
  explicit operator bool() const { return static_cast<bool>(impl_); }

  R operator()(Arg... arg) const {
    assert(impl_);
    auto function = impl_->erased_function();
    return reinterpret_cast<R (*)(
        const internal_serialization::SerializableFunctionBase&, Arg...)>(
        function)(*impl_, std::forward<Arg>(arg)...);
  }

 private:
  internal_serialization::SerializableFunctionBase::Ptr impl_;
  friend struct Serializer<SerializableFunction<R(Arg...)>>;
  friend struct garbage_collection::GarbageCollection<
      SerializableFunction<R(Arg...)>>;
};

template <typename R, typename... Arg>
struct Serializer<SerializableFunction<R(Arg...)>> {
  [[nodiscard]] static bool Encode(
      EncodeSink& sink, const SerializableFunction<R(Arg...)>& value) {
    return value.impl_->Encode(sink);
  }

  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   SerializableFunction<R(Arg...)>& value) {
    return internal_serialization::DecodeSerializableFunction(
        source, value.impl_, typeid(R(*)(Arg...)));
  }
};

/// Equivalent of `std::bind_front` that is serializable if the bound function
/// and arguments are serializable.
///
/// This can be used with any function and bound argument types, but is only
/// serializable if the function and bound argument types are.
template <typename Func, typename... BoundArg>
class BindFront {
 public:
  BindFront() = default;

  BindFront(const Func& func, const BoundArg&... bound_arg)
      : func_(func), bound_args_(bound_arg...) {}
  template <typename... Arg>
  decltype(std::declval<const Func&>()(std::declval<const BoundArg&>()...,
                                       std::declval<Arg>()...))
  operator()(Arg&&... arg) const {
    return std::apply(
        [&](const auto&... bound_arg) {
          return func_(bound_arg..., std::forward<Arg&&>(arg)...);
        },
        bound_args_);
  }
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.func_, x.bound_args_);
  };

 private:
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  internal::DefaultConstructibleFunctionIfEmpty<Func> func_;

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  std::tuple<internal::DefaultConstructibleFunctionIfEmpty<BoundArg>...>
      bound_args_;
};

template <typename Func, typename... BoundArg>
BindFront(const Func& func, const BoundArg&... bound_arg)
    -> BindFront<Func, BoundArg...>;

template <typename Func, typename... BoundArg>
class BindBack {
 public:
  BindBack() = default;

  BindBack(const Func& func, const BoundArg&... bound_arg)
      : func_(func), bound_args_(bound_arg...) {}
  template <typename... Arg>
  std::invoke_result_t<const Func&, Arg..., const BoundArg&...> operator()(
      Arg&&... arg) const {
    return std::apply(
        [&](const auto&... bound_arg) {
          return func_(std::forward<Arg>(arg)..., bound_arg...);
        },
        bound_args_);
  }
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.func_, x.bound_args_);
  };

 private:
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  internal::DefaultConstructibleFunctionIfEmpty<Func> func_;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS
  std::tuple<internal::DefaultConstructibleFunctionIfEmpty<BoundArg>...>
      bound_args_;
};

template <typename Func, typename... BoundArg>
BindBack(const Func& func, const BoundArg&... bound_arg)
    -> BindBack<Func, BoundArg...>;

}  // namespace serialization

using serialization::NonSerializable;       // NOLINT(misc-unused-using-decls)
using serialization::SerializableFunction;  // NOLINT(misc-unused-using-decls)

namespace garbage_collection {
template <typename R, typename... Arg>
struct GarbageCollection<serialization::SerializableFunction<R(Arg...)>> {
  static void Visit(
      GarbageCollectionVisitor& visitor,
      const serialization::SerializableFunction<R(Arg...)>& value) {
    if (!value) return;
    value.impl_->GarbageCollectionVisit(visitor);
  }
};
}  // namespace garbage_collection

}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_BATCH_H_
