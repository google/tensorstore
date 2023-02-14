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

#ifndef TENSORSTORE_SERIALIZATION_SERIALIZATION_H_
#define TENSORSTORE_SERIALIZATION_SERIALIZATION_H_

/// \file
///
/// Serialization framework for tensorstore.
///
/// This framework provides a mechanism for converting data structures defined
/// by the tensorstore library to/from byte streams.  Data structures including
/// cycles/shared references are supported via a special indirect object
/// reference mechanism.
///
/// This framework serves two primary use cases:
///
/// - Pickling support in the Python API: This framework fully interoperates
///   with the Python pickling mechanism.  Shared object references are
///   correctly handled across C++ and Python, including tensorstore data
///   structures that contain references to Python objects, such as
///   :py:obj:`tensorstore.chunked_computed` TensorStore objects created through
///   the Python API.
///
/// - tensorstore.distributed: This framework is used to send tasks and shared
///   data from the controller process to the workers processes, and to receive
///   and persist task results.
///
/// Format compatibility guarantees
/// -------------------------------
///
/// There are currently no backward or forward compatibility guarantees
/// regarding the serialization format: serialization is intended for short-term
/// storage, e.g. for storing intermediate data within a distributed pipeline,
/// not for long-term storage of results.
///
/// Except where otherwise noted, data encoded using a given version of
/// tensorstore on a platform with a given endianness, can only be decoded using
/// the same version of tensorstore, on a platform with the same endianness.
///
/// Additionally, the compatibility of the `SerializableFunction` representation
/// is by default even more limited: a `SerializableFunction` encoded with a
/// given binary is only guaranteed to be decodable by the identical binary (but
/// which may be running on a different machine).  This compatibility limitation
/// is a trade-off in exchange for avoiding the need for manual registration of
/// functions.  However, there is a mechanism by which unique identifiers may be
/// manually assigned to function types in order to obtain better compatibility.
///
/// Adding serialization support for a new type
/// -------------------------------------------
///
/// The simplest way to add serialization support for a type is to implement the
/// `ApplyMembers` protocol:
///
///     struct X {
///       int a;
///       std::string b;
///       constexpr static auto ApplyMembers = [](auto &&x, auto f) {
///         return f(x.a, x.b);
///       };
///     };
///
/// This provides a very simple mechanism for supporting serialization of
/// aggregate-like types.
///
/// For more complicated types, a specialization of
/// `tensorstore::serialization::Serializer` may be defined:
///
///     namespace tensorstore {
///     namespace serialization {
///     template <>
///     struct Serializer<X> {
///       [[nodiscard]] static bool Encode(EncodeSink& sink, const X& value);
///       [[nodiscard]] static bool Decode(DecodeSource& source, X& value);
///       constexpr static bool non_serializable() { return false; }
///     };
///     }  // namespace serialization
///     }  // namespace tensorstore
///
/// The `Encode` method (which may optionally be non-static) must write the
/// representation of `value` to `sink` and return `true` on success.  If an
/// error occurs, it should ensure an error status is set on `sink` by calling
/// `EncodeSink::Fail` and return `false`.
///
/// Similarly, the `Decode` method must initialize `value` from `source` and
/// return `true` on success.  If unexpected EOF occurs, it may `false` without
/// setting an error status on `source`.  If another error occurs, it must
/// ensure an error status is set on `sink` by calling `DecodeSource::Fail`, and
/// then return `false`.
///
/// The `non_serializable()` member is optional and is not normally specified.
/// It indicates special types that claim to support serialization at compile
/// time, but for which serialization is known to fail at run time; this
/// includes instances of `NonSerializable` as well as any (likely templated)
/// types that include nested `NonSerializable` members.  If not specified, it
/// is assume to equal `false`.  `NonSerializable` is used with
/// `SerializableFunction` to allow a single interface to be used for both
/// serialiable and non-serializable functions.

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "absl/functional/function_ref.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/riegeli_delimited.h"
#include "tensorstore/util/apply_members/apply_members.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

namespace serialization {

namespace internal_serialization {
void FailNonNull(DecodeSource& source);
void FailEof(DecodeSource& source);
}  // namespace internal_serialization

/// Abstract base output sink type used to encode object representations.
///
/// This interface is used by all `Serializer` implementations.
///
/// Any uniquely-owned data should be written directly to the `riegeli::Writer`
/// returned by `writer()`.  Shared references should be written using
/// `Indirect`.
class EncodeSink {
 public:
  /// Returns the `riegeli::Writer` that should be used to encode all
  /// uniquely-owned data.
  riegeli::Writer& writer() { return writer_; }

  /// Marks `writer()` as unhealthy with the specified error status.
  ///
  /// Any failing encode operation must call `Fail` to mark `writer()` as
  /// unhealthy, unless it is guaranteed that another write operation has
  /// already done so.
  ///
  /// \dchecks `!status.ok()`.
  void Fail(absl::Status status);

  /// Returns the error status of `writer()`.
  absl::Status status() const { return writer_.status(); }

  /// Finalizes encoding.
  ///
  /// Must be called after all objects encoded.
  ///
  /// \returns `true` if healthy, `false` if unhealthy.
  virtual bool Close() { return writer_.Close(); }

  /// Writes an indirect object reference.
  ///
  /// The object reference can be decoded using `DecodeSource::Indirect`.
  ///
  /// If `Indirect` is called more than once with the same `object`, the
  /// `object` will only be serialized and stored once.  This allows object
  /// graphs with (directed) cycles to be accurately encoded and decoded.  It is
  /// unspecified which `serializer` object will be used to perform the
  /// serialization.  The actual call to `serializer` may occur either before or
  /// after `Indirect` returns.  Therefore, any copies of `serializer` must
  /// remain valid to use until they are destroyed.
  ///
  /// \param object Non-null pointer to shared object.
  /// \param serializer Serializer to use to encode `object`.
  template <typename T,
            typename DirectSerializer = Serializer<std::shared_ptr<T>>>
  [[nodiscard]] bool Indirect(std::shared_ptr<T> object,
                              DirectSerializer serializer = {}) {
    return DoIndirect(
        typeid(std::shared_ptr<T>),
        [serializer = std::move(serializer)](
            EncodeSink& sink, const std::shared_ptr<void>& value) {
          return serializer.Encode(sink, std::static_pointer_cast<T>(value));
        },
        internal::StaticConstPointerCast<void>(std::move(object)));
  }

  /// Same as above, but the object is referenced using an `IntrusivePt` rather
  /// than a `shared_ptr`.
  ///
  /// \param object Non-null pointer to shared object.
  /// \param serializer Serializer to use to encode `object`.
  template <
      typename T, typename Traits,
      typename DirectSerializer = Serializer<internal::IntrusivePtr<T, Traits>>>
  [[nodiscard]] bool Indirect(internal::IntrusivePtr<T, Traits> object,
                              DirectSerializer serializer = {}) {
    return DoIndirect(
        typeid(internal::IntrusivePtr<T, Traits>),
        [serializer = std::move(serializer)](
            EncodeSink& sink, const std::shared_ptr<void>& value) {
          return serializer.Encode(sink, internal::IntrusivePtr<T, Traits>(
                                             static_cast<T*>(value.get())));
        },
        internal::StaticConstPointerCast<void>(
            internal::IntrusiveToShared(std::move(object))));
  }

  /// Type-erased encode function for use with `DoIndirect`.
  ///
  /// This uses `Poly` rather than `absl::FunctionView` because `DoIndirect` is
  /// not required to call the function before returning.
  using ErasedEncodeWrapperFunction =
      poly::Poly<0, /*Copyable=*/true,
                 bool(EncodeSink& sink,
                      const std::shared_ptr<void>& erased_value) const>;

  /// Writes a type-erased indirect object reference.
  ///
  /// \param type The typeid of the smart pointer type (not the pointee type).
  ///     An `EncodeSink` implementation may choose to handle certain known
  ///     types specially.
  /// \param encode Encode function that may be used to encode `object`.  May be
  ///     retained and called after `DoIndirect` returns.
  /// \param object Non-null shared object reference, to be de-duplicated
  ///     according to the value of `object.get()`.
  [[nodiscard]] virtual bool DoIndirect(const std::type_info& type,
                                        ErasedEncodeWrapperFunction encode,
                                        std::shared_ptr<void> object) = 0;

 protected:
  explicit EncodeSink(riegeli::Writer& writer) : writer_(writer) {}
  ~EncodeSink() = default;

 private:
  riegeli::Writer& writer_;
};

/// Returns a generic error indicating that decoding failed (due to invalid
/// input data).
absl::Status DecodeError();
absl::Status DecodeError(std::string_view message);

/// Input source used to decode object representations.
class DecodeSource {
 public:
  /// Reader from which the byte representation of objects should be read.
  riegeli::Reader& reader() { return reader_; }

  /// Marks `reader()` as unhealthy with the specified error status.
  ///
  /// Any decoding operation that fails for a reason other than EOF must call
  /// `Fail` to mark `reader()` as unhealthy, unless it is guaranteed that
  /// another failing read operation has already done so.
  ///
  /// \dchecks `!status.ok()`
  void Fail(absl::Status status);

  /// Returns the error status associated with `reader()`.
  absl::Status status() const { return reader_.status(); }

  /// Validates that all data has been consumed.
  virtual absl::Status Done() {
    if (reader_.VerifyEndAndClose()) return absl::OkStatus();
    return status();
  }

  /// Decodes an indirect object reference.
  ///
  /// If the same object (as determined by pointer identity) was encoded
  /// multiple times, it will be decoded using the specified `serializer` when
  /// the first indirect reference is decoded.  When subsequent references to
  /// the same object are decoded, `object` is set to a shared pointer to the
  /// previously-decoded object and `serializer` is ignored.  It is an error to
  /// attempt to decode a subsequent reference to a previously-decoded object
  /// with a different pointer type, even if the pointee types are related by
  /// inheritance.
  ///
  /// \param object Reference to be set to a pointer to the decoded object.
  /// \param serializer Serializer to use to decode the object.
  template <typename T,
            typename DirectSerializer = Serializer<std::shared_ptr<T>>>
  [[nodiscard]] bool Indirect(std::shared_ptr<T>& object,
                              DirectSerializer serializer = {}) {
    std::shared_ptr<void> void_ptr;
    if (!DoIndirect(
            typeid(std::shared_ptr<T>),
            [serializer = std::move(serializer)](DecodeSource& source,
                                                 std::shared_ptr<void>& value) {
              std::shared_ptr<T> typed_value;
              if (!serializer.Decode(source, typed_value)) return false;
              value = std::move(typed_value);
              return true;
            },
            void_ptr)) {
      return false;
    }
    object = internal::static_pointer_cast<T>(std::move(void_ptr));
    return true;
  }

  /// Same as above, but the object is referenced using an `IntrusivePt` rather
  /// than a `shared_ptr`.
  ///
  /// \param object Reference to be set to a pointer to the decoded object.
  /// \param serializer Serializer to use to decode the `object`.
  template <
      typename T, typename Traits,
      typename DirectSerializer = Serializer<internal::IntrusivePtr<T, Traits>>>
  [[nodiscard]] bool Indirect(internal::IntrusivePtr<T, Traits>& object,
                              DirectSerializer serializer = {}) {
    std::shared_ptr<void> void_ptr;
    if (!DoIndirect(
            typeid(internal::IntrusivePtr<T, Traits>),
            [&serializer](DecodeSource& source, std::shared_ptr<void>& value) {
              internal::IntrusivePtr<T, Traits> typed_value;
              if (!serializer.Decode(source, typed_value)) return false;
              value = internal::StaticConstPointerCast<void>(
                  internal::IntrusiveToShared(std::move(typed_value)));
              return true;
            },
            void_ptr)) {
      return false;
    }
    object.reset(static_cast<T*>(void_ptr.get()));
    return true;
  }

  /// Type-erased decode function for use with `DoIndirect`.
  using ErasedDecodeWrapperFunction = absl::FunctionRef<bool(
      DecodeSource& source, std::shared_ptr<void>& value)>;

  /// Reads a type-erased indirect object reference.
  ///
  /// \param type The typeid of the smart pointer type (not the pointee type).
  ///     A `DecodeSink` implementation may choose to handle certain known types
  ///     specially.  Additionally, when decoding another reference to a
  ///     previously-decoded object, the `DecodeSink` implementation must ensure
  ///     that `type` matches the type specified on the first call to
  ///     `DoIndirect`.
  /// \param decode Decode function that may be used to decode the object.
  ///     Guaranteed not to be retained after `DoIndirect` returns.
  /// \param value Reference to be set to a pointer to the decoded object.
  [[nodiscard]] virtual bool DoIndirect(const std::type_info& type,
                                        ErasedDecodeWrapperFunction decode,
                                        std::shared_ptr<void>& value) = 0;

 protected:
  DecodeSource(riegeli::Reader& reader) : reader_(reader) {}
  ~DecodeSource() = default;

 private:
  riegeli::Reader& reader_;
};

/// Wrapper that may be used to store a non-serializable function in a
/// `SerialzableFunction`.
///
/// This is useful in cases where serialization may optionally be desired.
///
/// Attempting to serialize the resultant `SerializableFunction` results in an
/// error at run time.
template <typename T>
struct NonSerializable : public T {
  // Reflection forwards to the base type for non-serialization use cases.
  static constexpr auto ApplyMembers = [](auto&& x, auto f) {
    return f(internal::BaseCast<T>(x));
  };
};

template <typename T>
NonSerializable(const T& x) -> NonSerializable<T>;

/// Indicates if `T` is an unqualified instance of `NonSerializable`.
///
/// See also `IsNonSerializableLike`.
template <typename T>
constexpr inline bool IsNonSerializable = false;

template <typename T>
constexpr inline bool IsNonSerializable<NonSerializable<T>> = true;

namespace internal_serialization {
absl::Status NonSerializableError();
}  // namespace internal_serialization

template <typename T>
struct Serializer<NonSerializable<T>> {
  [[nodiscard]] static bool Encode(EncodeSink& sink,
                                   const NonSerializable<T>& value) {
    sink.Fail(internal_serialization::NonSerializableError());
    return false;
  }
  [[nodiscard]] static bool Decode(DecodeSource& source,
                                   NonSerializable<T>& value) {
    source.Fail(internal_serialization::NonSerializableError());
    return false;
  }
  constexpr static bool non_serializable() { return true; }
};

/// Indicates whether a `Serializer` type is known to always fail at run time.
///
/// This may be used to reduce overhead (e.g. unnecessary registrations).
///
/// This is an optional protocol supported by some serializers.  Serialiers opt
/// into this protocol by defining a static constexpr nullary method
/// `non_serializable()` that returns a `bool`.
///
/// False positives are not allowed (i.e. if `IsNonSerializer<Serializer>` is
/// true, then `Serializer` must fail at run time) but false negatives are
/// allowed (i.e. for some serializer `Serializer` that does not support
/// serialization at run time, `IsNonSerializer<Serializer>` may still evaluate
/// to `false`).
template <typename Serializer, typename SFINAE = void>
constexpr inline bool IsNonSerializer = false;

template <typename Serializer>
constexpr inline bool IsNonSerializer<
    Serializer, std::void_t<decltype(&Serializer::non_serializable)>> =
    Serializer::non_serializable();

/// Indicates whether a type is "non-serializable", i.e. serialization is
/// supported at compile time but always results in an error at run time.
template <typename T>
constexpr inline bool IsNonSerializableLike = IsNonSerializer<Serializer<T>>;

/// Serializer for trivial types for which the in-memory representation is the
/// same as the encoded representation.
template <typename T>
struct MemcpySerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const T& value) {
    return sink.writer().Write(
        std::string_view(reinterpret_cast<const char*>(&value), sizeof(T)));
  }
  [[nodiscard]] static bool Decode(DecodeSource& source, T& value) {
    return source.reader().Read(sizeof(T), reinterpret_cast<char*>(&value));
  }
};

/// Use `MemcpySerializer` by default for built-in integer, floating-point, and
/// enum types, as well as for `bfloat16_t` and `float16_t`.
template <typename T>
struct Serializer<T, std::enable_if_t<SerializeUsingMemcpy<T>>>
    : public MemcpySerializer<T> {};

/// Serializer for `bool`.
///
/// We cannot use `MemcpySerializer` for `bool` because we must ensure that a
/// decoded value is exactly `0` or `1`.
template <>
struct Serializer<bool> {
  [[nodiscard]] static bool Encode(EncodeSink& sink, bool value) {
    return sink.writer().WriteByte(value);
  }
  [[nodiscard]] static bool Decode(DecodeSource& source, bool& value) {
    uint8_t v;
    if (!source.reader().ReadByte(v)) return false;
    value = static_cast<bool>(v);
    return true;
  }
};

/// Convenient interface for encoding an object with its default serializer.
template <typename T, typename ElementSerializer = Serializer<T>>
[[nodiscard]] bool Encode(EncodeSink& sink, const T& value,
                          const ElementSerializer& serialize = {}) {
  return serialize.Encode(sink, value);
}

/// Convenient interface for decoding an object with its default serializer.
template <typename T,
          typename ElementSerializer = Serializer<internal::remove_cvref_t<T>>>
[[nodiscard]] bool Decode(DecodeSource& source, T&& value,
                          const ElementSerializer& serialize = {}) {
  return serialize.Decode(source, value);
}

/// Serializes string-like types in length-delimited format.
template <typename String>
struct StringSerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const String& value) {
    return serialization::WriteDelimited(sink.writer(), value);
  }
  [[nodiscard]] static bool Decode(DecodeSource& source, String& value) {
    return serialization::ReadDelimited(source.reader(), value);
  }
};

template <>
struct Serializer<std::string> : public StringSerializer<std::string> {};

template <>
struct Serializer<absl::Cord> : public StringSerializer<absl::Cord> {};

template <>
struct Serializer<std::string_view>
    : public StringSerializer<std::string_view> {};

/// Convenience interface for encoding a heterogeneous sequence of values.
template <typename... T>
[[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE inline bool EncodeTuple(
    EncodeSink& sink, const T&... value) {
  return (serialization::Encode(sink, value) && ...);
}

/// Convenience interface for decoding a heterogeneous sequence of values.
template <typename... T>
[[nodiscard]] ABSL_ATTRIBUTE_ALWAYS_INLINE inline bool DecodeTuple(
    DecodeSource& source, T&&... value) {
  return (serialization::Decode(source, value) && ...);
}

/// Returns `true` if any argument has a non-serializable type.
struct IsAnyNonSerializable {
  template <typename... T>
  constexpr auto operator()(const T&... arg) const {
    return std::integral_constant<bool, (IsNonSerializableLike<T> || ...)>{};
  }
};

/// Serializer for types that implement the `ApplyMembers` protocol.
template <typename T>
struct ApplyMembersSerializer {
  [[nodiscard]] static bool Encode(EncodeSink& sink, const T& value) {
    return ApplyMembers<T>::Apply(value, [&sink](const auto&... member) {
      return (serialization::Encode(sink, member) && ...);
    });
  }

  [[nodiscard]] static bool Decode(DecodeSource& source, T& value) {
    return ApplyMembers<T>::Apply(value, [&source](auto&&... member) {
      return (serialization::Decode(source, member) && ...);
    });
  }

  constexpr static bool non_serializable() {
    return decltype(ApplyMembers<T>::Apply(std::declval<const T&>(),
                                           IsAnyNonSerializable{}))::value;
  }
};

template <typename T>
struct Serializer<
    T, std::enable_if_t<(SupportsApplyMembers<T> && !IsNonSerializable<T> &&
                         !SerializeUsingMemcpy<T>)>>
    : public ApplyMembersSerializer<T> {};

/// Serializes a container type.
///
/// The size is encoded followed by each element.  If the size will be already
/// known at decoding time, you can avoid redundantly encoding the size by using
/// `SpanSerializer` instead.
template <typename T, typename ValueType = typename T::value_type,
          typename ElementSerializer = Serializer<ValueType>>
struct ContainerSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, const T& value) const {
    if (!serialization::WriteSize(sink.writer(), value.size())) return false;
    for (const auto& element : value) {
      if (!serialization::Encode(sink, element, element_serializer)) {
        return false;
      }
    }
    return true;
  }
  [[nodiscard]] bool Decode(DecodeSource& source, T& value) const {
    value.clear();
    size_t size;
    if (!serialization::ReadSize(source.reader(), size)) return false;
    for (size_t i = 0; i < size; ++i) {
      ValueType element;
      if (!serialization::Decode(source, element, element_serializer)) {
        return false;
      }
      value.insert(value.end(), std::move(element));
    }
    return true;
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementSerializer
      element_serializer = {};

  constexpr static bool non_serializable() {
    return IsNonSerializer<ElementSerializer>;
  }
};

template <typename T,
          typename ElementSerializer = Serializer<typename T::value_type>>
struct OptionalSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, const T& value) const {
    return serialization::Encode(sink, static_cast<bool>(value)) &&
           (!value || element_serializer.Encode(sink, *value));
  }
  [[nodiscard]] bool Decode(DecodeSource& source, T& value) const {
    bool has_value;
    return serialization::Decode(source, has_value) &&
           (!has_value || element_serializer.Decode(source, value.emplace()));
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementSerializer element_serializer;
  constexpr static bool non_serializable() {
    return IsNonSerializer<ElementSerializer>;
  }
};

template <typename T, typename SFINAE = void>
inline constexpr bool IsSerializable = false;

template <typename T>
inline constexpr bool IsSerializable<
    T, std::void_t<decltype(Serializer<T>::Encode(std::declval<EncodeSink&>(),
                                                  std::declval<const T&>()))>> =
    std::is_default_constructible_v<T>;

/// Serialization of "nullable" values
/// ==================================
///
/// Some types like pointers support a "null" state in addition to the normal
/// value state.
///
/// For some types it is more natural to define a serializer for non-null values
/// specifically, and then in cases where we need to support null values as
/// well, we can adapt the base serializer using the `MaybeNullSerializer`
/// template.  `MaybeNullSerializer` first encodes a `bool` to indicate whether
/// the value is null, and then only if it is non-null encodes the value itself
/// using the base serializer.
///
/// Conversely, for other types it may be more natural to define a serializer
/// that handles both "null" and "non-null" states, but in some cases we may
/// wish to constrain the value to be non-null.  That can be accomplished using
/// `NonNullSerializer`, which adapts a base serializer and asserts the value is
/// non-null when encoding, and verifies the value is non-null when decoding.
///
/// For both `MaybeNullSerializer` and `NonNullSerializer`, "non-null" states
/// are determined using a predicate specified as an empty function object type,
/// such as `IsNonNull` or `IsValid`.

/// Function object type for use with `MaybeNullSerializer` that relies on
/// conversion to `bool`.
struct IsNonNull {
  template <typename T>
  constexpr bool operator()(const T& x) const {
    return static_cast<bool>(x);
  }
};

/// Function object type for use with `MaybeNullSerializer` that calls the
/// `valid()` method.
struct IsValid {
  template <typename T>
  constexpr bool operator()(const T& x) const {
    return x.valid();
  }
};

/// Serializer that adapts a `NonNullSerializer` that only handles "non-null"
/// values to also support "null" values.
///
/// \tparam T Type of object (e.g. smart pointer) to be serialized.
/// \tparam NonNullSerializer Serializer that may be used with non-null values
///     of type `T`.  For example, if `T` is a pointer type, `NonNullSerializer`
///     may be `NonNullPointerSerializer<T>`.
/// \tparam IsNullPredicate Stateless function object type that checks if an
///     object of type `T` is "null".
template <typename T, typename NonNullSerializer,
          typename IsNullPredicate = IsNonNull>
struct MaybeNullSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, const T& value) const {
    const bool valid = IsNullPredicate{}(value);
    if (!serialization::Encode(sink, valid)) return false;
    if (!valid) return true;
    return non_null_serializer.Encode(sink, value);
  }
  [[nodiscard]] bool Decode(DecodeSource& source, T& value) const {
    bool valid;
    if (!serialization::Decode(source, valid)) return false;
    if (!valid) return true;
    if (!non_null_serializer.Decode(source, value)) return false;
    assert(IsNullPredicate{}(value));
    return true;
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS NonNullSerializer
      non_null_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<NonNullSerializer>;
  }
};

/// Serializer that adapts a `BaseSerializer` that supports "null" values to
/// ensure that only "non-null" values are supported.
///
/// \tparam T Type of object (e.g. smart pointer) to be serialized.
/// \tparam BaseSerializer Serialier that may be used with "nullable" values of
///     type `T`.
/// \tparam Predicate Stateless function object type that checks if an object of
///     type `T` is "null".
template <typename T, typename BaseSerializer = Serializer<T>,
          typename Predicate = IsNonNull>
struct NonNullSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, const T& value) const {
    assert(Predicate{}(value));
    return base_serializer.Encode(sink, value);
  }
  [[nodiscard]] bool Decode(DecodeSource& source, T& value) const {
    if (!base_serializer.Decode(source, value)) return false;
    if (!Predicate{}(value)) {
      internal_serialization::FailNonNull(source);
      return false;
    }
    return true;
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS BaseSerializer base_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<BaseSerializer>;
  }
};

/// Serialization of pointers
/// =========================
///
/// For pointer-like types, there are several axes along which the serialization
/// implementation varies:
///
/// - Direct or indirect:
///
///   Direct: a separate copy of the pointee is encoded for each occurrence of
///   the pointer, and when deserialized each pointee will be unique.  This is
///   suitable for unique ownership, such as `std::unique_ptr`, and for
///   value-semantic types.  This is implemented by `NonNullPointerSerializer`.
///
///   Indirect: a single copy of the pointee is serialized (as determined by its
///   address), which is indirectly referenced by each pointer to it.  This is
///   suitable for shared ownership, such as `std::shared_ptr` and
///   `internal::IntrusivePtr`.  This is implemented by
///   `NonNullIndirectPointerSerializer`.
///
/// - Null or non-null: If only non-null pointers are supported, you can use
///   `NonNullPointerSerializer` or `NonNullIndirectPointerSerializer`.  If null
///   values are permitted, you can use `PointerSerializer` or
///   `IndirectPointerSerializer`, which adapt the corresponding non-null
///   pointer serializer using `MaybeNullSerializer`.
///
/// - Static type or polymorphic:
///
///   Static type: If the fully-derived type of the pointee is always known at
///   compile-time, we can simply deserialize the value directly.  This is
///   implemented by `NonNullPointerSerializer`.
///
///   Polymorphic: To encode/decode arbitrary derived types via a pointer to a
///   base type, `RegistrySerializer` defined in `registry.h` may be used.

/// Serializer for non-null smart pointers.
///
/// When decoding, allocates a new object using `operator new`.
///
/// By itself this does not support shared ownership, but may be used with
/// `IndirectPointerSerializer` to handle shared ownership.
///
/// \tparam ElementSerializer Serializer for the pointee.
template <typename Pointer,
          typename ElementSerializer =
              Serializer<std::remove_cv_t<typename Pointer::element_type>>>
struct NonNullPointerSerializer {
  using element_type = std::remove_cv_t<typename Pointer::element_type>;
  [[nodiscard]] bool Encode(EncodeSink& sink, const Pointer& value) const {
    assert(value);
    return element_serializer.Encode(sink, *value);
  }

  [[nodiscard]] bool Decode(DecodeSource& source, Pointer& value) const {
    value.reset(new element_type);
    return element_serializer.Decode(source, *value);
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementSerializer
      element_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<ElementSerializer>;
  }
};

template <typename Pointer,
          typename NonNullSerializer = NonNullPointerSerializer<Pointer>>
using PointerSerializer = MaybeNullSerializer<Pointer, NonNullSerializer>;

/// Serializes a non-null pointer type with support for shared ownership.
///
/// \tparam Pointer Pointer type to serialize.
/// \tparam NonNullSerializer Direct serializer for non-null objects of type
///     `Pointer`.
template <typename Pointer,
          typename NonNullSerializer = NonNullPointerSerializer<Pointer>>
struct NonNullIndirectPointerSerializer {
  [[nodiscard]] bool Encode(EncodeSink& sink, const Pointer& value) const {
    assert(value);
    return sink.Indirect(value, non_null_serializer);
  }

  [[nodiscard]] bool Decode(DecodeSource& source, Pointer& value) const {
    return source.Indirect(value, non_null_serializer);
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS NonNullSerializer
      non_null_serializer = {};
  constexpr static bool non_serializable() {
    return IsNonSerializer<NonNullSerializer>;
  }
};

template <typename Pointer,
          typename NonNullSerializer = NonNullPointerSerializer<Pointer>>
using IndirectPointerSerializer = MaybeNullSerializer<
    Pointer, NonNullIndirectPointerSerializer<Pointer, NonNullSerializer>>;

template <typename T>
struct Serializer<std::unique_ptr<T>,
                  std::enable_if_t<IsSerializable<std::remove_cv_t<T>>>>
    : public PointerSerializer<std::unique_ptr<T>> {};

template <typename T>
struct Serializer<std::shared_ptr<T>,
                  std::enable_if_t<IsSerializable<std::remove_cv_t<T>>>>
    : public IndirectPointerSerializer<std::shared_ptr<T>> {};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_SERIALIZATION_H_
