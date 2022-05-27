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

#ifndef TENSORSTORE_DRIVER_ENCODING_SPEC_H_
#define TENSORSTORE_DRIVER_ENCODING_SPEC_H_

#include <iosfwd>
#include <type_traits>

#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"

namespace tensorstore {

class CodecSpec;

namespace internal {

/// Each driver defines a derived class of `internal::CodecDriverSpec` that is
/// registered using `codec_spec_registry.h`.
class CodecDriverSpec : public AtomicReferenceCount<CodecDriverSpec> {
 public:
  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  // Pointer to derived `internal::CodecDriverSpec` type.
  template <typename T>
  using PtrT = internal::IntrusivePtr<T>;

  /// Checks this codec spec for equality with another codec spec.
  ///
  /// By default, compares the JSON representations and returns false if
  /// conversion to JSON fails.
  ///
  /// \param other Another codec spec, not guaranteed to have the same dynamic
  ///     type.  In most cases, an error should be returned if the dynamic type
  ///     differs.
  virtual bool EqualTo(const internal::CodecDriverSpec& other) const;

  /// Merges another codec spec into this spec.
  ///
  /// If incompatible constraints are specified, an error should be returned.
  ///
  /// \param other_base Other codec spec, not guaranteed to have the same
  ///     dynamic type.  In most cases, an error should be returned if the
  ///     dynamic type differs.
  virtual absl::Status DoMergeFrom(
      const internal::CodecDriverSpec& other_base) = 0;

  /// Same as above, but handles null `other`.
  absl::Status MergeFrom(const CodecSpec& other);

  /// Returns a new copy of this codec spec.
  virtual CodecSpec Clone() const = 0;

  virtual ~CodecDriverSpec();

  template <typename T, typename... Args>
  static PtrT<T> Make(Args&&... args) {
    static_assert(std::is_base_of_v<internal::CodecDriverSpec, T>);
    return internal::MakeIntrusivePtr<T>(std::forward<Args>(args)...);
  }
};

}  // namespace internal

/// Specifies compression and other encoding/decoding parameters for a
/// TensorStore driver.
///
/// This allows compression options to be specified as part of a `Schema`
/// independent of other driver parameters and metadata.
///
/// Since compression parameters are highly driver-specific, a `CodecSpec` is
/// always associated with a particular driver.
///
/// \relates Schema
class CodecSpec
    : public internal::IntrusivePtr<const internal::CodecDriverSpec> {
  using Base = internal::IntrusivePtr<const internal::CodecDriverSpec>;

 public:
  /// JSON serialization options.
  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  /// Constructs a null codec spec.
  ///
  /// \id default
  CodecSpec() = default;

  using Base::Base;
  CodecSpec(Base ptr) : Base(std::move(ptr)) {}

  /// Returns `true` if not null.
  bool valid() const { return static_cast<bool>(*this); }

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(CodecSpec, FromJsonOptions,
                                          ToJsonOptions)

  /// Merges this codec spec with another codec spec.
  ///
  /// Modifies `*this` to refer to the merged codec spec.  Merging two null
  /// codec specs simply results in a null codec spec.  Merging a null and
  /// non-null codec spec simply preserves the non-null codec spec.
  absl::Status MergeFrom(CodecSpec other);

  /// Merges two codec specs.
  static Result<CodecSpec> Merge(CodecSpec a, CodecSpec b);

  /// Compares two codec specs for equality by value.
  ///
  /// Null values are supported.
  friend bool operator==(const CodecSpec& a, const CodecSpec& b);
  friend bool operator!=(const CodecSpec& a, const CodecSpec& b) {
    return !(a == b);
  }

  /// Writes the JSON representation of `codec` to `os`.
  ///
  /// \param os Output stream.
  /// \param codec Codec to write, may be null.
  friend std::ostream& operator<<(std::ostream& os, const CodecSpec& codec);
};

namespace internal {

struct CodecSpecNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const CodecSpec& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   CodecSpec& value);

  // Also support `IntrusivePtr<internal::CodecDriverSpec>` for use by the
  // Python bindings.
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal::IntrusivePtr<internal::CodecDriverSpec>& value);
  [[nodiscard]] static bool Decode(
      serialization::DecodeSource& source,
      internal::IntrusivePtr<internal::CodecDriverSpec>& value);
};
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::CodecSpec)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::IntrusivePtr<tensorstore::internal::CodecDriverSpec>)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::CodecSpec)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal::IntrusivePtr<tensorstore::internal::CodecDriverSpec>)

#endif  // TENSORSTORE_DRIVER_ENCODING_SPEC_H_
