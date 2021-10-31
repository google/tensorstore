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

#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"

namespace tensorstore {

/// Specifies compression and other encoding/decoding parameters for a
/// TensorStore driver.
///
/// This allows compression options to be specified as part of a `Schema` or
/// `Schema` object independent of other driver parameters and
/// metadata.
///
/// Since compression parameters are highly driver-specific, a `CodecSpec` is
/// always associated with a particular driver.
///
/// Each driver defines a derived class of `CodecSpec` that is registered using
/// `codec_spec_registry.h`.
class CodecSpec : public internal::AtomicReferenceCount<CodecSpec> {
 public:
  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  /// Pointer to derived `CodecSpec` type.
  template <typename T>
  using PtrT = internal::IntrusivePtr<T>;

  class Ptr : public PtrT<const CodecSpec> {
    using Base = PtrT<const CodecSpec>;

   public:
    using ToJsonOptions = JsonSerializationOptions;
    using FromJsonOptions = JsonSerializationOptions;
    using Base::Base;
    Ptr(Base ptr) : Base(std::move(ptr)) {}
    bool valid() const { return static_cast<bool>(*this); }
    TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Ptr, FromJsonOptions, ToJsonOptions)

    /// Merges this codec spec with another codec spec.
    ///
    /// Modifies `*this` to refer to the merged codec spec.  Merging two null
    /// codec specs simply results in a null codec spec.  Merging a null and
    /// non-null codec spec simply preserves the non-null codec spec.
    absl::Status MergeFrom(Ptr other);

    /// Compares two codec specs by value (not by pointer).
    ///
    /// Null values are supported.
    friend bool operator==(const Ptr& a, const Ptr& b);
    friend bool operator!=(const Ptr& a, const Ptr& b) { return !(a == b); }

    /// Writes the JSON representation of `codec` to `os`.
    ///
    /// \param os Output stream.
    /// \param codec Codec to write, may be null.
    friend std::ostream& operator<<(std::ostream& os, const Ptr& codec);
  };

  /// Checks this codec spec for equality with another codec spec.
  ///
  /// By default, compares the JSON representations and returns false if
  /// conversion to JSON fails.
  ///
  /// \param other Another codec spec, not guaranteed to have the same dynamic
  ///     type.  In most cases, an error should be returned if the dynamic type
  ///     differs.
  virtual bool EqualTo(const CodecSpec& other) const;

  /// Merges another codec spec into this spec.
  ///
  /// If incompatible constraints are specified, an error should be returned.
  ///
  /// \param other_base Other codec spec, not guaranteed to have the same
  ///     dynamic type.  In most cases, an error should be returned if the
  ///     dynamic type differs.
  virtual absl::Status DoMergeFrom(const CodecSpec& other_base) = 0;

  /// Same as above, but handles null `other`.
  absl::Status MergeFrom(const Ptr& other);

  /// Merges two codec specs.
  static Result<Ptr> Merge(Ptr a, Ptr b);

  /// Returns a new copy of this codec spec.
  virtual Ptr Clone() const = 0;

  virtual ~CodecSpec();

  template <typename T>
  static PtrT<T> Make() {
    return PtrT<T>(new T);
  }
};

namespace internal {
struct CodecSpecPtrNonNullDirectSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const CodecSpec::Ptr& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   CodecSpec::Ptr& value);

  // Also support `IntrusivePtr<CodecSpec>` for use by the Python bindings.
  [[nodiscard]] static bool Encode(
      serialization::EncodeSink& sink,
      const internal::IntrusivePtr<CodecSpec>& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   internal::IntrusivePtr<CodecSpec>& value);
};
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(tensorstore::CodecSpec::Ptr)
TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal::IntrusivePtr<tensorstore::CodecSpec>)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::CodecSpec::Ptr)
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal::IntrusivePtr<tensorstore::CodecSpec>)

#endif  // TENSORSTORE_DRIVER_ENCODING_SPEC_H_
