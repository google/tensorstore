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

namespace tensorstore {

/// Specifies compression and other encoding/decoding parameters for a
/// TensorStore driver.
///
/// This allows compression options to be specified as part of a `Schema` or
/// `SchemaConstraints` object independent of other driver parameters and
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

  virtual ~CodecSpec();

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

    /// Compares two codec specs by value.
    ///
    /// Null values are supported.
    ///
    /// Currently comparison is done by JSON representation.  If JSON conversion
    /// of either argument fails, the values are considered unequal.
    friend bool operator==(const Ptr& a, const Ptr& b);
    friend bool operator!=(const Ptr& a, const Ptr& b) { return !(a == b); }

    /// Writes the JSON representation of `codec` to `os`.
    ///
    /// \param os Output stream.
    /// \param codec Codec to write, may be null.
    friend std::ostream& operator<<(std::ostream& os, const Ptr& codec);
  };

  template <typename T>
  static PtrT<T> Make() {
    return PtrT<T>(new T);
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ENCODING_SPEC_H_
