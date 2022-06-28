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

#ifndef TENSORSTORE_SERIALIZATION_BATCH_H_
#define TENSORSTORE_SERIALIZATION_BATCH_H_

/// \file
///
/// Facilities for serializing a sequence of objects to a single byte stream.
///
/// Shared pointers that are referenced multiple times are serialized only once,
/// similar to Python pickling.
///
/// This is used by tensorstore.distributed to serialize/deserialize task
/// results, and also for testing.

#include <memory>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace serialization {

/// EncodeSink that serializes objects to a single byte stream.
///
/// The same byte stream used for direct object encoding is also used for
/// indirect object encoding.
///
/// This is essentially the simplest possible `EncodeSink` implementation that
/// supports indirect object references.
class BatchEncodeSink final : public EncodeSink {
 public:
  explicit BatchEncodeSink(riegeli::Writer& writer);
  ~BatchEncodeSink();
  [[nodiscard]] bool DoIndirect(const std::type_info& type,
                                ErasedEncodeWrapperFunction encode,
                                std::shared_ptr<void> object) override;

 private:
  absl::flat_hash_map<std::shared_ptr<void>, size_t> indirect_map_;
};

/// `DecodeSource` for decoding the result of `BatchEncodeSink`.
class BatchDecodeSource final : public DecodeSource {
 public:
  BatchDecodeSource(riegeli::Reader& reader);
  ~BatchDecodeSource();

  [[nodiscard]] bool DoIndirect(const std::type_info& type,
                                ErasedDecodeWrapperFunction decode,
                                std::shared_ptr<void>& value) override;

 private:
  struct IndirectEntry {
    std::shared_ptr<void> value;
    const std::type_info* type;
  };
  std::vector<IndirectEntry> indirect_objects_;
};

/// Encodes a single object to a string.
template <typename T, typename ElementSerializer = Serializer<T>>
Result<std::string> EncodeBatch(const T& value,
                                const ElementSerializer& serializer = {}) {
  std::string buffer;
  riegeli::StringWriter writer(&buffer);
  BatchEncodeSink sink(writer);
  if (!serializer.Encode(sink, value) || !sink.Close()) {
    return sink.status();
  }
  return buffer;
}

/// Decodes a single object from a string.
template <typename T,
          typename ElementSerializer = Serializer<internal::remove_cvref_t<T>>>
absl::Status DecodeBatch(std::string_view encoded, T& value,
                         const ElementSerializer& serializer = {}) {
  riegeli::StringReader reader(encoded);
  BatchDecodeSource source(reader);
  if (!serializer.Decode(source, value)) {
    internal_serialization::FailEof(source);
  }
  return source.Done();
}

/// Overload that accepts rvalue reference, which is useful for mutable view
/// types like `span`.
template <typename T,
          typename ElementSerializer = Serializer<internal::remove_cvref_t<T>>>
absl::Status DecodeBatch(std::string_view encoded, T&& value,
                         const ElementSerializer& serializer = {}) {
  riegeli::StringReader reader(encoded);
  BatchDecodeSource source(reader);
  if (!serializer.Decode(source, value)) {
    internal_serialization::FailEof(source);
  }
  return source.Done();
}

template <typename T>
class MaybeDecode {
 public:
  absl::Status Decode(const std::string& arg) {
    return serialization::DecodeBatch(arg, value_);
  }
  const T& value() const { return value_; }
  T value_;
};

template <>
class MaybeDecode<std::string> {
 public:
  absl::Status Decode(const std::string& arg) {
    value_ = &arg;
    return absl::OkStatus();
  }
  const std::string& value() { return *value_; }
  const std::string* value_ = nullptr;
};

}  // namespace serialization
}  // namespace tensorstore

#endif  // TENSORSTORE_SERIALIZATION_BATCH_H_
