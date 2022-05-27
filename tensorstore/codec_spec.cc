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

#include "tensorstore/codec_spec.h"

#include <ostream>

#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/serialization/serialization.h"

namespace tensorstore {
namespace internal {

CodecSpecRegistry& GetCodecSpecRegistry() {
  static internal::NoDestructor<CodecSpecRegistry> registry;
  return *registry;
}

absl::Status CodecDriverSpec::MergeFrom(const CodecSpec& other) {
  if (!other) return absl::OkStatus();
  TENSORSTORE_RETURN_IF_ERROR(
      this->DoMergeFrom(*other),
      tensorstore::MaybeAnnotateStatus(
          _, tensorstore::StrCat("Cannot merge codec spec ", CodecSpec(this),
                                 " with ", other)));
  return absl::OkStatus();
}

bool CodecDriverSpec::EqualTo(const internal::CodecDriverSpec& other) const {
  auto a_json = CodecSpec(this).ToJson();
  auto b_json = CodecSpec(&other).ToJson();
  if (!a_json.ok() || !b_json.ok()) return false;
  return internal_json::JsonSame(*a_json, *b_json);
}

CodecDriverSpec::~CodecDriverSpec() = default;

}  // namespace internal

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(CodecSpec, [](auto is_loading,
                                                     const auto& options,
                                                     auto* obj, auto* j) {
  auto& registry = internal::GetCodecSpecRegistry();
  namespace jb = internal_json_binding;
  if constexpr (is_loading) {
    if (j->is_discarded()) {
      *obj = CodecSpec();
      return absl::OkStatus();
    }
  } else {
    if (!obj->valid()) {
      *j = ::nlohmann::json::value_t::discarded;
      return absl::OkStatus();
    }
  }
  return jb::Object(registry.MemberBinder("driver"))(is_loading, options, obj,
                                                     j);
})

bool operator==(const CodecSpec& a, const CodecSpec& b) {
  if (!a) return !b;
  if (!b) return false;
  return a->EqualTo(*b);
}

absl::Status CodecSpec::MergeFrom(CodecSpec other) {
  if (!other) {
    return absl::OkStatus();
  }
  if (!*this) {
    *this = std::move(other);
    return absl::OkStatus();
  }
  if (get()->use_count() != 1) {
    *this = get()->Clone();
  }
  return const_cast<internal::CodecDriverSpec&>(**this).MergeFrom(other);
}

Result<CodecSpec> CodecSpec::Merge(CodecSpec a, CodecSpec b) {
  TENSORSTORE_RETURN_IF_ERROR(a.MergeFrom(std::move(b)));
  return a;
}

std::ostream& operator<<(std::ostream& os, const CodecSpec& codec) {
  auto json_result = codec.ToJson();
  if (!json_result.ok()) return os << "<unprintable>";
  return os << json_result->dump();
}

namespace internal {

bool CodecSpecNonNullDirectSerializer::Encode(serialization::EncodeSink& sink,
                                              const CodecSpec& value) {
  assert(value);
  return serialization::JsonBindableSerializer<CodecSpec>::Encode(sink, value);
}

bool CodecSpecNonNullDirectSerializer::Encode(
    serialization::EncodeSink& sink,
    const internal::IntrusivePtr<internal::CodecDriverSpec>& value) {
  return Encode(sink, CodecSpec(value));
}

bool CodecSpecNonNullDirectSerializer::Decode(
    serialization::DecodeSource& source, CodecSpec& value) {
  if (!serialization::JsonBindableSerializer<CodecSpec>::Decode(source,
                                                                value)) {
    return false;
  }
  if (!value.valid()) {
    source.Fail(
        absl::DataLossError("Expected non-null internal::CodecDriverSpec"));
    return false;
  }
  return true;
}

bool CodecSpecNonNullDirectSerializer::Decode(
    serialization::DecodeSource& source,
    internal::IntrusivePtr<internal::CodecDriverSpec>& value) {
  CodecSpec temp;
  if (!Decode(source, temp)) return false;
  value =
      internal::const_pointer_cast<internal::CodecDriverSpec>(std::move(temp));
  return true;
}

}  // namespace internal

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::CodecSpec, tensorstore::serialization::JsonBindableSerializer<
                                tensorstore::CodecSpec>())
