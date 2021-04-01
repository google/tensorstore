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
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/no_destructor.h"

namespace tensorstore {
namespace internal {

CodecSpecRegistry& GetCodecSpecRegistry() {
  static internal::NoDestructor<CodecSpecRegistry> registry;
  return *registry;
}

}  // namespace internal

CodecSpec::~CodecSpec() = default;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(CodecSpec::Ptr, [](auto is_loading,
                                                          const auto& options,
                                                          auto* obj, auto* j) {
  auto& registry = internal::GetCodecSpecRegistry();
  namespace jb = internal_json_binding;
  if constexpr (is_loading) {
    if (j->is_discarded()) {
      *obj = CodecSpec::Ptr();
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

bool operator==(const CodecSpec::Ptr& a, const CodecSpec::Ptr& b) {
  auto a_json = a.ToJson();
  auto b_json = b.ToJson();
  if (!a_json.ok() || !b_json.ok()) return false;
  return internal_json::JsonSame(*a_json, *b_json);
}

std::ostream& operator<<(std::ostream& os, const CodecSpec::Ptr& codec) {
  auto json_result = codec.ToJson();
  if (!json_result.ok()) return os << "<unprintable>";
  return os << json_result->dump();
}

}  // namespace tensorstore
