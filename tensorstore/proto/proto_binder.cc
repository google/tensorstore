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

#include "tensorstore/proto/proto_binder.h"

#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/json_util.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

absl::Status JsonProtoBinderBase::operator()(std::true_type /*is_loading*/,
                                             const NoOptions& options,
                                             google::protobuf::Message* obj,
                                             ::nlohmann::json* j) const {
  if (!j->template get_ptr<::nlohmann::json::object_t*>()) {
    return internal_json::ExpectedError(*j, "object");
  }
  std::string json_ascii = j->dump();
  auto status = google::protobuf::util::JsonStringToMessage(json_ascii, obj);
  if (status.ok()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Expected JSON protocol buffer ", obj->GetDescriptor()->name(),
      " object, but received ", j->dump(), "  with error ",
      std::string_view(status.message().data(), status.message().size())));
}

absl::Status JsonProtoBinderBase::operator()(std::false_type /*is_loading*/,
                                             const NoOptions& options,
                                             const google::protobuf::Message* obj,
                                             ::nlohmann::json* j) const {
  std::string json_ascii;
  auto status = google::protobuf::util::MessageToJsonString(*obj, &json_ascii);
  if (!status.ok()) {
    return absl::InternalError(
        std::string_view(status.message().data(), status.message().size()));
  }
  auto j_parse = ::nlohmann::json::parse(json_ascii, nullptr, false);
  if (j_parse.template get_ptr<::nlohmann::json::object_t*>()) {
    *j = std::move(j_parse);
    return absl::OkStatus();
  }
  return absl::InternalError("Failed to serialize field as JSON proto");
}

absl::Status AsciiProtoBinderBase::operator()(std::true_type,
                                              const NoOptions& options,
                                              google::protobuf::Message* obj,
                                              ::nlohmann::json* j) const {
  auto* str = j->template get_ptr<const std::string*>();
  if (!str) {
    return internal_json::ExpectedError(*j, "string");
  }
  if (TryParseTextProto(*str, obj)) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Expected ASCII protocol buffer ", obj->GetDescriptor()->name(),
      " object, but received ", *str));
}

absl::Status AsciiProtoBinderBase::operator()(std::false_type,
                                              const NoOptions& options,
                                              const google::protobuf::Message* obj,
                                              ::nlohmann::json* j) const {
  std::string obj_text;
  google::protobuf::TextFormat::PrintToString(*obj, &obj_text);
  *j = obj_text;
  return absl::OkStatus();
}

}  // namespace internal_json_binding
}  // namespace tensorstore
