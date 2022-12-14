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

#ifndef TENSORSTORE_PROTO_PROTO_BINDER_H_
#define TENSORSTORE_PROTO_PROTO_BINDER_H_

#include <type_traits>

#include "absl/status/status.h"
#include "google/protobuf/message.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options_base.h"

namespace tensorstore {
namespace internal_json_binding {

struct JsonProtoBinderBase {
  absl::Status operator()(std::true_type /*is_loading*/,
                          const NoOptions& options, google::protobuf::Message* obj,
                          ::nlohmann::json* j) const;

  absl::Status operator()(std::false_type /*is_loading*/,
                          const NoOptions& options, const google::protobuf::Message* obj,
                          ::nlohmann::json* j) const;
};

struct AsciiProtoBinderBase {
  absl::Status operator()(std::true_type /*is_loading*/,
                          const NoOptions& options, google::protobuf::Message* obj,
                          ::nlohmann::json* j) const;

  absl::Status operator()(std::false_type /*is_loading*/,
                          const NoOptions& options, const google::protobuf::Message* obj,
                          ::nlohmann::json* j) const;
};

/// Parses a JSON-format protocol buffer into a Proto object
template <typename MessageType>
struct JsonProtoBinder : private JsonProtoBinderBase {
  inline absl::Status operator()(std::true_type /*is_loading*/,
                                 const NoOptions& options, MessageType* obj,
                                 ::nlohmann::json* j) const {
    return JsonProtoBinderBase::operator()(std::true_type{}, options, obj, j);
  }

  inline absl::Status operator()(std::false_type /*is_loading*/,
                                 const NoOptions& options,
                                 const MessageType* obj,
                                 ::nlohmann::json* j) const {
    return JsonProtoBinderBase::operator()(std::false_type{}, options, obj, j);
  }
};

/// Parses an ASCII-format protocol buffer into a Proto object.
template <typename MessageType>
struct AsciiProtoBinder : private AsciiProtoBinderBase {
  inline absl::Status operator()(std::true_type /*is_loading*/,
                                 const NoOptions& options, MessageType* obj,
                                 ::nlohmann::json* j) const {
    return AsciiProtoBinderBase::operator()(std::true_type{}, options, obj, j);
  }

  inline absl::Status operator()(std::false_type /*is_loading*/,
                                 const NoOptions& options,
                                 const MessageType* obj,
                                 ::nlohmann::json* j) const {
    return AsciiProtoBinderBase::operator()(std::false_type{}, options, obj, j);
  }
};

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_PROTO_BINDER_H_
