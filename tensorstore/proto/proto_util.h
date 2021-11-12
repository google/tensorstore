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

#ifndef TENSORSTORE_PROTO_PROTO_UTIL_H_
#define TENSORSTORE_PROTO_PROTO_UTIL_H_

#include <string>
#include <string_view>
#include <vector>

#include "google/protobuf/message.h"

namespace tensorstore {

// Try parsing 'asciipb' as the type of 'msg' and return true on success.
// If parsing fails return false and if 'errors' is not NULL report all
// parsing errors.
bool TryParseTextProto(std::string_view asciipb, google::protobuf::Message* msg,
                       std::vector<std::string>* errors = nullptr,
                       bool allow_partial_messages = true,
                       bool allow_unknown_extensions = false);

}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_PROTO_UTIL_H_
