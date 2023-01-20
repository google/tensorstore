// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_PROTO_PARSE_TEXT_PROTO_OR_DIE_H_
#define TENSORSTORE_PROTO_PARSE_TEXT_PROTO_OR_DIE_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/strings/str_join.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/proto/proto_util.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

/// Parse text as a protocol buffer or dies.
/// Only to be used in unit tests.
///
/// Example:
///    MyMessage msg = ParseTextProtoOrDie(my_text_proto);
///
class ParseTextProtoOrDie {
 public:
  ParseTextProtoOrDie(std::string text_proto,
                      SourceLocation loc TENSORSTORE_LOC_CURRENT_DEFAULT_ARG)
      : text_proto_(std::move(text_proto)), loc_(std::move(loc)) {}

  template <class T>
  operator T() {
    T message;
    std::vector<std::string> errors;

    if (!TryParseTextProto(text_proto_, &message)) {
      ABSL_LOG(INFO).AtLocation(loc_.file_name(), loc_.line())
          << "Failed to parse " << message.GetTypeName() << " from textproto:\n"
          << text_proto_ << "\nWith errors: " << absl::StrJoin(errors, "\n");
    }
    return message;
  }

 private:
  std::string text_proto_;
  SourceLocation loc_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_PARSE_TEXT_PROTO_OR_DIE_H_
