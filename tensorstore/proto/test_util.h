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

#ifndef TENSORSTORE_PROTO_TEST_UTIL_H_
#define TENSORSTORE_PROTO_TEST_UTIL_H_

#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/text_format.h"
#include "tensorstore/internal/logging.h"

namespace tensorstore {
// Error collector class for ProtoUtil::ParseFromString.
class TestUtilLogErrors : public google::protobuf::io::ErrorCollector {
 public:
  TestUtilLogErrors() = default;
  ~TestUtilLogErrors() override = default;

  // Indicates that there was an error in the input at the given line and
  // column numbers.  The numbers are zero-based, so we add 1 to them.
  void AddError(int line, google::protobuf::io::ColumnNumber column,
                const std::string& message) override {
    TENSORSTORE_LOG("Parse error (", line, ", ", column, "): ", message);
  }

  // Indicates that there was a warning in the input at the given line and
  // column numbers.  The numbers are zero-based, so we add 1 to them.
  void AddWarning(int line, google::protobuf::io::ColumnNumber column,
                  const std::string& message) override {
    TENSORSTORE_LOG("Parse error (", line, ", ", column, "): ", message);
  }

  void OutputToString(std::string* error_str) { *error_str = ""; }
};

template <typename Proto>
Proto ParseProtoOrDie(const std::string& asciipb) {
  Proto msg;
  TestUtilLogErrors log;
  google::protobuf::TextFormat::Parser parser;
  parser.AllowPartialMessage(true);
  parser.RecordErrorsTo(&log);

  if (!parser.ParseFromString(asciipb, &msg)) {
    TENSORSTORE_LOG_FATAL("Failed to parse proto:\n", asciipb);
  }
  return msg;
}

}  // namespace tensorstore

#endif  // TENSORSTORE_PROTO_TEST_UTIL_H_
