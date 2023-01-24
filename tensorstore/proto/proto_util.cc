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

#include "tensorstore/proto/proto_util.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace {

// Error collector class for ProtoUtil::ParseFromString.
class ErrorCollector : public google::protobuf::io::ErrorCollector {
 public:
  ErrorCollector() = default;
  ~ErrorCollector() override = default;

  // Indicates that there was an error in the input at the given line and
  // column numbers.  The numbers are zero-based, so we add 1 to them.
    void AddError(int line, google::protobuf::io::ColumnNumber column,
                  const std::string& message) override {
    // Proto parsing uses a line of -1 to indicate errors not associated with a
    // specific line.
    errors.emplace_back(tensorstore::StrCat(
        "Line: ", std::max(1, line + 1), ", col: ", column + 1, ": ", message));
  }

    void AddWarning(int line, google::protobuf::io::ColumnNumber column,
                    const std::string& message) override {
    errors.emplace_back(tensorstore::StrCat(
        "Line: ", std::max(1, line + 1), ", col: ", column + 1, ": ", message));
  }

  std::vector<std::string> errors;
};

}  // namespace

bool TryParseTextProto(absl::string_view asciipb, google::protobuf::Message* msg,
                       std::vector<std::string>* errors,
                       bool allow_partial_messages,
                       bool allow_unknown_extensions) {
  google::protobuf::TextFormat::Parser parser;
  parser.AllowPartialMessage(allow_partial_messages);
  parser.AllowUnknownExtension(allow_unknown_extensions);
  ErrorCollector error_collector;
  parser.RecordErrorsTo(&error_collector);
  google::protobuf::io::ArrayInputStream asciipb_istream(asciipb.data(), asciipb.size());
  if (parser.Parse(&asciipb_istream, msg)) {
    return true;
  }

  msg->Clear();  // Always return an empty message upon error.

  if (errors) {
    *errors = std::move(error_collector.errors);
  }
  return false;
}

}  // namespace tensorstore
