// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_TSCLI_ARGS_H_
#define TENSORSTORE_TSCLI_ARGS_H_

#include <functional>
#include <ostream>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace cli {

/// Parser for command line options.
class CommandParser {
 public:
  using ParseLongOption = std::function<absl::Status(std::string_view)>;
  using ParseBoolOption = std::function<void()>;
  using ParsePositionalArg = std::function<absl::Status(std::string_view)>;

  CommandParser(std::string_view name, std::string_view description)
      : name_(name), description_(description) {};

  std::string_view name() const { return name_; }

  // Adds a function to consume long arguments.
  // Long options must begin with a "--" prefix and have the form
  // `--name=value`.
  void AddLongOption(std::string_view name, std::string_view description,
                     ParseLongOption fn);

  // Adds a function to consume boolean arguments.
  // Boolean options may begin with a "--" prefix or a "-" prefix.
  // If using a "-" prefix, the option must be a single letter.
  void AddBoolOption(std::string_view name, std::string_view description,
                     ParseBoolOption fn);

  // Adds a function to consume positional arguments.
  void AddPositionalArgs(std::string_view name, ParseLongOption fn);

  void PrintHelp(std::ostream& out);

  absl::Status TryParse(tensorstore::span<char*> argv,
                        tensorstore::span<char*> positional_args);

 private:
  // Representation of a "long" cli option--one which takes a value.
  struct LongOption {
    // Long option name. Should begin with a "--" prefix.
    std::string_view longname;
    // Description of the option.
    std::string_view description;
    // Parsing function.  An error status indicates an invalid argument.
    ParseLongOption parse;
  };

  // Representation of a "bool" cli option.
  struct BoolOption {
    // Option name. May be a short name [-c] or may begin with a "--" prefix.
    std::string_view boolname;
    // Description of the option.
    std::string_view description;
    // Presence function.
    ParseBoolOption found;
  };

  std::string_view short_description() const { return description_; }

  std::string_view name_;
  std::string_view description_;
  std::vector<LongOption> long_options_;
  std::vector<BoolOption> bool_options_;
  std::string_view positional_name_;
  ParsePositionalArg positional_fn_;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_ARGS_H_
