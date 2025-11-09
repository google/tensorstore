// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_TSCLI_COMMAND_H_
#define TENSORSTORE_TSCLI_COMMAND_H_

#include <algorithm>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/tscli/command_parser.h"

namespace tensorstore {
namespace cli {

// Command encapsulates the option parsing and execution logic for a
// single command.
class Command {
 public:
  // Constructs a command with the given `name` and `usage`.  The name
  // is what the user types on the command line, and is later
  // available through the `name()` method.  The `usage` text is free
  // form English text describing the command.
  Command(std::string_view name, std::string_view short_description);

  Command(const Command&) = delete;
  Command& operator=(const Command&) = delete;
  virtual ~Command() = default;

  // Name of the command.
  std::string_view name() const { return parser().name(); }

  // Returns whether the command matches the name.
  bool MatchesCommand(std::string_view command) const {
    if (name() == command) return true;
    return std::find(aliases_.begin(), aliases_.end(), command) !=
           aliases_.end();
  }

  // Executes the command, returning a `Status` indicating OK or the
  // nature of the failure.  The `Command` will log any error
  // returned.  Derived classes must implement this.
  //
  // TODO: Add RunOptions struct to allow for more flexibility in how the
  // command is run.
  virtual absl::Status Run(Context::Spec context_spec) = 0;

  CommandParser& parser() { return parser_; }
  const CommandParser& parser() const { return parser_; }

 protected:
  void AddAlias(std::string_view alias) { aliases_.push_back(alias); }

 private:
  CommandParser parser_;
  std::vector<std::string_view> aliases_;
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_COMMAND_H_
