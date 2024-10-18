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
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace cli {

/// Flag representation after calling absl::ParseAbseilFlagsOnly.
struct CommandFlags {
  std::vector<char*> argv;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  std::vector<std::string_view> positional_args;
};

/// Representation of a "long" cli option--one which takes a value.
struct LongOption {
  // Long option name. Should begin with a "--" prefix.
  std::string_view longname;
  // Parsing function.  An error status indicates an invalid argument.
  std::function<absl::Status(std::string_view)> parse;
};

/// Representation of a "bool" cli option.
struct BoolOption {
  // Option name. May be a short name [-c] or may begin with a "--" prefix.
  std::string_view boolname;
  // Presence function.
  std::function<void()> found;
};

// Try to parse cli options.
absl::Status TryParseOptions(CommandFlags& flags,
                             tensorstore::span<LongOption> long_options,
                             tensorstore::span<BoolOption> bool_options);

// Convert a glob pattern to a regular expression.
std::string GlobToRegex(std::string_view glob);

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_ARGS_H_
