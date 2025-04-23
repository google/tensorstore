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

#include <algorithm>
#include <array>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"  // IWYU pragma: keep
#include "absl/base/no_destructor.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"     // IWYU pragma: keep
#include "absl/log/initialize.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/tscli/command.h"
#include "tensorstore/tscli/command_parser.h"
#include "tensorstore/tscli/copy_command.h"
#include "tensorstore/tscli/list_command.h"
#include "tensorstore/tscli/print_spec_command.h"
#include "tensorstore/tscli/print_stats_command.h"
#include "tensorstore/tscli/search_command.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/span.h"

using ::tensorstore::cli::Command;
using ::tensorstore::internal_metrics::FormatCollectedMetric;
using ::tensorstore::internal_metrics::GetMetricRegistry;

ABSL_FLAG(std::optional<std::string>, metrics_prefix, std::nullopt,
          "Prefix for metrics to output.");

ABSL_FLAG(tensorstore::JsonAbslFlag<tensorstore::Context::Spec>, context_spec,
          {},
          "Context spec for writing data.  This can be used to control the "
          "number of concurrent write operations of the underlying key-value "
          "store.  See examples at the start of the source file.");

namespace {

tensorstore::span<Command*> AllCommands() {
  static absl::NoDestructor<::tensorstore::cli::CopyCommand> copy;
  static absl::NoDestructor<::tensorstore::cli::ListCommand> list;
  static absl::NoDestructor<::tensorstore::cli::SearchCommand> search;
  static absl::NoDestructor<::tensorstore::cli::PrintSpecCommand> print_spec;
  static absl::NoDestructor<::tensorstore::cli::PrintStatsCommand> print_stats;

  static std::array<Command*, 5> commands{copy.get(), list.get(), search.get(),
                                          print_spec.get(), print_stats.get()};
  return commands;
}

void DumpMetrics(std::string_view prefix) {
  std::vector<std::string> lines;
  for (const auto& metric : GetMetricRegistry().CollectWithPrefix(prefix)) {
    FormatCollectedMetric(metric, [&lines](bool has_value, std::string line) {
      if (has_value) lines.emplace_back(std::move(line));
    });
  }

  // `lines` is unordered, which isn't great for benchmark comparison.
  std::sort(std::begin(lines), std::end(lines));
  std::cerr << std::endl;
  for (const auto& l : lines) {
    std::cerr << l << std::endl;
  }
  std::cerr << std::endl;
}

// Generate the program usage message.
void OutputUsageMessage(std::string_view program, std::ostream& out) {
  auto idx = program.find_last_of("/\\");
  if (idx != std::string_view::npos) program.remove_prefix(idx + 1);
  out << "Usage:\n  " << program
      << " [absl options...] <command> [command args]\n\n";
  out << "Commands:\n";
  for (const auto& command : AllCommands()) {
    command->parser().PrintHelp(std::cerr);
    out << "\n";
  }
}

int RealMain(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::vector<char*> positional_args;
  std::vector<absl::UnrecognizedFlag> unrecognized_flags;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args, unrecognized_flags);

  if (positional_args.size() < 2) {
    OutputUsageMessage(argv[0], std::cerr);
    return 1;
  }

  std::string_view command = positional_args[1];
  std::reverse(positional_args.begin(), positional_args.end());
  positional_args.pop_back();
  positional_args.pop_back();
  std::reverse(positional_args.begin(), positional_args.end());

  if (command == "help") {
    OutputUsageMessage(argv[0], std::cerr);
    return 0;
  }

  absl::Status status;
  for (const auto& x : AllCommands()) {
    if (!x->MatchesCommand(command)) continue;

    status = x->parser().TryParse(tensorstore::span<char*>(argv, argc),
                                  positional_args);
    if (status.ok()) {
      status = x->Run(absl::GetFlag(FLAGS_context_spec).value);
    }
    break;
  }

  if (absl::GetFlag(FLAGS_metrics_prefix).has_value()) {
    DumpMetrics(*absl::GetFlag(FLAGS_metrics_prefix));
  }

  if (status.ok()) return 0;

  std::cerr << status << std::endl;
  if (absl::IsInvalidArgument(status)) {
    OutputUsageMessage(argv[0], std::cerr);
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  // Run the main function.
  return RealMain(argc, argv);
}
