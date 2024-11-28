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
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/log_severity.h"  // IWYU pragma: keep
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"     // IWYU pragma: keep
#include "absl/log/initialize.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "absl/flags/parse.h"
#include "tensorstore/internal/metrics/collect.h"
#include "tensorstore/internal/metrics/registry.h"
#include "tensorstore/tscli/args.h"
#include "tensorstore/tscli/cli.h"
#include "tensorstore/util/json_absl_flag.h"

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

const char kUsage[] =
    "Usage:\n"
    "  tscli [global options..] <command> [command args]\n\n"
    "Commands:\n"
    "  copy\n"
    "      Copies keys from one kvstore to another.\n"
    "  copy --source <source-kvstore-spec> --target <target-kvstore-spec>\n"
    "\n"
    "  list [-h/--human][-b/--brief]\n"
    "      List keys in a kvstore.\n"
    "  list --source <source-kvstore-spec> [glob...]\n"
    "  list [spec...]\n"
    "\n"
    "  search [-b/--brief][-f/--full]\n"
    "      Search for tensorstores in a kvstore.\n"
    "  search --source=<source-kvstore-spec> \n"
    "  search [spec...]\n"
    "\n"
    "  print_spec [--include_defaults]\n"
    "      Print the spec for a tensorstore.\n"
    "  print --spec=<tensorstore-spec>\n"
    "  print [spec...]\n"
    "\n"
    "  print_stats [-b/--brief][-f/--full]\n"
    "      Print storage statistics for a tensorstore.\n"
    "  print_stats --spec=<tensorstore-spec> [box...]\n"
    "  print_stats [spec...]\n"
    "\n"
    "  help\n";

void PrintShortHelp(std::string_view program) {
  std::cerr << program << "\n" << kUsage << std::endl;
}

void PrintHelp(std::string_view program) {
  std::cerr << program << "\n" << kUsage << std::endl;
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

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  ::tensorstore::cli::CommandFlags flags;
  flags.argv = std::vector<char*>(argv, argv + argc);

  std::vector<char*> positional_args;
  absl::ParseAbseilFlagsOnly(argc, argv, positional_args,
                             flags.unrecognized_flags);

  if (positional_args.size() < 2) {
    PrintShortHelp(argv[0]);
    return 1;
  }
  std::string_view command = positional_args[1];
  flags.positional_args.assign(positional_args.begin() + 2,
                               positional_args.end());

  if (command == "help") {
    PrintHelp(argv[0]);
    return 0;
  }

  absl::Status status;
  if (command == "copy") {
    status = ::tensorstore::cli::RunKvstoreCopy(
        absl::GetFlag(FLAGS_context_spec).value, std::move(flags));
  } else if (command == "ls" || command == "list") {
    status = ::tensorstore::cli::RunKvstoreList(
        absl::GetFlag(FLAGS_context_spec).value, std::move(flags));
  } else if (command == "search") {
    status = ::tensorstore::cli::RunTsSearch(
        absl::GetFlag(FLAGS_context_spec).value, std::move(flags));
  } else if (command == "print_spec") {
    status = ::tensorstore::cli::RunTsPrintSpec(
        absl::GetFlag(FLAGS_context_spec).value, std::move(flags));
  } else if (command == "print_stats") {
    status = ::tensorstore::cli::RunTsPrintStorageStatistics(
        absl::GetFlag(FLAGS_context_spec).value, std::move(flags));
  }

  if (absl::GetFlag(FLAGS_metrics_prefix).has_value()) {
    DumpMetrics(*absl::GetFlag(FLAGS_metrics_prefix));
  }

  if (status.ok()) return 0;

  std::cerr << status << std::endl;
  if (absl::IsInvalidArgument(status)) {
    PrintHelp(argv[0]);
  }
  return 1;
}
