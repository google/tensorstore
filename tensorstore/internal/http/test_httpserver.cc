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

#include "tensorstore/internal/http/test_httpserver.h"

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "re2/re2.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/os/subprocess.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

ABSL_FLAG(std::string, test_httpserver_binary, "",
          "Path to the test http server binary.");

using ::tensorstore::internal::SpawnSubprocess;
using ::tensorstore::internal::SubprocessOptions;
using ::tensorstore::internal_os::AwaitReadablePipe;
using ::tensorstore::internal_os::ReadFromFile;

namespace tensorstore {
namespace internal_http {

TestHttpServer::TestHttpServer() = default;

TestHttpServer::~TestHttpServer() {
  if (child_) {
    child_->Kill().IgnoreError();
    auto join_result = child_->Join();
    if (!join_result.ok()) {
      ABSL_LOG(ERROR) << "Joining test_httpserver subprocess failed: "
                      << join_result.status();
    }
  }
}

// Spawns the subprocess and sets the http address.
void TestHttpServer::SpawnProcess() {
  if (child_) return;

  root_path_ =
      internal::PathDirnameBasename(absl::GetFlag(FLAGS_test_httpserver_binary))
          .first;

  SubprocessOptions options{absl::GetFlag(FLAGS_test_httpserver_binary)};
  options.stdout_action = SubprocessOptions::Pipe{};

  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto spawn_proc, SpawnSubprocess(options));
  ABSL_CHECK(spawn_proc.stdout_pipe() !=
             internal_os::FileDescriptorTraits::Invalid());

  // Serving on ('127.0.0.1', 40807)
  static LazyRE2 kServingPattern = {
      "(?m)^Serving on (?:[(']*)([^:']+)[:',\\s]+(\\d+).*"};

  // Give the child process several seconds to start.
  auto deadline = absl::Now() + absl::Seconds(10);

  size_t offset = 0;
  char buf[1024];
  auto remaining = tensorstore::span(buf, sizeof(buf));
  while (!remaining.empty()) {
    // Wait until the child process is has more output.
    TENSORSTORE_CHECK_OK(AwaitReadablePipe(spawn_proc.stdout_pipe(), deadline));
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto n, ReadFromFile(spawn_proc.stdout_pipe(), remaining));
    offset += n;
    remaining.subspan(n);

    // Look for the serving line in the output.
    std::string_view host;
    std::string_view port;
    if (RE2::PartialMatch(std::string_view(buf, offset), *kServingPattern,
                          &host, &port)) {
      http_address_ = absl::StrFormat("%s:%s", host, port);
      ABSL_LOG(INFO) << "Serving on " << http_address_;
      break;
    }
  }

  // Check to see if the process has terminated; it should be running.
  auto join_result = spawn_proc.Join(/*block=*/false);
  if (http_address_.empty() || !absl::IsUnavailable(join_result.status())) {
    ABSL_LOG(FATAL) << "Failed to start process: " << join_result.status()
                    << " " << std::string_view(buf, offset);
  }

  child_.emplace(std::move(spawn_proc));
}

}  // namespace internal_http
}  // namespace tensorstore
