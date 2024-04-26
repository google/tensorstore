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

#include "tensorstore/kvstore/s3/s3_context.h"

#include <cstdarg>
#include <memory>

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>

#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

absl::Mutex context_mu_;
std::weak_ptr<AwsContext> context_;

}  // namespace

AWSLogSystem::AWSLogSystem(Aws::Utils::Logging::LogLevel log_level)
  : log_level_(log_level) {}

Aws::Utils::Logging::LogLevel AWSLogSystem::GetLogLevel(void) const {
  return log_level_;
}

void AWSLogSystem::SetLogLevel(Aws::Utils::Logging::LogLevel log_level) {
  log_level_ = log_level;
}

  // Writes the stream to ProcessFormattedStatement.
void AWSLogSystem::LogStream(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                 const Aws::OStringStream& messageStream) {
  LogMessage(log_level, messageStream.rdbuf()->str().c_str());
}

void AWSLogSystem::Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
           const char* format, ...) {
  char buffer[256];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, 256, format, args);
  va_end(args);
  LogMessage(log_level, buffer);
}

void AWSLogSystem::LogMessage(Aws::Utils::Logging::LogLevel log_level, const std::string & message) {
  switch(log_level) {
    case Aws::Utils::Logging::LogLevel::Info:
      ABSL_LOG(INFO) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Warn:
      ABSL_LOG(WARNING) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Error:
      ABSL_LOG(ERROR) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Fatal:
      ABSL_LOG(FATAL) << message;
      break;
    case Aws::Utils::Logging::LogLevel::Trace:
    case Aws::Utils::Logging::LogLevel::Debug:
    default:
      ABSL_LOG(INFO) << message;
      break;
  }
}


// Initialise AWS API and Logging
std::shared_ptr<AwsContext> GetAwsContext() {
  absl::MutexLock lock(&context_mu_);
  if(context_.use_count() > 0) {
    ABSL_LOG(INFO) << "Returning existing AwsContext";
    return context_.lock();
  }

  ABSL_LOG(INFO) << "Initialising AWS API";
  auto options = Aws::SDKOptions{};
  Aws::InitAPI(options);
  auto log = Aws::MakeShared<AWSLogSystem>(kAWSTag, Aws::Utils::Logging::LogLevel::Info);
  Aws::Utils::Logging::InitializeAWSLogging(std::move(log));
  auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>(kAWSTag);

  auto ctx = std::shared_ptr<AwsContext>(
    new AwsContext{
        std::move(options),
        std::move(log),
        std::move(provider)},
      [](AwsContext * ctx) {
        absl::MutexLock lock(&context_mu_);
        ABSL_LOG(INFO) << "Shutting down AWS API";
        Aws::Utils::Logging::ShutdownAWSLogging();
        Aws::ShutdownAPI(ctx->options);
        delete ctx;
    });
  context_ = ctx;
  return ctx;
}

}  // namespace internal_kvstore_s3
}  // neamespace tensorstore
