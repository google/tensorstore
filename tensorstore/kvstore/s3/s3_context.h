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

#ifndef TENSORSTORE_KVSTORE_S3_S3_CONTEXT_H_
#define TENSORSTORE_KVSTORE_S3_S3_CONTEXT_H_

#include <memory>
#include <string>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>


namespace tensorstore {
namespace internal_kvstore_s3 {

static constexpr char kAWSTag[] = "AWS";

class AWSLogSystem : public Aws::Utils::Logging::LogSystemInterface {
public:
  AWSLogSystem(Aws::Utils::Logging::LogLevel log_level);
  Aws::Utils::Logging::LogLevel GetLogLevel(void) const override;
  void SetLogLevel(Aws::Utils::Logging::LogLevel log_level);

  // Writes the stream to ProcessFormattedStatement.
  void LogStream(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                 const Aws::OStringStream& messageStream) override;

  // Flushes the buffered messages if the logger supports buffering
  void Flush() override { return; };

  // Overridden, but prefer the safer LogStream
  void Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
           const char* format, ...) override;

private:
  void LogMessage(Aws::Utils::Logging::LogLevel log_level, const std::string & message);
  Aws::Utils::Logging::LogLevel log_level_;
};


struct AwsContext {
  Aws::SDKOptions options;
  std::shared_ptr<Aws::Utils::Logging::LogSystemInterface> log_system_;
  std::shared_ptr<Aws::Auth::AWSCredentialsProvider> cred_provider_;
};

// Initialise AWS API and Logging
std::shared_ptr<AwsContext> GetAwsContext();


}  // namespace internal_kvstore_s3
}  // neamespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_S3_CONTEXT_H_