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

#include "tensorstore/kvstore/s3_sdk/s3_context.h"

#include <cstdarg>
#include <memory>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/core/utils/stream/ResponseStream.h>

#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/kvstore/s3_sdk/cord_streambuf.h"

using AwsHttpClient = ::Aws::Http::HttpClient;
using AwsHttpRequest = ::Aws::Http::HttpRequest;
using AwsHttpResponse = ::Aws::Http::HttpResponse;
using AwsStandardHttpRequest = ::Aws::Http::Standard::StandardHttpRequest;
using AwsStandardHttpResponse = ::Aws::Http::Standard::StandardHttpResponse;
using AwsRateLimiterInterface = ::Aws::Utils::RateLimits::RateLimiterInterface;
using AwsLogLevel = ::Aws::Utils::Logging::LogLevel;
using AwsLogSystemInterface = ::Aws::Utils::Logging::LogSystemInterface;

using ::Aws::Http::HttpMethodMapper::GetNameForHttpMethod;
using ::Aws::Auth::DefaultAWSCredentialsProviderChain;

using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;
using ::tensorstore::internal_http::IssueRequestOptions;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

static constexpr char kAwsTag[] = "AWS";
static constexpr char kUserAgentHeader[] = "user-agent";
static constexpr std::size_t k1MB = 1024 * 1024;

// Context guarded by mutex
absl::Mutex context_mu_;
std::weak_ptr<AwsContext> context_ ABSL_GUARDED_BY(context_mu_);

/// Provides a custom Aws HttpClient.
/// Overrides Aws::HttpClient::MakeRequest to convert AWS HttpRequests
/// into tensorstore HttpRequests which are issued on the default  tensorstore
/// HTTP transport. The returned tensorstore HttpResponse is
// converted into an AWS HttpResponse
class CustomHttpClient : public AwsHttpClient {
public:
  struct RequestAndPayload {
    HttpRequest request;
    absl::Cord cord;
  };

  // Converts an Aws StandardHttpRequest to a tensorstore HttpRequest
  RequestAndPayload FromAwsRequest(const std::shared_ptr<AwsHttpRequest> & aws_request) const {
    auto aws_headers = aws_request->GetHeaders();
    auto headers = std::vector<std::string>{};
    for(auto &[name, value]: aws_headers) {
      headers.emplace_back(absl::StrCat(name, ": ", value));
    }
    std::string user_agent;
    if(auto it = aws_headers.find(kUserAgentHeader); it != aws_headers.end()) {
      user_agent = it->second;
    }

    absl::Cord payload;

    // Get the underlying body as a Cord
    if (auto body = aws_request->GetContentBody(); body) {
      // Fast path, extract underlying Cord
      if (auto cordstreambuf = dynamic_cast<CordStreamBuf *>(body->rdbuf());
          cordstreambuf) {
        payload = cordstreambuf->DetachCord();
        // TODO: remove this
      } else {
        // Slow path, copy characters off the stream into Cord
        std::vector<char> buffer(absl::CordBuffer::kDefaultLimit);
        std::streampos original = body->tellg();
        while (body->read(buffer.data(), buffer.size()) || body->gcount() > 0) {
          payload.Append(absl::string_view(buffer.data(), body->gcount()));
        }

        if(payload.size() > k1MB) {
          ABSL_LOG(WARNING) << "Copied HttpRequest body of size " << payload.size() << " from iostream";
        }

        // Reset stream
        body->clear();
        body->seekg(original);
      }
    }

    return RequestAndPayload{
      HttpRequest{
        GetNameForHttpMethod(aws_request->GetMethod()),
        aws_request->GetURIString(true),
        std::move(user_agent),
        std::move(headers)},
      std::move(payload)
    };
  }

  // Converts a tensorstore response to an Aws StandardHttpResponse
  std::shared_ptr<AwsStandardHttpResponse> ToAwsResponse(
      HttpResponse & ts_response,
      const std::shared_ptr<AwsHttpRequest> & aws_request) const {

    auto aws_response = Aws::MakeShared<AwsStandardHttpResponse>(kAwsTag, aws_request);
    aws_response->SetResponseCode(static_cast<Aws::Http::HttpResponseCode>(ts_response.status_code));
    for(auto &[name, value]: aws_response->GetHeaders()) {
      aws_response->AddHeader(name, value);
    }

    // Move Cord into the body stream
    if(!ts_response.payload.empty()) {
      auto & body = aws_response->GetResponseBody();
      if(auto cordstreambuf = dynamic_cast<CordStreamBuf *>(body.rdbuf());
          cordstreambuf) {
        // Fast path, directly assign the Cord
        cordstreambuf->AssignCord(std::move(ts_response.payload));
      } else {
        if(ts_response.payload.size() > k1MB) {
          ABSL_LOG(WARNING) << "Copied HttpResponse body of size " << ts_response.payload.size() << " to iostream";
        }

        body << ts_response.payload;
      }
    }

    return aws_response;
  }

  /// Overrides the SDK mechanism for issuing AWS HttpRequests
  /// Converts AWS HttpRequests to their tensorstore requivalent,
  /// which is issued on the default tensorstore transport.
  /// The tensorstore response is converted into an AWS HttpResponse.
  std::shared_ptr<AwsHttpResponse> MakeRequest(
    const std::shared_ptr<AwsHttpRequest> & request,
    AwsRateLimiterInterface* readLimiter = nullptr,
    AwsRateLimiterInterface* writeLimiter = nullptr) const override {
      // Issue the wrapped HttpRequest on a tensorstore executor
      auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();
      auto [ts_request, payload] = FromAwsRequest(request);
      ABSL_LOG(INFO) << ts_request << " " << payload;
      auto future = transport->IssueRequest(ts_request, IssueRequestOptions(payload));
      // TODO: if possible use a continuation (future.ExecuteWhenReady) here
      auto response = future.value();
      ABSL_LOG(INFO) << response;
      return ToAwsResponse(response, request);
    };
};


/// Custom factory overriding Aws::Http::DefaultHttpFatory
/// Generates a CustomHttpClient (which defers to tensorflow's curl library)
/// as well as overriding CreateHttpRequest to return
/// Standard Http Requests
class CustomHttpFactory : public Aws::Http::HttpClientFactory {
public:
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
    const Aws::Client::ClientConfiguration & clientConfiguration) const override {
      ABSL_LOG(INFO) << "Constructing custom HTTP Client";
      return Aws::MakeShared<CustomHttpClient>(kAwsTag);
  };

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
    const Aws::String &uri, Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory &streamFactory) const override {
      ABSL_LOG(INFO) << "Constructing custom HttpRequest";
      return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
    const Aws::Http::URI& uri, Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory& streamFactory) const override
  {
    ABSL_LOG(INFO) << "Constructing custom HttpRequest";
    auto request = Aws::MakeShared<AwsStandardHttpRequest>(kAwsTag, uri, method);
    request->SetResponseStreamFactory(streamFactory);
    return request;
  }
};


/// Connect the AWS SDK's logging system to Abseil logging
class AWSLogSystem : public AwsLogSystemInterface {
public:
  AWSLogSystem(AwsLogLevel log_level) : log_level_(log_level) {};
  AwsLogLevel GetLogLevel(void) const override {
    return log_level_;
  };

  // Writes the stream to ProcessFormattedStatement.
  void LogStream(AwsLogLevel log_level, const char* tag,
                 const Aws::OStringStream& messageStream) override {
    LogMessage(log_level, messageStream.rdbuf()->str().c_str());
  }

  // Flushes the buffered messages if the logger supports buffering
  void Flush() override { return; };

  // Overridden, but prefer the safer LogStream
  void Log(AwsLogLevel log_level, const char* tag,
           const char* format, ...) override;

private:
  void LogMessage(AwsLogLevel log_level, const std::string & message);
  AwsLogLevel log_level_;
};


void AWSLogSystem::Log(AwsLogLevel log_level, const char* tag,
           const char* format, ...) {
  char buffer[256];
  va_list args;
  va_start(args, format);
  vsnprintf(buffer, 256, format, args);
  va_end(args);
  LogMessage(log_level, buffer);
}

void AWSLogSystem::LogMessage(AwsLogLevel log_level, const std::string & message) {
  switch(log_level) {
    case AwsLogLevel::Info:
      ABSL_LOG(INFO) << message;
      break;
    case AwsLogLevel::Warn:
      ABSL_LOG(WARNING) << message;
      break;
    case AwsLogLevel::Error:
      ABSL_LOG(ERROR) << message;
      break;
    case AwsLogLevel::Fatal:
      ABSL_LOG(FATAL) << message;
      break;
    case AwsLogLevel::Trace:
    case AwsLogLevel::Debug:
    default:
      ABSL_LOG(INFO) << message;
      break;
  }
}

}  // namespace

Aws::IOStream * CordBackedResponseStreamFactory() {
  return Aws::New<Aws::Utils::Stream::DefaultUnderlyingStream>(
    kAwsTag, Aws::MakeUnique<CordStreamBuf>(kAwsTag));
}

// Initialise AWS API and Logging
std::shared_ptr<AwsContext> GetAwsContext() {
  absl::MutexLock lock(&context_mu_);
  if(context_.use_count() > 0) {
    ABSL_LOG(INFO) << "Returning existing AwsContext";
    return context_.lock();
  }

  auto options = Aws::SDKOptions{};
  // Customise HttpClientFactory
  // Disable curl init/cleanup, tensorstore should control this
  // Don't install the SIGPIPE handler
  options.httpOptions.httpClientFactory_create_fn = []() {
    return Aws::MakeShared<CustomHttpFactory>(kAwsTag);
  };
  options.httpOptions.initAndCleanupCurl = false;
  options.httpOptions.installSigPipeHandler = false;

  // Install AWS -> Abseil Logging Translator
  //auto level = AwsLogLevel::Debug;
  auto level = AwsLogLevel::Info;
  options.loggingOptions.logLevel = level;
  options.loggingOptions.logger_create_fn = [level=level]() {
    return Aws::MakeShared<AWSLogSystem>(kAwsTag, level);
  };

  ABSL_LOG(INFO) << "Initialising AWS SDK API";
  Aws::InitAPI(options);
  ABSL_LOG(INFO) << "AWS SDK API Initialised";

  auto provider = Aws::MakeShared<DefaultAWSCredentialsProviderChain>(kAwsTag);

  auto ctx = std::shared_ptr<AwsContext>(
    new AwsContext{
        std::move(options),
        std::move(provider)},
      [](AwsContext * ctx) {
        absl::MutexLock lock(&context_mu_);
        ABSL_LOG(INFO) << "Shutting down AWS SDK API";
        Aws::ShutdownAPI(ctx->options);
        ABSL_LOG(INFO) << "AWS SDK API Shutdown";
        delete ctx;
    });
  context_ = ctx;
  return ctx;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
