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
#include <aws/core/http/HttpClient.h>
#include <aws/core/http/standard/StandardHttpRequest.h>
#include <aws/core/http/standard/StandardHttpResponse.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/LogSystemInterface.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/core/utils/StringUtils.h>

#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"

#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"

using ::tensorstore::internal_http::IssueRequestOptions;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

static constexpr char kAwsTag[] = "AWS";

// Context guarded by mutex
absl::Mutex context_mu_;
std::weak_ptr<AwsContext> context_ ABSL_GUARDED_BY(context_mu_);

/// Wraps a tensorstore HttpRequest in a Aws HttpRequest interface
class HttpRequestWrapper : public Aws::Http::Standard::StandardHttpRequest {
public:
  ::tensorstore::internal_http::HttpRequest request_;
  absl::Cord payload_;

  HttpRequestWrapper(const Aws::Http::URI & uri, Aws::Http::HttpMethod method) :
    StandardHttpRequest(uri, method),
    payload_{} {

      request_.method = Aws::Http::HttpMethodMapper::GetNameForHttpMethod(method);
      request_.url = uri.GetURIString(true);  // include the query string
  };

  virtual void SetHeaderValue(const Aws::String& headerName, const Aws::String& headerValue) override {
    request_.headers.push_back(absl::StrCat(headerName, ": ", headerValue));
    StandardHttpRequest::SetHeaderValue(headerName, headerValue);
  }

  virtual void SetHeaderValue(const char* headerName, const Aws::String& headerValue) override {
    request_.headers.push_back(absl::StrCat(headerName, ": ", headerValue));
    StandardHttpRequest::SetHeaderValue(headerName, headerValue);
  }

  virtual void AddContentBody(const std::shared_ptr<Aws::IOStream>& strContext) override {
    StandardHttpRequest::AddContentBody(strContext);
    if(!strContext) {
      return;
    }

    // Copy characters off the stream into the Cord
    // TODO: This is impractical for large data and
    // should be mitigated by an iostream backed by a Cord

    // Remember the current position in the stream
    std::streampos original = strContext->tellg();
    const size_t bufferSize = 1024*1024;
    std::vector<char> buffer(bufferSize);

    while (strContext->read(buffer.data(), buffer.size()) || strContext->gcount() > 0) {
        payload_.Append(absl::Cord(absl::string_view(buffer.data(), strContext->gcount())));
    }

    strContext->clear();
    strContext->seekg(original);

    ABSL_LOG(INFO) << "AddContentBody " << payload_.size();
  }
};

/// Wraps a tensorstore HttpResponse in an Aws HttpResponse interface
class HttpResponseWrapper: public Aws::Http::Standard::StandardHttpResponse {
public:
  ::tensorstore::internal_http::HttpResponse response_;

  HttpResponseWrapper(
        ::tensorstore::internal_http::HttpResponse response,
        const std::shared_ptr<const Aws::Http::HttpRequest> & originatingRequest) :
      ::Aws::Http::Standard::StandardHttpResponse(originatingRequest),
      response_(std::move(response)) {

      // Cast int response code to an HttpResponseCode enum
      // Potential for undefined behaviour here,
      // but AWS probably? won't respond with
      // a response code it doesn't know about
      SetResponseCode(static_cast<Aws::Http::HttpResponseCode>(response_.status_code));

      // TODO
      // Add the payload to the Response Body if present
      // This incurs a copy, which should be avoided by subclassing
      // Aws::IOStream
      if(!response_.payload.empty()) {
        GetResponseBody() << response_.payload;
      }
  };

  virtual void AddHeader(const Aws::String& headerName, const Aws::String& headerValue) override {
    StandardHttpResponse::AddHeader(headerName, headerValue);
    response_.headers.insert({headerName, headerValue});
  }
};


/// Provides a custom Aws HttpClient.
/// Overrides the Aws::HttpClient::MakeRequest to accept HttpRequestWrappers
/// (produce by CustomHttpFactory below), issue a tensorstore HttpRequest,
/// receive a tensorstore HttpResponse to be wrapped in a HttpResponseWrapper
class CustomHttpClient : public Aws::Http::HttpClient {
public:
  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
    const std::shared_ptr<Aws::Http::HttpRequest> & request,
    Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
    Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
      if(auto req_adapter = std::dynamic_pointer_cast<HttpRequestWrapper>(request); req_adapter) {
        // Issue the wrapped HttpRequest on a tensorstore executor
        auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();
        ABSL_LOG(INFO) << req_adapter->request_ << " " << req_adapter->payload_;
        auto req_options = req_adapter->payload_.empty() ?
          IssueRequestOptions{} :
          IssueRequestOptions(std::move(req_adapter->payload_));
        auto future = transport->IssueRequest(
          req_adapter->request_, std::move(req_options));
        // TODO
        // Figure out how to use a continuation future.ExecuteWhenReady here
        auto response = future.value();
        ABSL_LOG(INFO) << response;
        return Aws::MakeShared<HttpResponseWrapper>(kAwsTag, response, request);
      }

      auto fail = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>(kAwsTag, request);
      fail->SetResponseCode(Aws::Http::HttpResponseCode::PRECONDITION_FAILED);
      return fail;
    };
};


/// Custom factory overriding Aws::Http::DefaultHttpFatory
/// Generates a CustomHttpClient (which defers to tensorflow's curl library)
/// as well as overriding Createhttp Request to return
/// HttpRequestWrappers
class CustomHttpFactory : public Aws::Http::HttpClientFactory {
public:
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
    const Aws::Client::ClientConfiguration & clientConfiguration) const override {
      ABSL_LOG(INFO) << "Making a custom HTTP Client";
      return Aws::MakeShared<CustomHttpClient>(kAwsTag);
  };

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
    const Aws::String &uri, Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory &streamFactory) const override {
      return CreateHttpRequest(Aws::Http::URI(uri), method, streamFactory);
  }

  std::shared_ptr<Aws::Http::HttpRequest> CreateHttpRequest(
    const Aws::Http::URI& uri, Aws::Http::HttpMethod method,
    const Aws::IOStreamFactory& streamFactory) const override
  {
      auto request = Aws::MakeShared<HttpRequestWrapper>(kAwsTag, uri, method);
      request->SetResponseStreamFactory(streamFactory);
      return request;
  }
};


/// Connect the AWS SDK's logging system to Abseil logging
class AWSLogSystem : public Aws::Utils::Logging::LogSystemInterface {
public:
  AWSLogSystem(Aws::Utils::Logging::LogLevel log_level) : log_level_(log_level) {};
  Aws::Utils::Logging::LogLevel GetLogLevel(void) const override {
    return log_level_;
  };

  // Writes the stream to ProcessFormattedStatement.
  void LogStream(Aws::Utils::Logging::LogLevel log_level, const char* tag,
                 const Aws::OStringStream& messageStream) override {
    LogMessage(log_level, messageStream.rdbuf()->str().c_str());
  }

  // Flushes the buffered messages if the logger supports buffering
  void Flush() override { return; };

  // Overridden, but prefer the safer LogStream
  void Log(Aws::Utils::Logging::LogLevel log_level, const char* tag,
           const char* format, ...) override;

private:
  void LogMessage(Aws::Utils::Logging::LogLevel log_level, const std::string & message);
  Aws::Utils::Logging::LogLevel log_level_;
};


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

}  // namespace

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
  //auto level = Aws::Utils::Logging::LogLevel::Debug;
  auto level = Aws::Utils::Logging::LogLevel::Info;
  options.loggingOptions.logLevel = level;
  options.loggingOptions.logger_create_fn = [level=level]() {
    return Aws::MakeShared<AWSLogSystem>(kAwsTag, level);
  };

  ABSL_LOG(INFO) << "Initialising AWS SDK API";
  Aws::InitAPI(options);
  ABSL_LOG(INFO) << "Done Initialising AWS SDK API";

  auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>(kAwsTag);

  auto ctx = std::shared_ptr<AwsContext>(
    new AwsContext{
        std::move(options),
        std::move(provider)},
      [](AwsContext * ctx) {
        absl::MutexLock lock(&context_mu_);
        ABSL_LOG(INFO) << "Shutting down AWS SDK API";
        Aws::ShutdownAPI(ctx->options);
        delete ctx;
    });
  context_ = ctx;
  return ctx;
}

}  // namespace internal_kvstore_s3
}  // neamespace tensorstore
