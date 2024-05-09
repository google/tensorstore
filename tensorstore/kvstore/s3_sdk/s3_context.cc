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
#include <ios>
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

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {

static constexpr char kAwsTag[] = "AWS";

absl::Mutex context_mu_;
std::weak_ptr<AwsContext> context_;

}  // namespace

/// Wraps a tensorstore HttpRequest in a Aws HttpRequest interface
class HttpRequestAdapter : public Aws::Http::HttpRequest {
public:
  ::tensorstore::internal_http::HttpRequest request_;
  Aws::Http::HeaderValueCollection headers_;
  Aws::IOStreamFactory stream_factory_;
  std::shared_ptr<Aws::IOStream> body_;

  HttpRequestAdapter(const Aws::Http::URI & uri, Aws::Http::HttpMethod method) :
    HttpRequest(uri, method),
    headers_(),
    stream_factory_(),
    body_(nullptr) {

      request_.method = Aws::Http::HttpMethodMapper::GetNameForHttpMethod(method);
      request_.url = uri.GetURIString(true);
  };

  virtual Aws::Http::HeaderValueCollection GetHeaders() const override {
    return headers_;
  }

  virtual const Aws::String & GetHeaderValue(const char* headerName) const override {
    auto it = headers_.find(headerName);
    assert(it != headers_.end());
    return it->second;
  }

  virtual bool HasHeader(const char* name) const override {
    return headers_.find(name) != headers_.end();
  }

  virtual void SetHeaderValue(const Aws::String& headerName, const Aws::String& headerValue) override {
    headers_.insert({headerName, headerValue});
  }

  virtual void SetHeaderValue(const char* headerName, const Aws::String& headerValue) override {
    headers_.insert({
      Aws::Utils::StringUtils::ToLower(headerName),
      Aws::Utils::StringUtils::Trim(headerValue.c_str())});
  }

  virtual void DeleteHeader(const char* headerName) override {
    if(auto it = headers_.find(Aws::Utils::StringUtils::ToLower(headerName)); it != headers_.end()) {
      headers_.erase(it);
    }
  }

  virtual int64_t GetSize() const override {
    return headers_.size();
  }

  virtual void AddContentBody(const std::shared_ptr<Aws::IOStream>& strContext) override {
    body_ = strContext;
  }

  virtual const std::shared_ptr<Aws::IOStream>& GetContentBody() const override {
    return body_;
  }

  virtual void SetResponseStreamFactory(const Aws::IOStreamFactory& streamFactory) override {
    stream_factory_ = streamFactory;
  }

  virtual const Aws::IOStreamFactory& GetResponseStreamFactory() const override {
    return stream_factory_;
  }
};

/// Wraps a tensorstore HttpResponse in an Aws HttpResponse interface
class HttpResponseAdapter: public Aws::Http::HttpResponse {
public:
  ::tensorstore::internal_http::HttpResponse response_;
  ::Aws::Utils::Stream::ResponseStream body_stream_;

  HttpResponseAdapter(
        ::tensorstore::internal_http::HttpResponse response,
        const std::shared_ptr<const Aws::Http::HttpRequest> & originatingRequest) :
      ::Aws::Http::HttpResponse(originatingRequest),
      response_(std::move(response)),
      body_stream_(originatingRequest->GetResponseStreamFactory()) {

      // Cast int response code to an HttpResponseCode enum
      // Potential for undefined behaviour here,
      // but AWS probably? won't respond with
      // a response code it doesn't know about
      SetResponseCode(static_cast<Aws::Http::HttpResponseCode>(response_.status_code));
  };

  virtual Aws::Utils::Stream::ResponseStream && SwapResponseStreamOwnership() override {
    return std::move(body_stream_);
  }

  virtual void AddHeader(const Aws::String& headerName, const Aws::String& headerValue) override {
    response_.headers.insert({headerName, headerValue});
  }

  virtual bool HasHeader(const char* headerName) const override {
    return response_.headers.find(Aws::Utils::StringUtils::ToLower(headerName)) != response_.headers.end();
  }

  virtual Aws::Http::HeaderValueCollection GetHeaders() const override {
    Aws::Http::HeaderValueCollection headers;
    for(const auto & header: response_.headers) {
      headers.insert({header.first, header.second});
    }
    return headers;
  }

  virtual const Aws::String & GetHeader(const Aws::String& headerName) const override {
    auto it = response_.headers.find(headerName);
    assert(it != response_.headers.end());
    return it->second;
  }

  virtual Aws::IOStream & GetResponseBody() const override {
    return body_stream_.GetUnderlyingStream();
  }
};


class CustomHttpClient : public Aws::Http::HttpClient {
public:
  std::shared_ptr<Aws::Http::HttpResponse> MakeRequest(
    const std::shared_ptr<Aws::Http::HttpRequest> & request,
    Aws::Utils::RateLimits::RateLimiterInterface* readLimiter = nullptr,
    Aws::Utils::RateLimits::RateLimiterInterface* writeLimiter = nullptr) const override {
      absl::Cord payload;
      if(auto req_adapter = std::dynamic_pointer_cast<HttpRequestAdapter>(request); req_adapter) {
        if(auto iostream = req_adapter->GetContentBody(); iostream) {
          // This is untested and probably broken
          // Ideally, we'd want a streambuf wrapping an underlying Cord
          // to avoid the copy here, especially for responses
          auto rdbuf = iostream->rdbuf();
          std::streamsize size = rdbuf->pubseekoff(0, iostream->end);
          auto cord_buffer = absl::CordBuffer::CreateWithDefaultLimit(size);
          absl::Span<char> data = cord_buffer.available_up_to(size);
          rdbuf->sgetn(data.data(), data.size());
          cord_buffer.IncreaseLengthBy(data.size());
          payload.Append(std::move(cord_buffer));
        }

        auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();
        ABSL_LOG(INFO) << req_adapter->request_;
        auto future = transport->IssueRequest(
          req_adapter->request_,
          ::tensorstore::internal_http::IssueRequestOptions(payload));
        // future.ExecuteWhenReady may be desirable
        auto response = future.value();
        ABSL_LOG(INFO) << response;
        return Aws::MakeShared<HttpResponseAdapter>(kAWSTag, response, request);
      }

      auto fail = Aws::MakeShared<Aws::Http::Standard::StandardHttpResponse>(kAwsTag, request);
      fail->SetResponseCode(Aws::Http::HttpResponseCode::PRECONDITION_FAILED);
      return fail;
    };
};


/// Custom factory overriding Aws::Http::DefaultHttpFatory
class CustomHttpFactory : public Aws::Http::HttpClientFactory {
public:
  std::shared_ptr<Aws::Http::HttpClient> CreateHttpClient(
    const Aws::Client::ClientConfiguration & clientConfiguration) const override {
      ABSL_LOG(INFO) << "Making a custom HTTP Client";
      return Aws::MakeShared<CustomHttpClient>(kAWSTag);
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
      auto request = Aws::MakeShared<HttpRequestAdapter>(kAWSTag, uri, method);
      request->SetResponseStreamFactory(streamFactory);
      return request;
  }
};

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
  auto level = Aws::Utils::Logging::LogLevel::Info;
  options.loggingOptions.logLevel = level;
  options.loggingOptions.logger_create_fn = [level=level]() {
    return Aws::MakeShared<AWSLogSystem>(kAWSTag, level);
  };

  ABSL_LOG(INFO) << "Initialising AWS SDK API";
  Aws::InitAPI(options);
  ABSL_LOG(INFO) << "Done Initialising AWS SDK API";

  auto provider = Aws::MakeShared<Aws::Auth::DefaultAWSCredentialsProviderChain>(kAWSTag);

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
