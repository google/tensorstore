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

absl::Mutex context_mu_;
std::weak_ptr<AwsContext> context_;

/// Wraps a tensorstore HttpRequest in a Aws HttpRequest interface
class HttpRequestAdapter : public Aws::Http::HttpRequest {
public:
  ::tensorstore::internal_http::HttpRequest request_;
  Aws::Http::HeaderValueCollection headers_;
  Aws::IOStreamFactory stream_factory_;
  std::shared_ptr<Aws::IOStream> body_;
  absl::Cord payload_;

  HttpRequestAdapter(const Aws::Http::URI & uri, Aws::Http::HttpMethod method) :
    HttpRequest(uri, method),
    headers_{},
    stream_factory_{},
    body_(nullptr),
    payload_{} {

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
    // ABSL_LOG(INFO) << "Setting header " << headerName << " " << headerValue;
    request_.headers.push_back(absl::StrCat(headerName, ": ", headerValue));
    headers_.insert({std::move(headerName), std::move(headerValue)});
  }

  virtual void SetHeaderValue(const char* headerName, const Aws::String& headerValue) override {
    auto lower_name = Aws::Utils::StringUtils::ToLower(headerName);
    auto trimmed_value = Aws::Utils::StringUtils::Trim(headerValue.c_str());
    SetHeaderValue(lower_name, trimmed_value);
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

    //ABSL_LOG(INFO) << "AddContentBody " << strContext << " " << body_;;

    if(!body_) {
      return;
    }

    const size_t bufferSize = 4096;
    std::vector<char> buffer(bufferSize);

    while (body_->read(buffer.data(), buffer.size()) || body_->gcount() > 0) {
        payload_.Append(absl::Cord(absl::string_view(buffer.data(), body_->gcount())));
    }
    ABSL_LOG(INFO) << "AddContentBody " << payload_.size();
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

      // Add the payload to the Response Body is present
      // This incurs a copy, which should be avoided by subclassing
      // Aws::IOStream
      if(!response_.payload.empty()) {
        GetResponseBody() << response_.payload;
      }
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
      ABSL_LOG(INFO) << "Making a request ";
      if(auto req_adapter = std::dynamic_pointer_cast<HttpRequestAdapter>(request); req_adapter) {
        auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();
        ABSL_LOG(INFO) << req_adapter->request_ << " " << req_adapter->payload_;
        auto req_options = req_adapter->payload_.empty() ?
          IssueRequestOptions{} :
          IssueRequestOptions(std::move(req_adapter->payload_));
        auto future = transport->IssueRequest(
          req_adapter->request_, std::move(req_options));
        // future.ExecuteWhenReady is desirable
        auto response = future.value();
        ABSL_LOG(INFO) << response;
        return Aws::MakeShared<HttpResponseAdapter>(kAwsTag, response, request);
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
      auto request = Aws::MakeShared<HttpRequestAdapter>(kAwsTag, uri, method);
      request->SetResponseStreamFactory(streamFactory);
      return request;
  }
};

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


AWSLogSystem::AWSLogSystem(Aws::Utils::Logging::LogLevel log_level) : log_level_(log_level) {};

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
  auto level = Aws::Utils::Logging::LogLevel::Debug;
  //auto level = Aws::Utils::Logging::LogLevel::Info;
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
