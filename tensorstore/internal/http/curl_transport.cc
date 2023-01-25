// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/http/curl_transport.h"

#include <stdlib.h>

#include <clocale>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include <curl/curl.h>
#include "tensorstore/internal/cord_util.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/thread.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {
namespace {

auto& http_request_started = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_started", "HTTP requests started");

auto& http_request_completed = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_completed", "HTTP requests completed");

auto& http_request_errors = internal_metrics::Counter<int64_t, int>::New(
    "/tensorstore/http/request_errors", "code", "HTTP requests with errors");

auto& http_request_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_bytes", "HTTP request bytes transmitted");

auto& http_response_bytes = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/response_bytes", "HTTP response bytes received");

auto& http_active = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/http/active", "HTTP requests considered active");

auto& http_request_latency_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/request_latency_ms", "HTTP request latency (ms)");

// Cached configuration from environment variables.
struct CurlConfig {
  bool verbose = std::getenv("TENSORSTORE_CURL_VERBOSE") != nullptr;
  std::optional<std::string> ca_path = internal::GetEnv("TENSORSTORE_CA_PATH");
  std::optional<std::string> ca_bundle =
      internal::GetEnv("TENSORSTORE_CA_BUNDLE");
};

const CurlConfig& CurlEnvConfig() {
  static const internal::NoDestructor<CurlConfig> curl_config{};
  return *curl_config;
}

struct CurlRequestState {
  CurlHandleFactory* factory_;
  CurlPtr handle_;
  CurlHeaders headers_;
  absl::Cord payload_;
  absl::Cord::CharIterator payload_it_;
  size_t payload_remaining_;
  HttpResponse response_;
  Promise<HttpResponse> promise_;
  char error_buffer_[CURL_ERROR_SIZE] = {0};
  absl::Time start_time_;

  CurlRequestState(CurlHandleFactory* factory)
      : factory_(factory), handle_(factory->CreateHandle()) {
    InitializeCurlHandle(handle_.get());

    const auto& config = CurlEnvConfig();
    if (config.verbose) {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_VERBOSE, 1L));
    }

    if (const auto& x = config.ca_path) {
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_CAPATH, x->c_str()));
    }
    if (const auto& x = config.ca_bundle) {
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_CAINFO, x->c_str()));
    }
    // NOTE: When there are no ca certs, we may want to set:
    // CURLOPT_SSL_VERIFYPEER CURLOPT_SSL_VERIFYHOST

    // Use a 512k buffer.
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_BUFFERSIZE, 512 * 1024));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_TCP_NODELAY, 1L));

    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA, this));
    TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION,
                                        &CurlRequestState::CurlWriteCallback));

    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA, this));
    TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION,
                                        &CurlRequestState::CurlHeaderCallback));

    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_ERRORBUFFER, error_buffer_));
    TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_PRIVATE, this));
  }

  ~CurlRequestState() {
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_PRIVATE, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_READDATA, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_READFUNCTION, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_SEEKDATA, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_SEEKFUNCTION, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION, nullptr));
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_ERRORBUFFER, nullptr));

    factory_->CleanupHandle(std::move(handle_));
  }

  void Setup(const HttpRequest& request, absl::Cord payload,
             absl::Duration request_timeout, absl::Duration connect_timeout) {
    start_time_ = absl::Now();

    // For thread safety, don't use signals to time out name resolves (when
    // async name resolution is not supported).
    //
    // https://curl.haxx.se/libcurl/c/threadsafe.html
    TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_NOSIGNAL, 1L));

    std::string user_agent = request.user_agent() + GetCurlUserAgentSuffix();
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_USERAGENT, user_agent.c_str()));

    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_URL, request.url().c_str()));

    // Convert headers to a curl slist
    curl_slist* head = nullptr;
    for (const std::string& h : request.headers()) {
      head = curl_slist_append(head, h.c_str());
    }
    headers_.reset(head);
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_HTTPHEADER, headers_.get()));
    if (request.accept_encoding()) {
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_ACCEPT_ENCODING, ""));
    }

    if (request_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(request_timeout);
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1));
    }
    if (connect_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(connect_timeout);
      TENSORSTORE_CHECK_OK(CurlEasySetopt(
          handle_.get(), CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1));
    }

    payload_ = std::move(payload);
    payload_remaining_ = payload_.size();
    if (payload_remaining_ > 0) {
      payload_it_ = payload_.char_begin();

      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_READDATA, this));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_READFUNCTION,
                                          &CurlRequestState::CurlReadCallback));
      // Seek callback allows curl to re-read input, which it sometimes needs to
      // do.
      //
      // https://curl.haxx.se/mail/lib-2010-01/0183.html
      //
      // If this is not set, curl may fail with CURLE_SEND_FAIL_REWIND.
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_SEEKFUNCTION,
                                          &CurlRequestState::CurlSeekCallback));
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_SEEKDATA, this));
    }

    if (request.method() == "GET") {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_PIPEWAIT, 1L));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_HTTPGET, 1L));
    } else if (request.method() == "HEAD") {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_NOBODY, 1L));
    } else if (request.method() == "PUT") {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_UPLOAD, 1L));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_PUT, 1L));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(
          handle_.get(), CURLOPT_INFILESIZE_LARGE, payload_remaining_));
    } else if (request.method() == "POST") {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_POST, 1L));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(
          handle_.get(), CURLOPT_POSTFIELDSIZE_LARGE, payload_remaining_));
    } else if (request.method() == "PATCH") {
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_UPLOAD, 1L));
      TENSORSTORE_CHECK_OK(
          CurlEasySetopt(handle_.get(), CURLOPT_CUSTOMREQUEST, "PATCH"));
      TENSORSTORE_CHECK_OK(CurlEasySetopt(
          handle_.get(), CURLOPT_POSTFIELDSIZE_LARGE, payload_remaining_));
    } else {
      // Such as "DELETE"
      TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_CUSTOMREQUEST,
                                          request.method().c_str()));
    }
  }

  void SetHTTP2() {
    TENSORSTORE_CHECK_OK(CurlEasySetopt(handle_.get(), CURLOPT_HTTP_VERSION,
                                        CURL_HTTP_VERSION_2_0));
  }

  void SetForbidReuse() {
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    TENSORSTORE_CHECK_OK(
        CurlEasySetopt(handle_.get(), CURLOPT_FORBID_REUSE, 1));
  }

  absl::Status CurlCodeToStatus(CURLcode code) {
    if (code == CURLE_OK) {
      return absl::OkStatus();
    }
    ABSL_LOG(WARNING) << "Error [" << code << "]=" << curl_easy_strerror(code)
                      << " in curl operation\n"
                      << error_buffer_;
    return ::tensorstore::internal_http::CurlCodeToStatus(code, error_buffer_);
  }

  static std::size_t CurlWriteCallback(void* contents, std::size_t size,
                                       std::size_t nmemb, void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    http_response_bytes.IncrementBy(size);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    self->response_.payload.Append(data);
    return data.size();
  }

  static std::size_t CurlReadCallback(void* contents, std::size_t size,
                                      std::size_t nmemb, void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    size_t n = std::min(size * nmemb, self->payload_remaining_);
    http_request_bytes.IncrementBy(n);
    internal::CopyCordToSpan(self->payload_it_, {static_cast<char*>(contents),
                                                 static_cast<ptrdiff_t>(n)});
    self->payload_remaining_ -= n;
    return n;
  }

  static int CurlSeekCallback(void* userdata, curl_off_t offset, int origin) {
    if (origin != SEEK_SET) {
      // According to the documentation:
      // https://curl.haxx.se/libcurl/c/CURLOPT_SEEKFUNCTION.html
      //
      // curl only uses SEEK_SET.
      return CURL_SEEKFUNC_CANTSEEK;
    }
    auto* self = static_cast<CurlRequestState*>(userdata);
    if (offset < 0 || offset > self->payload_.size()) {
      return CURL_SEEKFUNC_FAIL;
    }
    self->payload_it_ = self->payload_.char_begin();
    absl::Cord::Advance(&self->payload_it_, static_cast<size_t>(offset));
    self->payload_remaining_ =
        self->payload_.size() - static_cast<size_t>(offset);
    return CURL_SEEKFUNC_OK;
  }

  static std::size_t CurlHeaderCallback(void* contents, std::size_t size,
                                        std::size_t nmemb, void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    return AppendHeaderData(self->response_.headers, data);
  }
};

class MultiTransportImpl {
 public:
  explicit MultiTransportImpl(std::shared_ptr<CurlHandleFactory> factory)
      : factory_(factory), multi_(factory_->CreateMultiHandle()) {
    thread_ = internal::Thread({"curl_handler"}, [this] { Run(); });
  }

  ~MultiTransportImpl() {
    done_ = true;
    curl_multi_wakeup(multi_.get());

    thread_.Join();
    factory_->CleanupMultiHandle(std::move(multi_));
  }

  Future<HttpResponse> StartRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout);

  void FinishRequest(CURL* e, CURLcode code);

  void Run();

  void HandlePendingMesssages(size_t& active_count);

  CurlRequestState* GetStatePointer(CURL* e) {
    CurlRequestState* pvt = nullptr;
    curl_easy_getinfo(e, CURLINFO_PRIVATE, &pvt);
    return pvt;
  }

  std::shared_ptr<CurlHandleFactory> factory_;
  CurlMulti multi_;

  absl::Mutex mutex_;
  std::vector<CURL*> pending_requests_;
  std::atomic<bool> done_{false};

  internal::Thread thread_;
};

Future<HttpResponse> MultiTransportImpl::StartRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  auto state = std::make_unique<CurlRequestState>(factory_.get());
  http_request_started.Increment();
  state->Setup(request, std::move(payload), request_timeout, connect_timeout);
  state->SetHTTP2();

  auto pair = PromiseFuturePair<HttpResponse>::Make();
  state->promise_ = std::move(pair.promise);

  // Transfer ownership into the curl handle.
  CURL* e = state->handle_.get();
  assert(state.get() == GetStatePointer(e));
  state.release();

  // Add the handle to the curl_multi state.
  // TODO: Add an ExecuteWhenNotNeeded callback which removes
  // the handle from the pending / active requests set.
  {
    absl::MutexLock l(&mutex_);
    pending_requests_.emplace_back(e);
  }
  curl_multi_wakeup(multi_.get());

  return std::move(pair.future);
}

void MultiTransportImpl::FinishRequest(CURL* e, CURLcode code) {
  // Transfer ownership out of the curl handle.
  std::unique_ptr<CurlRequestState> state(GetStatePointer(e));

  if (code == CURLE_HTTP2) {
    ABSL_LOG(WARNING) << "CURLE_HTTP2 " << state->error_buffer_;
    // If there was an error in the HTTP2 framing, try and force
    // CURL to close the connection stream.
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    state->SetForbidReuse();
  }

  auto latency = absl::Now() - state->start_time_;
  http_request_latency_ms.Observe(absl::ToInt64Milliseconds(latency));
  http_request_completed.Increment();

  if (code != CURLE_OK) {
    state->promise_.SetResult(state->CurlCodeToStatus(code));
  } else {
    state->response_.status_code = CurlGetResponseCode(e);
    http_request_errors.Increment(state->response_.status_code);
    state->promise_.SetResult(std::move(state->response_));
  }
}

void MultiTransportImpl::HandlePendingMesssages(size_t& active_count) {
  // curl_multi_perform is the main curl method that performs work
  // on the existing curl handles.
  int active_requests;
  while (CURLM_CALL_MULTI_PERFORM ==
         curl_multi_perform(multi_.get(), &active_requests)) {
    /* loop */
  }
  http_active.Set(active_requests);

  // Pull pending CURLMSG_DONE events off the multi handle.
  for (;;) {
    int messages_in_queue;
    const auto* m = curl_multi_info_read(multi_.get(), &messages_in_queue);
    if (!m) break;
    // Remove completed message from curl multi handle.
    if (m->msg == CURLMSG_DONE) {
      CURLcode result = m->data.result;
      CURL* e = m->easy_handle;

      active_count--;
      curl_multi_remove_handle(multi_.get(), e);
      FinishRequest(e, result);
    }
  }
}

void MultiTransportImpl::Run() {
  // track active count separate from the curl_multi so it's available without
  // calling curl_multi_perform or similar.
  size_t active_count = 0;
  for (;;) {
    {
      absl::MutexLock l(&mutex_);

      // Add any pending requests.
      for (CURL* e : pending_requests_) {
        CurlRequestState* state = nullptr;
        curl_easy_getinfo(e, CURLINFO_PRIVATE, &state);

        // This future has been cancelled before we even begin.
        if (!state->promise_.result_needed()) continue;

        CURLMcode mcode = curl_multi_add_handle(multi_.get(), e);
        if (mcode == CURLM_OK) {
          active_count++;
        } else {
          // This shouldn't happen unless things have really gone pear-shaped.
          state->promise_.SetResult(
              CurlMCodeToStatus(mcode, "in curl_multi_add_handle"));
        }
      }
      pending_requests_.clear();
    }

    // Handle any pending transfers.
    HandlePendingMesssages(active_count);

    // Stop if there are no active requests and shutdown has been requested.
    if (active_count == 0) {
      if (done_.load()) break;
    }

    // Wait for more transfers to complete.  Rely on curl_multi_wakeup to
    // notify that non-transfer work is ready, otherwise wake up once per
    // timeout interval.
    const int timeout_ms = std::numeric_limits<int>::max();  // infinite
    int numfds = 0;
    CURLMcode mcode =
        curl_multi_poll(multi_.get(), nullptr, 0, timeout_ms, &numfds);
    if (mcode != CURLM_OK) {
      auto status = CurlMCodeToStatus(mcode, "in CurlMultiTransport");
      ABSL_LOG(WARNING) << "Error [" << mcode << "] " << status;
    }
  }
}

}  // namespace

class CurlTransport::Impl : public MultiTransportImpl {
 public:
  using MultiTransportImpl::MultiTransportImpl;
};

CurlTransport::CurlTransport(std::shared_ptr<CurlHandleFactory> factory)
    : impl_(std::make_unique<Impl>(std::move(factory))) {}

CurlTransport::~CurlTransport() = default;

Future<HttpResponse> CurlTransport::IssueRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  return impl_->StartRequest(request, std::move(payload), request_timeout,
                             connect_timeout);
}

namespace {
struct GlobalTransport {
  GlobalTransport()
      : transport(
            std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory())) {}

  std::shared_ptr<HttpTransport> transport;
};

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

static GlobalTransport& GetGlobalTransport() {
  static auto* g = new GlobalTransport();
  return *g;
}

}  // namespace

std::shared_ptr<HttpTransport> GetDefaultHttpTransport() {
  absl::MutexLock l(&global_mu);
  return GetGlobalTransport().transport;
}

/// Sets the default CurlMultiTransport. Exposed for test mocking.
void SetDefaultHttpTransport(std::shared_ptr<HttpTransport> t) {
  absl::MutexLock l(&global_mu);
  if (!t) {
    t = std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory());
  }
  GetGlobalTransport().transport = std::move(t);
}

}  // namespace internal_http
}  // namespace tensorstore
