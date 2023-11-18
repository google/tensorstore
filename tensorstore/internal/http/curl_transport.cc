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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include <curl/curl.h>
#include "tensorstore/internal/cord_util.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_factory.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_wrappers.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {
namespace {

auto& http_request_started = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_started", "HTTP requests started");

auto& http_request_completed = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_completed", "HTTP requests completed");

auto& http_request_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/request_bytes", "HTTP request bytes transmitted");

auto& http_request_header_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/request_header_bytes",
        "HTTP request bytes transmitted");

auto& http_response_codes = internal_metrics::Counter<int64_t, int>::New(
    "/tensorstore/http/response_codes", "code",
    "HTTP response status code counts");

auto& http_response_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/response_bytes", "HTTP response bytes received");

auto& http_active = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/http/active", "HTTP requests considered active");

auto& http_total_time_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/total_time_ms", "HTTP total latency (ms)");

auto& http_first_byte_latency_us =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/first_byte_latency_us",
        "HTTP first byte received latency (us)");

ABSL_CONST_INIT internal_log::VerboseFlag curl_logging("curl");

// Concurrent CURL HTTP/2 streams.
int32_t GetMaxHttp2ConcurrentStreams() {
  auto limit = internal::GetEnvValue<int32_t>(
      "TENSORSTORE_HTTP2_MAX_CONCURRENT_STREAMS");
  if (limit && (*limit <= 0 || *limit > 1000)) {
    ABSL_LOG(WARNING)
        << "Failed to parse TENSORSTORE_HTTP2_MAX_CONCURRENT_STREAMS: "
        << *limit;
    limit = std::nullopt;
  }
  return limit.value_or(4);  // New default streams.
}

// Cached configuration from environment variables.
struct CurlConfig {
  bool verbose = internal::GetEnvValue<bool>("TENSORSTORE_CURL_VERBOSE")
                     .value_or(curl_logging.Level(0));
  std::optional<std::string> ca_path = internal::GetEnv("TENSORSTORE_CA_PATH");
  std::optional<std::string> ca_bundle =
      internal::GetEnv("TENSORSTORE_CA_BUNDLE");
  int64_t low_speed_time_seconds =
      internal::GetEnvValue<int64_t>("TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS")
          .value_or(0);
  int64_t low_speed_limit_bytes =
      internal::GetEnvValue<int64_t>("TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES")
          .value_or(0);
};

const CurlConfig& CurlEnvConfig() {
  static const internal::NoDestructor<CurlConfig> curl_config{};
  return *curl_config;
}

struct CurlRequestState {
  std::shared_ptr<CurlHandleFactory> factory_;
  CurlHandle handle_;
  CurlHeaders headers_;
  absl::Cord payload_;
  absl::Cord::CharIterator payload_it_;
  size_t payload_remaining_;
  HttpResponse response_;
  Promise<HttpResponse> promise_;
  char error_buffer_[CURL_ERROR_SIZE] = {0};

  CurlRequestState(std::shared_ptr<CurlHandleFactory> factory)
      : factory_(std::move(factory)), handle_(CurlHandle::Create(*factory_)) {
    const auto& config = CurlEnvConfig();
    if (config.verbose) {
      handle_.SetOption(CURLOPT_VERBOSE, 1L);
    }
    handle_.SetOption(CURLOPT_ERRORBUFFER, error_buffer_);

    // For thread safety, don't use signals to time out name resolves (when
    // async name resolution is not supported).
    //
    // https://curl.haxx.se/libcurl/c/threadsafe.html
    handle_.SetOption(CURLOPT_NOSIGNAL, 1L);

    // Follow curl command manpage to set up default values for low speed
    // timeout:
    // https://curl.se/docs/manpage.html#-Y
    if (config.low_speed_time_seconds > 0 || config.low_speed_limit_bytes > 0) {
      auto seconds = config.low_speed_time_seconds > 0
                         ? config.low_speed_time_seconds
                         : 30;
      auto bytes =
          config.low_speed_limit_bytes > 0 ? config.low_speed_limit_bytes : 1;
      handle_.SetOption(CURLOPT_LOW_SPEED_TIME, seconds);
      handle_.SetOption(CURLOPT_LOW_SPEED_LIMIT, bytes);
    }

    if (const auto& x = config.ca_path) {
      handle_.SetOption(CURLOPT_CAPATH, x->c_str());
    }
    if (const auto& x = config.ca_bundle) {
      handle_.SetOption(CURLOPT_CAINFO, x->c_str());
    }
    // NOTE: When there are no ca certs, we may want to set:
    // CURLOPT_SSL_VERIFYPEER CURLOPT_SSL_VERIFYHOST

    // Use a 512k buffer.
    handle_.SetOption(CURLOPT_BUFFERSIZE, 512 * 1024);
    handle_.SetOption(CURLOPT_TCP_NODELAY, 1L);

    handle_.SetOption(CURLOPT_WRITEDATA, this);
    handle_.SetOption(CURLOPT_WRITEFUNCTION,
                      &CurlRequestState::CurlWriteCallback);

    handle_.SetOption(CURLOPT_HEADERDATA, this);
    handle_.SetOption(CURLOPT_HEADERFUNCTION,
                      &CurlRequestState::CurlHeaderCallback);

    // Consider: CURLOPT_XFERINFOFUNCTION for increased logging.
  }

  ~CurlRequestState() {
    handle_.SetOption(CURLOPT_WRITEDATA, nullptr);
    handle_.SetOption(CURLOPT_WRITEFUNCTION, nullptr);
    handle_.SetOption(CURLOPT_READDATA, nullptr);
    handle_.SetOption(CURLOPT_READFUNCTION, nullptr);
    handle_.SetOption(CURLOPT_SEEKDATA, nullptr);
    handle_.SetOption(CURLOPT_SEEKFUNCTION, nullptr);
    handle_.SetOption(CURLOPT_HEADERDATA, nullptr);
    handle_.SetOption(CURLOPT_HEADERFUNCTION, nullptr);
    handle_.SetOption(CURLOPT_LOW_SPEED_TIME, 0L);
    handle_.SetOption(CURLOPT_LOW_SPEED_LIMIT, 0L);
    handle_.SetOption(CURLOPT_VERBOSE, 0);
    handle_.SetOption(CURLOPT_ERRORBUFFER, nullptr);
    CurlHandle::Cleanup(*factory_, std::move(handle_));
  }

  void Prepare(const HttpRequest& request, absl::Cord payload,
               absl::Duration request_timeout, absl::Duration connect_timeout) {
    handle_.SetOption(CURLOPT_URL, request.url.c_str());

    std::string user_agent = request.user_agent + GetCurlUserAgentSuffix();
    handle_.SetOption(CURLOPT_USERAGENT, user_agent.c_str());

    // Convert headers to a curl slist
    curl_slist* head = nullptr;
    size_t header_bytes_ = 0;
    for (const std::string& h : request.headers) {
      head = curl_slist_append(head, h.c_str());
      header_bytes_ += h.size();
    }
    headers_.reset(head);
    handle_.SetOption(CURLOPT_HTTPHEADER, headers_.get());
    if (request.accept_encoding) {
      handle_.SetOption(CURLOPT_ACCEPT_ENCODING, "");
    }

    if (request_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(request_timeout);

      handle_.SetOption(CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (connect_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(connect_timeout);

      handle_.SetOption(CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1);
    }

    payload_ = std::move(payload);
    payload_remaining_ = payload_.size();
    if (payload_remaining_ > 0) {
      payload_it_ = payload_.char_begin();

      handle_.SetOption(CURLOPT_READDATA, this);
      handle_.SetOption(CURLOPT_READFUNCTION,
                        &CurlRequestState::CurlReadCallback);
      // Seek callback allows curl to re-read input, which it sometimes needs to
      // do.
      //
      // https://curl.haxx.se/mail/lib-2010-01/0183.html
      //
      // If this is not set, curl may fail with CURLE_SEND_FAIL_REWIND.
      handle_.SetOption(CURLOPT_SEEKDATA, this);
      handle_.SetOption(CURLOPT_SEEKFUNCTION,
                        &CurlRequestState::CurlSeekCallback);
    }

    if (request.method == "GET") {
      handle_.SetOption(CURLOPT_PIPEWAIT, 1L);
      handle_.SetOption(CURLOPT_HTTPGET, 1L);
    } else if (request.method == "HEAD") {
      handle_.SetOption(CURLOPT_NOBODY, 1L);
    } else if (request.method == "PUT") {
      handle_.SetOption(CURLOPT_UPLOAD, 1L);
      handle_.SetOption(CURLOPT_PUT, 1L);

      handle_.SetOption(CURLOPT_INFILESIZE_LARGE, payload_remaining_);
    } else if (request.method == "POST") {
      handle_.SetOption(CURLOPT_POST, 1L);

      handle_.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, payload_remaining_);
    } else if (request.method == "PATCH") {
      handle_.SetOption(CURLOPT_UPLOAD, 1L);
      handle_.SetOption(CURLOPT_CUSTOMREQUEST, "PATCH");

      handle_.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, payload_remaining_);
    } else {
      // Such as "DELETE"
      handle_.SetOption(CURLOPT_CUSTOMREQUEST, request.method.c_str());
    }

    // Record metrics.
    http_request_started.Increment();
    http_request_bytes.Observe(payload_remaining_);
    http_request_header_bytes.Observe(header_bytes_);
  }

  void SetHTTP2() {
    handle_.SetOption(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
  }

  void SetForbidReuse() {
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    handle_.SetOption(CURLOPT_FORBID_REUSE, 1);
  }

  static size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                                  void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    self->response_.payload.Append(data);
    return data.size();
  }

  static size_t CurlReadCallback(void* contents, size_t size, size_t nmemb,
                                 void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    size_t n = std::min(size * nmemb, self->payload_remaining_);
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

  static size_t CurlHeaderCallback(void* contents, size_t size, size_t nmemb,
                                   void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    return AppendHeaderData(self->response_.headers, data);
  }
};

class MultiTransportImpl {
 public:
  explicit MultiTransportImpl(std::shared_ptr<CurlHandleFactory> factory)
      : factory_(std::move(factory)), multi_(factory_->CreateMultiHandle()) {
    assert(factory_);
    // Without any option, the CURL library multiplexes up to 100 http/2 streams
    // over a single connection. In practice there's a tradeoff between
    // concurrent streams and latency/throughput of requests. Empirical tests
    // suggest that using a small number of streams per connection increases
    // throughput of large transfers, which is common in tensorstore.
    static int32_t max_concurrent_streams = GetMaxHttp2ConcurrentStreams();
    curl_multi_setopt(multi_.get(), CURLMOPT_MAX_CONCURRENT_STREAMS,
                      max_concurrent_streams);
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

  void FinishRequest(std::unique_ptr<CurlRequestState> state, CURLcode code);

  // Runs the thread loop.
  void Run();

  int64_t AddPendingTransfers();
  int64_t RemoveCompletedTransfers();

  std::shared_ptr<CurlHandleFactory> factory_;
  CurlMulti multi_;

  absl::Mutex mutex_;
  std::vector<std::unique_ptr<CurlRequestState>> pending_requests_;
  std::atomic<bool> done_{false};

  internal::Thread thread_;
};

Future<HttpResponse> MultiTransportImpl::StartRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  assert(factory_);
  auto state = std::make_unique<CurlRequestState>(factory_);
  state->Prepare(request, std::move(payload), request_timeout, connect_timeout);
  state->SetHTTP2();

  auto pair = PromiseFuturePair<HttpResponse>::Make();
  state->promise_ = std::move(pair.promise);

  // Add the handle to the curl_multi state.
  // TODO: Add an ExecuteWhenNotNeeded callback which removes
  // the handle from the pending / active requests set.
  {
    absl::MutexLock l(&mutex_);
    pending_requests_.push_back(std::move(state));
  }
  curl_multi_wakeup(multi_.get());

  return std::move(pair.future);
}

void MultiTransportImpl::FinishRequest(std::unique_ptr<CurlRequestState> state,
                                       CURLcode code) {
  if (code == CURLE_HTTP2) {
    ABSL_LOG(WARNING) << "CURLE_HTTP2 " << state->error_buffer_;
    // If there was an error in the HTTP2 framing, try and force
    // CURL to close the connection stream.
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    state->SetForbidReuse();
  }

  // NOTE: Consider recording curl getinfo options:
  // https://curl.se/libcurl/c/easy_getinfo_options.html
  http_request_completed.Increment();
  http_response_bytes.Observe(state->response_.payload.size());

  // Record the first byte latency.
  {
    curl_off_t first_byte_us = 0;
    state->handle_.GetInfo(CURLINFO_STARTTRANSFER_TIME_T, &first_byte_us);
    http_first_byte_latency_us.Observe(first_byte_us);
  }
  // Record the total time.
  {
    curl_off_t total_time_us = 0;
    state->handle_.GetInfo(CURLINFO_TOTAL_TIME_T, &total_time_us);
    http_total_time_ms.Observe(total_time_us / 1000);
  }

  if (code != CURLE_OK) {
    /// Transfer failed; set the status
    ABSL_LOG(WARNING) << "Error [" << code << "]=" << curl_easy_strerror(code)
                      << " in curl operation\n"
                      << state->error_buffer_;
    state->promise_.SetResult(CurlCodeToStatus(code, state->error_buffer_));
    return;
  }

  state->response_.status_code = state->handle_.GetResponseCode();
  http_response_codes.Increment(state->response_.status_code);
  state->promise_.SetResult(std::move(state->response_));
}

void MultiTransportImpl::Run() {
  // track active count separate from the curl_multi so it's available without
  // calling curl_multi_perform or similar.
  int64_t active_count = 0;
  for (;;) {
    // Add any pending transfers.
    active_count += AddPendingTransfers();

    // Perform work.
    {
      int running_handles = 0;
      CURLMcode mcode;
      do {
        mcode = curl_multi_perform(multi_.get(), &running_handles);
        http_active.Set(running_handles);
      } while (mcode == CURLM_CALL_MULTI_PERFORM);

      if (mcode != CURLM_OK) {
        ABSL_LOG(WARNING) << CurlMCodeToStatus(mcode, "in curl_multi_perform");
      }
    }

    active_count -= RemoveCompletedTransfers();

    // Stop if there are no active requests and shutdown has been requested.
    if (active_count == 0) {
      if (done_.load()) break;
    }

    // Wait for more transfers to complete.  Rely on curl_multi_wakeup to
    // notify that non-transfer work is ready, otherwise wake up once per
    // timeout interval.
    // Allow spurious EINTR to wake the loop; it does no harm here.
    {
      const int timeout_ms = std::numeric_limits<int>::max();  // infinite
      int numfds = 0;
      errno = 0;
      CURLMcode mcode =
          curl_multi_poll(multi_.get(), nullptr, 0, timeout_ms, &numfds);
      if (mcode != CURLM_OK) {
        ABSL_LOG(WARNING) << CurlMCodeToStatus(mcode, "in curl_multi_poll");
      }
    }
  }
}

int64_t MultiTransportImpl::AddPendingTransfers() {
  int64_t active_count = 0;
  absl::MutexLock l(&mutex_);

  // Add any pending requests.
  for (auto& state : pending_requests_) {
    // This future has been cancelled before we even begin.
    if (!state->promise_.result_needed()) continue;

    // Set the CURLINFO_PRIVATE data to take pointer ownership.
    state->handle_.SetOption(CURLOPT_PRIVATE, state.get());

    CURL* e = state->handle_.get();
    CURLMcode mcode = curl_multi_add_handle(multi_.get(), e);
    if (mcode == CURLM_OK) {
      // ownership successfully transferred.
      state.release();
      active_count++;
    } else {
      // This shouldn't happen unless things have really gone pear-shaped.
      state->promise_.SetResult(
          CurlMCodeToStatus(mcode, "in curl_multi_add_handle"));
    }
  }

  pending_requests_.clear();
  return active_count;
}

int64_t MultiTransportImpl::RemoveCompletedTransfers() {
  // Pull pending CURLMSG_DONE events off the multi handle.
  int64_t completed = 0;
  CURLMsg* m = nullptr;
  do {
    int messages_in_queue;
    m = curl_multi_info_read(multi_.get(), &messages_in_queue);

    // Remove completed message from curl multi handle.
    if (m && m->msg == CURLMSG_DONE) {
      CURLcode result = m->data.result;
      CURL* e = m->easy_handle;

      completed++;
      curl_multi_remove_handle(multi_.get(), e);

      CurlRequestState* pvt = nullptr;
      curl_easy_getinfo(e, CURLINFO_PRIVATE, &pvt);
      assert(pvt);
      std::unique_ptr<CurlRequestState> state(pvt);
      state->handle_.SetOption(CURLOPT_PRIVATE, nullptr);

      FinishRequest(std::move(state), result);
    }
  } while (m != nullptr);

  return completed;
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
  assert(impl_);
  return impl_->StartRequest(request, std::move(payload), request_timeout,
                             connect_timeout);
}

namespace {
struct GlobalTransport {
  std::shared_ptr<HttpTransport> transport_;

  std::shared_ptr<HttpTransport> Get() {
    if (!transport_) {
      transport_ =
          std::make_shared<CurlTransport>(GetDefaultCurlHandleFactory());
    }
    return transport_;
  }

  void Set(std::shared_ptr<HttpTransport> transport) {
    transport_ = std::move(transport);
  }
};

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

static GlobalTransport& GetGlobalTransport() {
  static auto* g = new GlobalTransport();
  return *g;
}

}  // namespace

std::shared_ptr<HttpTransport> GetDefaultHttpTransport() {
  absl::MutexLock l(&global_mu);
  return GetGlobalTransport().Get();
}

/// Sets the default CurlMultiTransport. Exposed for test mocking.
void SetDefaultHttpTransport(std::shared_ptr<HttpTransport> t) {
  absl::MutexLock l(&global_mu);
  return GetGlobalTransport().Set(std::move(t));
}

}  // namespace internal_http
}  // namespace tensorstore
