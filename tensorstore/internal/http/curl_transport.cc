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
#include <cerrno>
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
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <curl/curl.h>
#include "tensorstore/internal/container/circular_queue.h"
#include "tensorstore/internal/cord_util.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_factory.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/curl_wrappers.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"
#include "tensorstore/internal/thread/thread.h"

ABSL_FLAG(std::optional<uint32_t>, tensorstore_http_threads, std::nullopt,
          "Threads to use for http requests. "
          "Overrides TENSORSTORE_HTTP_THREADS.");

using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal_container::CircularQueue;
using ::tensorstore::internal_metrics::MetricMetadata;

namespace tensorstore {
namespace internal_http {
namespace {

auto& http_request_started = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_started",
    MetricMetadata("HTTP requests started"));

auto& http_request_completed = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/http/request_completed",
    MetricMetadata("HTTP requests completed"));

auto& http_request_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/request_bytes",
        MetricMetadata("HTTP request bytes transmitted",
                       internal_metrics::Units::kBytes));

auto& http_request_header_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/request_header_bytes",
        MetricMetadata("HTTP request bytes transmitted",
                       internal_metrics::Units::kBytes));

auto& http_response_codes = internal_metrics::Counter<int64_t, int>::New(
    "/tensorstore/http/response_codes", "code",
    MetricMetadata("HTTP response status code counts"));

auto& http_response_bytes =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/response_bytes",
        MetricMetadata("HTTP response bytes received",
                       internal_metrics::Units::kBytes));

auto& http_active = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/http/active",
    MetricMetadata("HTTP requests considered active"));

auto& http_total_time_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/total_time_ms",
        MetricMetadata("HTTP total latency (ms)",
                       internal_metrics::Units::kMilliseconds));

auto& http_first_byte_latency_us =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/first_byte_latency_us",
        MetricMetadata("HTTP first byte received latency (us)",
                       internal_metrics::Units::kMicroseconds));

auto& http_poll_time_ns =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/http/http_poll_time_ns",
        MetricMetadata("HTTP time spent in curl_multi_poll (ns)",
                       internal_metrics::Units::kNanoseconds));

uint32_t GetHttpThreads() {
  return std::max(1u, GetFlagOrEnvValue(FLAGS_tensorstore_http_threads,
                                        "TENSORSTORE_HTTP_THREADS")
                          .value_or(4u));
}

struct CurlRequestState {
  std::shared_ptr<CurlHandleFactory> factory_;
  CurlHandle handle_;
  CurlHeaders headers_;
  absl::Cord payload_;
  absl::Cord::CharIterator payload_it_;
  size_t payload_remaining_;
  HttpResponseHandler* response_handler_ = nullptr;
  size_t response_payload_size_ = 0;
  bool status_set = false;
  char error_buffer_[CURL_ERROR_SIZE];

  CurlRequestState(std::shared_ptr<CurlHandleFactory> factory)
      : factory_(std::move(factory)), handle_(CurlHandle::Create(*factory_)) {
    error_buffer_[0] = 0;
    handle_.SetOption(CURLOPT_ERRORBUFFER, error_buffer_);

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
    handle_.SetOption(CURLOPT_ERRORBUFFER, nullptr);
    CurlHandle::Cleanup(*factory_, std::move(handle_));
  }

  void Prepare(const HttpRequest& request, IssueRequestOptions options) {
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

    if (options.request_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(options.request_timeout);
      handle_.SetOption(CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (options.connect_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(options.connect_timeout);
      handle_.SetOption(CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1);
    }

    payload_ = std::move(options.payload);
    payload_remaining_ = payload_.size();
    if (payload_remaining_ > 0) {
      payload_it_ = payload_.char_begin();

      handle_.SetOption(CURLOPT_READDATA, this);
      handle_.SetOption(CURLOPT_READFUNCTION,
                        &CurlRequestState::CurlReadCallback);
      // Seek callback allows curl to re-read input, which it sometimes needs
      // to do.
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

    // Maybe set HTTP version on the request.
    switch (options.http_version) {
      case IssueRequestOptions::HttpVersion::kHttp1:
        handle_.SetOption(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
        break;
      case IssueRequestOptions::HttpVersion::kHttp2:
        handle_.SetOption(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
        break;
      case IssueRequestOptions::HttpVersion::kHttp2TLS:
        handle_.SetOption(CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2TLS);
        break;
      case IssueRequestOptions::HttpVersion::kHttp2PriorKnowledge:
        handle_.SetOption(CURLOPT_HTTP_VERSION,
                          CURL_HTTP_VERSION_2_PRIOR_KNOWLEDGE);
        break;
      default:
        break;
    }

    // Record metrics.
    http_request_started.Increment();
    http_request_bytes.Observe(payload_remaining_);
    http_request_header_bytes.Observe(header_bytes_);
  }

  void SetForbidReuse() {
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    handle_.SetOption(CURLOPT_FORBID_REUSE, 1);
  }

  bool MaybeSetStatusAndProcess() {
    if (status_set) return true;
    auto status_code = handle_.GetResponseCode();
    // Status < 200 are intermediate and handled by libcurl.
    if (status_code < 200) return false;
    response_handler_->OnStatus(status_code);
    status_set = true;
    return true;
  }

  static size_t CurlHeaderCallback(void* contents, size_t size, size_t nmemb,
                                   void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    if (self->MaybeSetStatusAndProcess()) {
      self->response_handler_->OnResponseHeader(data);
    }
    return data.size();
  }

  static size_t CurlWriteCallback(void* contents, size_t size, size_t nmemb,
                                  void* userdata) {
    auto* self = static_cast<CurlRequestState*>(userdata);
    auto data =
        std::string_view(static_cast<char const*>(contents), size * nmemb);
    if (self->MaybeSetStatusAndProcess()) {
      self->response_payload_size_ += data.size();
      self->response_handler_->OnResponseBody(data);
    }
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
};

class MultiTransportImpl {
 public:
  MultiTransportImpl(std::shared_ptr<CurlHandleFactory> factory,
                     size_t nthreads);

  ~MultiTransportImpl();

  void EnqueueRequest(const HttpRequest& request, IssueRequestOptions options,
                      HttpResponseHandler* response_handler);

  void FinishRequest(std::unique_ptr<CurlRequestState> state, CURLcode code);

 private:
  struct ThreadData {
    std::atomic<int64_t> count = 0;
    CurlMulti multi;
    absl::Mutex mutex;
    CircularQueue<std::unique_ptr<CurlRequestState>> pending{16};
    bool done = false;
  };

  // Runs the thread loop.
  void Run(ThreadData& thread_data);

  void MaybeAddPendingTransfers(ThreadData& thread_data);
  void RemoveCompletedTransfers(ThreadData& thread_data);

  std::shared_ptr<CurlHandleFactory> factory_;
  std::atomic<bool> done_{false};

  std::unique_ptr<ThreadData[]> thread_data_;
  std::vector<internal::Thread> threads_;
};

MultiTransportImpl::MultiTransportImpl(
    std::shared_ptr<CurlHandleFactory> factory, size_t nthreads)
    : factory_(std::move(factory)) {
  assert(factory_);
  threads_.reserve(nthreads);
  thread_data_ = std::make_unique<ThreadData[]>(nthreads);
  for (size_t i = 0; i < nthreads; ++i) {
    thread_data_[i].multi = factory_->CreateMultiHandle();
    threads_.push_back(
        internal::Thread({"curl_multi_thread"},
                         [this, index = i] { Run(thread_data_[index]); }));
  }
}

MultiTransportImpl::~MultiTransportImpl() {
  done_ = true;

  // Wake everything...
  for (size_t i = 0; i < threads_.size(); ++i) {
    auto& thread_data = thread_data_[i];
    absl::MutexLock l(&thread_data.mutex);
    thread_data.done = true;
    curl_multi_wakeup(thread_data.multi.get());
  }
  for (auto& thread : threads_) {
    thread.Join();
  }
  for (size_t i = 0; i < threads_.size(); ++i) {
    factory_->CleanupMultiHandle(std::move(thread_data_[i].multi));
  }
}

void MultiTransportImpl::EnqueueRequest(const HttpRequest& request,
                                        IssueRequestOptions options,
                                        HttpResponseHandler* response_handler) {
  if (done_.load()) {
    response_handler->OnFailure(
        absl::InternalError("MultiTransportImpl is shutting down"));
    return;
  }

  auto state = std::make_unique<CurlRequestState>(factory_);
  state->response_handler_ = response_handler;
  state->Prepare(request, std::move(options));

  // Select the thread with the fewest active connections, and then enqueue
  // the request on that thread.
  size_t selected_index = 0;
  for (size_t i = 1; i < threads_.size(); ++i) {
    if (thread_data_[i].count < thread_data_[selected_index].count) {
      selected_index = i;
    }
  }

  auto& selected = thread_data_[selected_index];
  absl::MutexLock l(&selected.mutex);
  selected.pending.push_back(std::move(state));
  selected.count++;
  curl_multi_wakeup(selected.multi.get());
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
  http_response_bytes.Observe(state->response_payload_size_);

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
    state->response_handler_->OnFailure(
        CurlCodeToStatus(code, state->error_buffer_));
    return;
  }

  http_response_codes.Increment(state->handle_.GetResponseCode());
  assert(state->status_set);
  state->response_handler_->OnComplete();
}

void MultiTransportImpl::Run(ThreadData& thread_data) {
  for (;;) {
    // Add any pending transfers.
    MaybeAddPendingTransfers(thread_data);

    if (thread_data.count == 0) {
      // There are no active requests; wait for active requests or shutdown.
      absl::MutexLock l(&thread_data.mutex);
      if (thread_data.done) break;
      thread_data.mutex.Await(absl::Condition(
          +[](ThreadData* td) { return !td->pending.empty() || td->done; },
          &thread_data));
      if (thread_data.done) break;
      continue;
    }

    // Wait for more transfers to complete.  Rely on curl_multi_wakeup to
    // notify that non-transfer work is ready, otherwise wake up once per
    // timeout interval.
    // Allow spurious EINTR to wake the loop; it does no harm here.
    const int timeout_ms = std::numeric_limits<int>::max();  // infinite
    int numfds = 0;
    errno = 0;
    auto start_poll = absl::Now();
    CURLMcode mcode = curl_multi_poll(thread_data.multi.get(), nullptr, 0,
                                      timeout_ms, &numfds);
    if (mcode != CURLM_OK) {
      ABSL_LOG(WARNING) << CurlMCodeToStatus(mcode, "in curl_multi_poll");
    }
    http_poll_time_ns.Observe(
        absl::ToInt64Nanoseconds(absl::Now() - start_poll));

    // Perform work.
    {
      int running_handles = 0;
      CURLMcode mcode;
      do {
        mcode = curl_multi_perform(thread_data.multi.get(), &running_handles);
        http_active.Set(running_handles);
      } while (mcode == CURLM_CALL_MULTI_PERFORM);

      if (mcode != CURLM_OK) {
        ABSL_LOG(WARNING) << CurlMCodeToStatus(mcode, "in curl_multi_perform");
      }
    }

    RemoveCompletedTransfers(thread_data);
  }

  // Remove the handle
  assert(thread_data.count == 0);
}

void MultiTransportImpl::MaybeAddPendingTransfers(ThreadData& thread_data) {
  absl::MutexLock l(&thread_data.mutex);
  while (!thread_data.pending.empty()) {
    std::unique_ptr<CurlRequestState> state =
        std::move(thread_data.pending.front());
    thread_data.pending.pop_front();

    assert(state != nullptr);
    // Add state to multi handle.
    // Set the CURLINFO_PRIVATE data to take pointer ownership.
    state->handle_.SetOption(CURLOPT_PRIVATE, state.get());

    CURL* e = state->handle_.get();
    CURLMcode mcode = curl_multi_add_handle(thread_data.multi.get(), e);
    if (mcode == CURLM_OK) {
      // ownership successfully transferred.
      state.release();
    } else {
      // This shouldn't happen unless things have really gone pear-shaped.
      thread_data.count--;
      state->handle_.SetOption(CURLOPT_PRIVATE, nullptr);
      state->response_handler_->OnFailure(
          CurlMCodeToStatus(mcode, "in curl_multi_add_handle"));
    }
  };
}

void MultiTransportImpl::RemoveCompletedTransfers(ThreadData& thread_data) {
  // Pull pending CURLMSG_DONE events off the multi handle.
  CURLMsg* m = nullptr;
  do {
    int messages_in_queue;
    m = curl_multi_info_read(thread_data.multi.get(), &messages_in_queue);

    // Remove completed message from curl multi handle.
    if (m && m->msg == CURLMSG_DONE) {
      CURLcode result = m->data.result;
      CURL* e = m->easy_handle;
      curl_multi_remove_handle(thread_data.multi.get(), e);
      thread_data.count--;
      CurlRequestState* pvt = nullptr;
      curl_easy_getinfo(e, CURLINFO_PRIVATE, &pvt);
      assert(pvt);
      std::unique_ptr<CurlRequestState> state(pvt);
      state->handle_.SetOption(CURLOPT_PRIVATE, nullptr);
      FinishRequest(std::move(state), result);
    }
  } while (m != nullptr);
}

}  // namespace

class CurlTransport::Impl : public MultiTransportImpl {
 public:
  using MultiTransportImpl::MultiTransportImpl;
};

CurlTransport::CurlTransport(std::shared_ptr<CurlHandleFactory> factory)
    : impl_(std::make_unique<Impl>(std::move(factory),
                                   /*nthreads=*/GetHttpThreads())) {}

CurlTransport::~CurlTransport() = default;

void CurlTransport::IssueRequestWithHandler(
    const HttpRequest& request, IssueRequestOptions options,
    HttpResponseHandler* response_handler) {
  assert(impl_);

  impl_->EnqueueRequest(request, std::move(options), response_handler);
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
