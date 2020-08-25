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

#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include <curl/curl.h>
#include "tensorstore/internal/cord_util.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_http {
namespace {

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

  CurlRequestState(CurlHandleFactory* factory)
      : factory_(factory), handle_(factory->CreateHandle()) {
    InitializeCurlHandle(handle_.get());

    const auto& config = CurlEnvConfig();
    if (config.verbose) {
      CurlEasySetopt(handle_.get(), CURLOPT_VERBOSE, 1L);
      // TODO: Consider also using CURLOPT_DEBUGFUNCTION
    }

    if (const auto& x = config.ca_path) {
      CurlEasySetopt(handle_.get(), CURLOPT_CAPATH, x->c_str());
    }

    if (const auto& x = config.ca_bundle) {
      CurlEasySetopt(handle_.get(), CURLOPT_CAINFO, x->c_str());
    }

    CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA, this);
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION,
                   &CurlRequestState::CurlWriteCallback);

    CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA, this);
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION,
                   &CurlRequestState::CurlHeaderCallback);

    CurlEasySetopt(handle_.get(), CURLOPT_ERRORBUFFER, error_buffer_);
  }

  ~CurlRequestState() {
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_WRITEFUNCTION,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_READDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_READFUNCTION,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_SEEKDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_SEEKFUNCTION,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERDATA,
                   static_cast<void*>(nullptr));
    CurlEasySetopt(handle_.get(), CURLOPT_HEADERFUNCTION,
                   static_cast<void (*)()>(nullptr));

    CurlEasySetopt(handle_.get(), CURLOPT_ERRORBUFFER, nullptr);

    factory_->CleanupHandle(std::move(handle_));
  }

  void Setup(const HttpRequest& request, absl::Cord payload,
             absl::Duration request_timeout, absl::Duration connect_timeout) {
    // For thread safety, don't use signals to time out name resolves (when
    // async name resolution is not supported).
    //
    // https://curl.haxx.se/libcurl/c/threadsafe.html
    CurlEasySetopt(handle_.get(), CURLOPT_NOSIGNAL, 1L);

    std::string user_agent = request.user_agent() + GetCurlUserAgentSuffix();
    CurlEasySetopt(handle_.get(), CURLOPT_USERAGENT, user_agent.c_str());

    CurlEasySetopt(handle_.get(), CURLOPT_URL, request.url().c_str());

    // Convert headers to a curl slist
    curl_slist* head = nullptr;
    for (const std::string& h : request.headers()) {
      head = curl_slist_append(head, h.c_str());
    }
    headers_.reset(head);
    CurlEasySetopt(handle_.get(), CURLOPT_HTTPHEADER, headers_.get());
    if (request.accept_encoding()) {
      CurlEasySetopt(handle_.get(), CURLOPT_ACCEPT_ENCODING, "");
    }

    if (request_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(request_timeout);
      CurlEasySetopt(handle_.get(), CURLOPT_TIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (connect_timeout > absl::ZeroDuration()) {
      auto ms = absl::ToInt64Milliseconds(connect_timeout);
      CurlEasySetopt(handle_.get(), CURLOPT_CONNECTTIMEOUT_MS, ms > 0 ? ms : 1);
    }
    if (!request.method().empty()) {
      CurlEasySetopt(handle_.get(), CURLOPT_CUSTOMREQUEST,
                     request.method().c_str());
    }

    if (!payload.empty()) {
      payload_ = std::move(payload);
      payload_it_ = payload_.char_begin();
      payload_remaining_ = payload_.size();
      if (!request.method().empty()) {
        TENSORSTORE_LOG("Changing custom http method [", request.method(),
                        "] to POST");
      }
      CurlEasySetopt(handle_.get(), CURLOPT_POST, 1L);
      CurlEasySetopt(handle_.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                     payload_.size());
      CurlEasySetopt(handle_.get(), CURLOPT_READDATA, this);
      CurlEasySetopt(handle_.get(), CURLOPT_READFUNCTION,
                     &CurlRequestState::CurlReadCallback);
      // Seek callback allows curl to re-read input, which it sometimes needs to
      // do.
      //
      // https://curl.haxx.se/mail/lib-2010-01/0183.html
      //
      // If this is not set, curl may fail with CURLE_SEND_FAIL_REWIND.
      CurlEasySetopt(handle_.get(), CURLOPT_SEEKFUNCTION,
                     &CurlRequestState::CurlSeekCallback);
      CurlEasySetopt(handle_.get(), CURLOPT_SEEKDATA, this);
    } else if (request.method().empty()) {
      CurlEasySetopt(handle_.get(), CURLOPT_HTTPGET, 1L);
    }
  }

  void SetHTTP2() {
    CurlEasySetopt(handle_.get(), CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2_0);
  }

  size_t WriteCallback(absl::string_view data) {
    response_.payload.Append(data);
    return data.size();
  }

  size_t ReadCallback(char* data, size_t size) {
    size_t n = std::min(size, payload_remaining_);
    internal::CopyCordToSpan(payload_it_, {data, static_cast<ptrdiff_t>(n)});
    payload_remaining_ -= n;
    return n;
  }

  size_t SeekCallback(curl_off_t offset, int origin) {
    if (origin != SEEK_SET) {
      // According to the documentation:
      // https://curl.haxx.se/libcurl/c/CURLOPT_SEEKFUNCTION.html
      //
      // curl only uses SEEK_SET.
      return CURL_SEEKFUNC_CANTSEEK;
    }
    if (offset < 0 || offset > payload_.size()) {
      return CURL_SEEKFUNC_FAIL;
    }
    payload_it_ = payload_.char_begin();
    absl::Cord::Advance(&payload_it_, static_cast<size_t>(offset));
    payload_remaining_ = payload_.size() - static_cast<size_t>(offset);
    return CURL_SEEKFUNC_OK;
  }

  size_t HeaderCallback(absl::string_view data) {
    return AppendHeaderData(response_.headers, data);
  }

  void SetForbidReuse() {
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    CurlEasySetopt(handle_.get(), CURLOPT_FORBID_REUSE, 1);
  }

  Status CurlCodeToStatus(CURLcode code) {
    if (code == CURLE_OK) {
      return absl::OkStatus();
    }
    TENSORSTORE_LOG("Error [", code, "]=", curl_easy_strerror(code),
                    " in curl operation\n", error_buffer_);
    return ::tensorstore::internal_http::CurlCodeToStatus(code, error_buffer_);
  }

  static std::size_t CurlWriteCallback(void* contents, std::size_t size,
                                       std::size_t nmemb, void* userdata) {
    return static_cast<CurlRequestState*>(userdata)->WriteCallback(
        absl::string_view(static_cast<char const*>(contents), size * nmemb));
  }

  static std::size_t CurlReadCallback(void* contents, std::size_t size,
                                      std::size_t nmemb, void* userdata) {
    return static_cast<CurlRequestState*>(userdata)->ReadCallback(
        static_cast<char*>(contents), size * nmemb);
  }

  static int CurlSeekCallback(void* userdata, curl_off_t offset, int origin) {
    return static_cast<CurlRequestState*>(userdata)->SeekCallback(offset,
                                                                  origin);
  }

  static std::size_t CurlHeaderCallback(void* contents, std::size_t size,
                                        std::size_t nmemb, void* userdata) {
    return static_cast<CurlRequestState*>(userdata)->HeaderCallback(
        absl::string_view(static_cast<char const*>(contents), size * nmemb));
  }
};

class MultiTransportImpl {
 public:
  explicit MultiTransportImpl(std::shared_ptr<CurlHandleFactory> factory)
      : factory_(factory), multi_(factory_->CreateMultiHandle()) {
    thread_ = std::thread([this] { Run(); });
  }

  ~MultiTransportImpl() {
    {
      absl::MutexLock lock(&mutex_);
      curl_multi_wakeup(multi_.get());
      done_ = true;
    }
    thread_.join();
    factory_->CleanupMultiHandle(std::move(multi_));
  }

  Future<HttpResponse> StartRequest(const HttpRequest& request,
                                    absl::Cord payload,
                                    absl::Duration request_timeout,
                                    absl::Duration connect_timeout);

  void FinishRequest(CURL* e, CURLcode code);

  void Run();

  std::shared_ptr<CurlHandleFactory> factory_;
  CurlMulti multi_;

  absl::Mutex mutex_;
  std::vector<CURL*> pending_requests_;
  size_t active_requests_ = 0;

  std::thread thread_;
  bool done_ = false;
};

Future<HttpResponse> MultiTransportImpl::StartRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  auto state = absl::make_unique<CurlRequestState>(factory_.get());
  state->Setup(request, std::move(payload), request_timeout, connect_timeout);
  state->SetHTTP2();

  auto pair = PromiseFuturePair<HttpResponse>::Make();
  state->promise_ = std::move(pair.promise);

  CURL* e = state->handle_.get();
  CurlEasySetopt(e, CURLOPT_PRIVATE, static_cast<void*>(state.get()));
  state.release();

  // Add the handle to the curl_multi state.
  // TODO: Add an ExecuteWhenNotNeeded callback which removes
  // the handle from the pending / active requests set.
  {
    absl::MutexLock l(&mutex_);
    pending_requests_.emplace_back(e);
    curl_multi_wakeup(multi_.get());
  }

  return std::move(pair.future);
}

void MultiTransportImpl::FinishRequest(CURL* e, CURLcode code) {
  std::unique_ptr<CurlRequestState> state([e] {
    CurlRequestState* pvt = nullptr;
    curl_easy_getinfo(e, CURLINFO_PRIVATE, &pvt);
    CurlEasySetopt(e, CURLOPT_PRIVATE, nullptr);
    return pvt;
  }());

  if (code == CURLE_HTTP2) {
    TENSORSTORE_LOG("CURLE_HTTP2 ", state->error_buffer_);
    // If there was an error in the HTTP2 framing, try and force
    // CURL to close the connection stream.
    // https://curl.haxx.se/libcurl/c/CURLOPT_FORBID_REUSE.html
    state->SetForbidReuse();
  }

  curl_multi_remove_handle(multi_.get(), e);

  if (code != CURLE_OK) {
    state->promise_.SetResult(state->CurlCodeToStatus(code));
  } else {
    state->response_.status_code = CurlGetResponseCode(e);
    state->promise_.SetResult(std::move(state->response_));
  }
}

void MultiTransportImpl::Run() {
  int active_requests = 0;
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
          active_requests++;
        } else {
          // This shouldn't happen unless things have really gone pear-shaped.
          state->promise_.SetResult(
              CurlMCodeToStatus(mcode, "in curl_multi_add_handle"));
        }
      }
      pending_requests_.clear();

      if (active_requests == 0) {
        // Shutdown has been requested.
        if (done_) break;

        // Nothing running in the curl context yet.
        mutex_.Await(absl::Condition(
            +[](MultiTransportImpl* that) {
              return !that->pending_requests_.empty() || that->done_;
            },
            this));
        continue;
      }
    }

    // curl_multi_perform is the main curl method that performs work
    // on the existing curl handles.
    while (CURLM_CALL_MULTI_PERFORM ==
           curl_multi_perform(multi_.get(), &active_requests)) {
      /* loop */
    }

    // Try to read any messages in the queue.
    for (;;) {
      int messages_in_queue;
      const auto* m = curl_multi_info_read(multi_.get(), &messages_in_queue);
      if (!m) break;
      FinishRequest(m->easy_handle, m->data.result);
      if (messages_in_queue == 0) break;
    }

    if (active_requests > 0) {
      // Wait for more transfers to complete.  Rely on curl_multi_wakeup to
      // notify that non-transfer work is ready, otherwise wake up once per
      // timeout interval.
      const int timeout_ms = std::numeric_limits<int>::max();  // infinite
      int numfds = 0;
      CURLMcode mcode =
          curl_multi_poll(multi_.get(), nullptr, 0, timeout_ms, &numfds);
      if (mcode != CURLM_OK) {
        auto status = CurlMCodeToStatus(mcode, "in CurlMultiTransport");
        TENSORSTORE_LOG("Error [", mcode, "] ", status, "\n");
      }
    }
  }
}

}  // namespace

class CurlTransport::Impl : public MultiTransportImpl {
 public:
  using MultiTransportImpl::MultiTransportImpl;
};

CurlTransport::CurlTransport(std::shared_ptr<CurlHandleFactory> factory)
    : impl_(absl::make_unique<Impl>(std::move(factory))) {}

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
