// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/http/e2e_http2_test.h"

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/types.h>
#else
#include <BaseTsd.h>
using ssize_t = SSIZE_T;
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nghttp2/nghttp2.h>
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/http/e2e_test_util.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace tensorstore {
namespace internal_http {
namespace {

using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::StatusIs;
using ::tensorstore::transport_test_utils::LoopbackServer;
using ::tensorstore::transport_test_utils::SendAll;
using ::tensorstore::transport_test_utils::ShutdownSocket;
using ::tensorstore::transport_test_utils::ShutdownSocketSend;
using ::tensorstore::transport_test_utils::socket_t;
using ::tensorstore::transport_test_utils::WaitForRead;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Le;
using ::testing::Not;

using H2Headers = std::vector<std::pair<std::string, std::string>>;

// A response the scripted server sends on a stream.
struct H2Response {
  int status = 200;
  std::string body;
  // Response header fields in addition to :status and content-length.
  H2Headers headers;
  // When non-empty, sent as a trailing HEADERS frame carrying END_STREAM.
  H2Headers trailers;
  // When true, the headers advertise `body` but the body is never sent.
  bool stall_body = false;
};

// A request stream as seen by the scripted server policy.
struct H2StreamEvent {
  int32_t stream_id = 0;
  // Zero-based index of the connection in accept order.
  int connection_index = 0;
  // Zero-based index of this stream among streams opened on its connection.
  int stream_index_on_connection = 0;
  std::string method;
  std::string path;
  // Request body received so far.
  std::string request_body;
  // Whether the client has finished sending the request.
  bool request_end_stream_seen = false;
};

class H2Connection;

// Actions the policy may take in response to an `H2StreamEvent`.
class H2Actions {
 public:
  H2Actions(H2Connection& connection, const H2StreamEvent& event)
      : connection_(connection), event_(event) {}

  // Sends a non-final (1xx) response header block on the event's stream.
  void SendInterim(int status, H2Headers headers = {});
  // Sends a final response on the event's stream.
  void Respond(H2Response response);
  void Respond(int status, std::string body) {
    H2Response response;
    response.status = status;
    response.body = std::move(body);
    Respond(std::move(response));
  }
  // Sends a final response on a specific earlier stream on this connection.
  void RespondOnStream(int32_t stream_id, int status, std::string body);
  // Resets the event's stream with `error_code`.
  void RstStream(uint32_t error_code);
  // Sends GOAWAY and closes the connection once pending output is flushed.
  void Goaway(int32_t last_stream_id, uint32_t error_code);
  // Sends a SETTINGS frame updating MAX_CONCURRENT_STREAMS.
  void SendSettings(uint32_t max_concurrent_streams);
  // Grants flow-control credit on both the connection and the event's stream.
  void GrantWindow(uint32_t increment);

 private:
  H2Connection& connection_;
  const H2StreamEvent& event_;
};

// Decides how the server answers each request stream.
//
// The policy is invoked on the thread serving the stream's connection, so it
// may run concurrently for different connections; any state shared across
// connections must be synchronized.
using H2Policy = std::function<void(const H2StreamEvent&, H2Actions&)>;

struct ScriptedH2ServerOptions {
  // Invoke the policy when request headers arrive rather than when the
  // request is complete, so the server can respond before the upload ends.
  bool act_on_headers = false;
  // When nonzero, invoke the policy once at least this many request body
  // bytes have arrived (even before END_STREAM).
  size_t act_after_request_bytes = 0;
  uint32_t max_concurrent_streams = 100;
  uint32_t initial_window_size = 65535;
  // When false, the server never sends WINDOW_UPDATE, so uploads larger
  // than the initial window block indefinitely.
  bool replenish_window = true;
  // When nonzero, the server returns request flow-control credit in
  // WINDOW_UPDATE frames of at most this many bytes, for both the stream
  // and the connection, instead of nghttp2's batched updates.
  uint32_t window_update_increment = 0;
};

// Tracks responses sent by a `ScriptedH2Server`, shared by all connections.
class ResponseLog {
 public:
  void Record(std::string_view method, std::string_view path) {
    absl::MutexLock lock(mu_);
    ++counts_[absl::StrCat(method, " ", path)];
  }
  int Count(std::string_view method, std::string_view path) const {
    absl::MutexLock lock(mu_);
    return CountLocked(absl::StrCat(method, " ", path));
  }
  // Waits until `Count(method, path) >= count`. Returns false on timeout.
  bool AwaitCount(std::string_view method, std::string_view path, int count,
                  absl::Duration timeout) const {
    const std::string key = absl::StrCat(method, " ", path);
    auto reached = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return CountLocked(key) >= count;
    };
    absl::MutexLock lock(mu_);
    return mu_.AwaitWithTimeout(absl::Condition(&reached), timeout);
  }
  void RecordSettingsAck() {
    absl::MutexLock lock(mu_);
    ++settings_acks_;
  }
  bool AwaitSettingsAckCount(int count, absl::Duration timeout) const {
    auto reached = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      return settings_acks_ >= count;
    };
    absl::MutexLock lock(mu_);
    return mu_.AwaitWithTimeout(absl::Condition(&reached), timeout);
  }
  void RecordBytesAfterResponse(size_t bytes) {
    absl::MutexLock lock(mu_);
    bytes_after_response_ += bytes;
  }
  size_t bytes_after_response() const {
    absl::MutexLock lock(mu_);
    return bytes_after_response_;
  }

 private:
  int CountLocked(const std::string& key) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto it = counts_.find(key);
    return it == counts_.end() ? 0 : it->second;
  }

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, int> counts_ ABSL_GUARDED_BY(mu_);
  int settings_acks_ ABSL_GUARDED_BY(mu_) = 0;
  size_t bytes_after_response_ ABSL_GUARDED_BY(mu_) = 0;
};

// One server-side HTTP/2 connection, driven by nghttp2 on a worker thread.
class H2Connection {
 public:
  H2Connection(socket_t fd, int connection_index,
               const ScriptedH2ServerOptions& options, const H2Policy& policy,
               ResponseLog& log, const std::atomic<bool>& stop)
      : fd_(fd),
        connection_index_(connection_index),
        options_(options),
        policy_(policy),
        log_(log),
        stop_(stop) {}

  ~H2Connection() {
    if (session_) nghttp2_session_del(session_);
  }

  void Serve() {
    if (!Init()) return;
    char buf[16384];
    while (true) {
      if (!Flush()) break;
      if (goaway_sent_ && !nghttp2_session_want_write(session_)) {
        LingeringClose();
        return;
      }
      if (!nghttp2_session_want_read(session_) &&
          !nghttp2_session_want_write(session_)) {
        break;
      }
      if (!WaitForRead(fd_, stop_)) break;
      auto n = ::recv(fd_, buf, sizeof(buf), 0);
      if (n <= 0) break;
      if (nghttp2_session_mem_recv(session_,
                                   reinterpret_cast<const uint8_t*>(buf),
                                   static_cast<size_t>(n)) < 0) {
        break;
      }
    }
    ShutdownSocket(fd_);
  }

  void SendInterim(int32_t stream_id, int status, const H2Headers& headers) {
    std::string status_str = absl::StrCat(status);
    std::vector<nghttp2_nv> nva = {MakeNv(":status", status_str)};
    AppendNv(headers, nva);
    Succeeded(nghttp2_submit_headers(session_, NGHTTP2_FLAG_NONE, stream_id,
                                     nullptr, nva.data(), nva.size(), nullptr),
              "nghttp2_submit_headers");
  }

  void RespondOnStream(int32_t stream_id, H2Response response) {
    auto it = streams_.find(stream_id);
    if (it == streams_.end()) {
      ADD_FAILURE() << "Respond on unknown stream " << stream_id;
      return;
    }
    Stream& stream = it->second;
    stream.responded = true;
    stream.response = std::move(response);
    stream.body_offset = 0;
    const H2Response& r = stream.response;

    std::string status_str = absl::StrCat(r.status);
    std::string content_length = absl::StrCat(r.body.size());
    std::vector<nghttp2_nv> nva = {MakeNv(":status", status_str)};
    // RFC 9110 forbids Content-Length in a 204 response.
    if (r.status != 204) {
      nva.push_back(MakeNv("content-length", content_length));
    }
    AppendNv(r.headers, nva);

    // Without a data provider, END_STREAM rides on HEADERS and no DATA frames
    // follow.
    const bool headers_only =
        r.body.empty() && r.trailers.empty() && !r.stall_body;
    nghttp2_data_provider provider;
    provider.source.ptr = &stream;
    provider.read_callback = &H2Connection::ReadBody;
    if (!Succeeded(
            nghttp2_submit_response(session_, stream_id, nva.data(), nva.size(),
                                    headers_only ? nullptr : &provider),
            "nghttp2_submit_response")) {
      return;
    }
    log_.Record(stream.event.method, stream.event.path);
  }

  void Respond(const H2StreamEvent& event, H2Response response) {
    RespondOnStream(event.stream_id, std::move(response));
  }

  void RstStream(int32_t stream_id, uint32_t error_code) {
    Succeeded(nghttp2_submit_rst_stream(session_, NGHTTP2_FLAG_NONE, stream_id,
                                        error_code),
              "nghttp2_submit_rst_stream");
  }

  void Goaway(int32_t last_stream_id, uint32_t error_code) {
    if (Succeeded(nghttp2_submit_goaway(session_, NGHTTP2_FLAG_NONE,
                                        last_stream_id, error_code, nullptr, 0),
                  "nghttp2_submit_goaway")) {
      goaway_sent_ = true;
    }
  }

  void SendSettings(uint32_t max_concurrent_streams) {
    nghttp2_settings_entry settings[] = {
        {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, max_concurrent_streams},
    };
    Succeeded(nghttp2_submit_settings(session_, NGHTTP2_FLAG_NONE, settings,
                                      std::size(settings)),
              "nghttp2_submit_settings");
  }

  void GrantWindow(int32_t stream_id, uint32_t increment) {
    Flush();
    auto n = static_cast<int32_t>(increment);
    nghttp2_submit_window_update(session_, NGHTTP2_FLAG_NONE, 0, n);
    nghttp2_submit_window_update(session_, NGHTTP2_FLAG_NONE, stream_id, n);
  }

 private:
  struct Stream {
    H2StreamEvent event;
    bool dispatched = false;
    bool responded = false;
    H2Response response;
    size_t body_offset = 0;
  };

  static nghttp2_nv MakeNv(std::string_view name, std::string_view value) {
    nghttp2_nv nv;
    nv.name = reinterpret_cast<uint8_t*>(const_cast<char*>(name.data()));
    nv.value = reinterpret_cast<uint8_t*>(const_cast<char*>(value.data()));
    nv.namelen = name.size();
    nv.valuelen = value.size();
    nv.flags = NGHTTP2_NV_FLAG_NONE;
    return nv;
  }

  static void AppendNv(const H2Headers& headers, std::vector<nghttp2_nv>& nva) {
    for (const auto& [name, value] : headers) {
      nva.push_back(MakeNv(name, value));
    }
  }

  // Records a test failure if the nghttp2 call `what` returned error `rv`.
  // Server threads must not abort the test binary, so failures are
  // reported rather than CHECKed.
  static bool Succeeded(int rv, std::string_view what) {
    if (rv == 0) return true;
    ADD_FAILURE() << what << " failed: " << nghttp2_strerror(rv);
    return false;
  }

  bool Init() {
    nghttp2_session_callbacks* callbacks;
    if (!Succeeded(nghttp2_session_callbacks_new(&callbacks),
                   "nghttp2_session_callbacks_new")) {
      return false;
    }
    nghttp2_session_callbacks_set_on_begin_headers_callback(
        callbacks, &H2Connection::OnBeginHeaders);
    nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                     &H2Connection::OnHeader);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
        callbacks, &H2Connection::OnDataChunkRecv);
    nghttp2_session_callbacks_set_on_frame_recv_callback(
        callbacks, &H2Connection::OnFrameRecv);
    nghttp2_session_callbacks_set_on_stream_close_callback(
        callbacks, &H2Connection::OnStreamClose);

    nghttp2_option* option;
    if (!Succeeded(nghttp2_option_new(&option), "nghttp2_option_new")) {
      nghttp2_session_callbacks_del(callbacks);
      return false;
    }
    const bool manual_window_updates =
        !options_.replenish_window || options_.window_update_increment != 0;
    nghttp2_option_set_no_auto_window_update(option,
                                             manual_window_updates ? 1 : 0);

    int rv = nghttp2_session_server_new2(&session_, callbacks, this, option);
    nghttp2_option_del(option);
    nghttp2_session_callbacks_del(callbacks);
    if (!Succeeded(rv, "nghttp2_session_server_new2")) return false;

    nghttp2_settings_entry settings[] = {
        {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS,
         options_.max_concurrent_streams},
        {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, options_.initial_window_size},
    };
    if (!Succeeded(nghttp2_submit_settings(session_, NGHTTP2_FLAG_NONE,
                                           settings, std::size(settings)),
                   "nghttp2_submit_settings")) {
      return false;
    }
    if (options_.initial_window_size > NGHTTP2_INITIAL_CONNECTION_WINDOW_SIZE) {
      if (!Succeeded(
              nghttp2_session_set_local_window_size(
                  session_, NGHTTP2_FLAG_NONE, 0, options_.initial_window_size),
              "nghttp2_session_set_local_window_size")) {
        return false;
      }
    }
    return true;
  }

  bool Flush() {
    while (true) {
      const uint8_t* data;
      ssize_t n = nghttp2_session_mem_send(session_, &data);
      if (n < 0) return false;
      if (n == 0) return true;
      if (!SendAll(fd_, std::string_view(reinterpret_cast<const char*>(data),
                                         static_cast<size_t>(n)))) {
        return false;
      }
    }
  }

  // Half-closes and drains input before closing so that unread client bytes
  // do not turn the close into a TCP RST that discards the GOAWAY.
  void LingeringClose() {
    ShutdownSocketSend(fd_);
    char buf[4096];
    absl::Time deadline = absl::Now() + absl::Seconds(2);
    while (WaitForRead(fd_, stop_, deadline)) {
      if (::recv(fd_, buf, sizeof(buf), 0) <= 0) break;
    }
    ShutdownSocket(fd_);
  }

  void MaybeDispatch(Stream& stream) {
    if (stream.dispatched) return;
    const bool ready_by_bytes =
        options_.act_after_request_bytes > 0 &&
        stream.event.request_body.size() >= options_.act_after_request_bytes;
    if (!options_.act_on_headers && !ready_by_bytes &&
        !stream.event.request_end_stream_seen) {
      return;
    }
    stream.dispatched = true;
    H2Actions actions(*this, stream.event);
    policy_(stream.event, actions);
  }

  // Returns `length` bytes of flow-control credit in small increments.
  void DribbleWindowUpdates(int32_t stream_id, size_t length) {
    const uint32_t step = options_.window_update_increment;
    if (!options_.replenish_window || step == 0) return;
    while (length > 0) {
      auto n = static_cast<int32_t>(std::min<size_t>(length, step));
      length -= n;
      // Fails harmlessly once the stream is closed.
      nghttp2_submit_window_update(session_, NGHTTP2_FLAG_NONE, stream_id, n);
      nghttp2_submit_window_update(session_, NGHTTP2_FLAG_NONE, 0, n);
    }
  }

  static int OnBeginHeaders(nghttp2_session*, const nghttp2_frame* frame,
                            void* user_data) {
    auto* self = static_cast<H2Connection*>(user_data);
    if (frame->hd.type != NGHTTP2_HEADERS ||
        frame->headers.cat != NGHTTP2_HCAT_REQUEST) {
      return 0;
    }
    Stream& stream = self->streams_[frame->hd.stream_id];
    stream.event.stream_id = frame->hd.stream_id;
    stream.event.connection_index = self->connection_index_;
    stream.event.stream_index_on_connection = self->next_stream_index_++;
    return 0;
  }

  static int OnHeader(nghttp2_session*, const nghttp2_frame* frame,
                      const uint8_t* name, size_t namelen, const uint8_t* value,
                      size_t valuelen, uint8_t, void* user_data) {
    auto* self = static_cast<H2Connection*>(user_data);
    auto it = self->streams_.find(frame->hd.stream_id);
    if (it == self->streams_.end()) return 0;
    std::string_view n(reinterpret_cast<const char*>(name), namelen);
    std::string_view v(reinterpret_cast<const char*>(value), valuelen);
    if (n == ":method") {
      it->second.event.method = std::string(v);
    } else if (n == ":path") {
      it->second.event.path = std::string(v);
    }
    return 0;
  }

  static int OnDataChunkRecv(nghttp2_session*, uint8_t, int32_t stream_id,
                             const uint8_t* data, size_t len, void* user_data) {
    auto* self = static_cast<H2Connection*>(user_data);
    auto it = self->streams_.find(stream_id);
    if (it != self->streams_.end()) {
      if (it->second.responded) {
        self->log_.RecordBytesAfterResponse(len);
      }
      it->second.event.request_body.append(reinterpret_cast<const char*>(data),
                                           len);
    }
    self->DribbleWindowUpdates(stream_id, len);
    return 0;
  }

  static int OnFrameRecv(nghttp2_session*, const nghttp2_frame* frame,
                         void* user_data) {
    auto* self = static_cast<H2Connection*>(user_data);
    if (frame->hd.type == NGHTTP2_SETTINGS &&
        (frame->hd.flags & NGHTTP2_FLAG_ACK) != 0) {
      self->log_.RecordSettingsAck();
      return 0;
    }
    if (frame->hd.type != NGHTTP2_HEADERS && frame->hd.type != NGHTTP2_DATA) {
      return 0;
    }
    auto it = self->streams_.find(frame->hd.stream_id);
    if (it == self->streams_.end()) return 0;
    if (frame->hd.flags & NGHTTP2_FLAG_END_STREAM) {
      it->second.event.request_end_stream_seen = true;
    }
    self->MaybeDispatch(it->second);
    return 0;
  }

  static int OnStreamClose(nghttp2_session*, int32_t stream_id, uint32_t,
                           void* user_data) {
    static_cast<H2Connection*>(user_data)->streams_.erase(stream_id);
    return 0;
  }

  static ssize_t ReadBody(nghttp2_session* session, int32_t stream_id,
                          uint8_t* buf, size_t length, uint32_t* data_flags,
                          nghttp2_data_source* source, void*) {
    auto* stream = static_cast<Stream*>(source->ptr);
    const H2Response& r = stream->response;
    if (r.stall_body) return NGHTTP2_ERR_DEFERRED;
    size_t n = std::min(length, r.body.size() - stream->body_offset);
    memcpy(buf, r.body.data() + stream->body_offset, n);
    stream->body_offset += n;
    if (stream->body_offset == r.body.size()) {
      *data_flags |= NGHTTP2_DATA_FLAG_EOF;
      if (!r.trailers.empty()) {
        *data_flags |= NGHTTP2_DATA_FLAG_NO_END_STREAM;
        std::vector<nghttp2_nv> nva;
        AppendNv(r.trailers, nva);
        if (nghttp2_submit_trailer(session, stream_id, nva.data(),
                                   nva.size()) != 0) {
          return NGHTTP2_ERR_TEMPORAL_CALLBACK_FAILURE;
        }
      }
    }
    return static_cast<ssize_t>(n);
  }

  socket_t fd_;
  int connection_index_;
  const ScriptedH2ServerOptions& options_;
  const H2Policy& policy_;
  ResponseLog& log_;
  const std::atomic<bool>& stop_;
  nghttp2_session* session_ = nullptr;
  // std::map keeps `Stream` addresses stable for the body data provider.
  std::map<int32_t, Stream> streams_;
  int next_stream_index_ = 0;
  bool goaway_sent_ = false;
};

void H2Actions::SendInterim(int status, H2Headers headers) {
  connection_.SendInterim(event_.stream_id, status, headers);
}

void H2Actions::Respond(H2Response response) {
  connection_.Respond(event_, std::move(response));
}

void H2Actions::RespondOnStream(int32_t stream_id, int status,
                                std::string body) {
  H2Response response;
  response.status = status;
  response.body = std::move(body);
  connection_.RespondOnStream(stream_id, std::move(response));
}

void H2Actions::RstStream(uint32_t error_code) {
  connection_.RstStream(event_.stream_id, error_code);
}

void H2Actions::Goaway(int32_t last_stream_id, uint32_t error_code) {
  connection_.Goaway(last_stream_id, error_code);
}

void H2Actions::SendSettings(uint32_t max_concurrent_streams) {
  connection_.SendSettings(max_concurrent_streams);
}

void H2Actions::GrantWindow(uint32_t increment) {
  connection_.GrantWindow(event_.stream_id, increment);
}

// Loopback HTTP/2 cleartext (prior knowledge) server whose per-stream
// behavior is scripted by an `H2Policy`.
//
// Do not add TLS here: see `Http2TransportTest`. Tests that need TLS,
// ALPN or an nghttp2-independent peer belong in `H2InteropTransportTest`.
class ScriptedH2Server {
 public:
  ScriptedH2Server(ScriptedH2ServerOptions options, H2Policy policy)
      : options_(std::move(options)),
        policy_(std::move(policy)),
        server_([this](socket_t fd, int connection_index,
                       const std::atomic<bool>& stop) {
          H2Connection(fd, connection_index, options_, policy_, log_, stop)
              .Serve();
        }) {}

  std::string base_url() const {
    return absl::StrCat("http://", server_.hostport());
  }

  HttpRequest Request(std::string_view method, std::string_view path,
                      RequestHeaders headers = {}) const {
    return MakeRequest(base_url(), method, path, headers);
  }

  int accepted_connections() const { return server_.accepted_connections(); }

  int ResponseCount(std::string_view method, std::string_view path) const {
    return log_.Count(method, path);
  }

  // Waits until at least `count` responses to `method` `path` were sent.
  // Returns false on timeout.
  bool AwaitResponseCount(std::string_view method, std::string_view path,
                          int count, absl::Duration timeout) const {
    return log_.AwaitCount(method, path, count, timeout);
  }

  bool AwaitSettingsAckCount(int count, absl::Duration timeout) const {
    return log_.AwaitSettingsAckCount(count, timeout);
  }

  size_t bytes_after_response() const { return log_.bytes_after_response(); }

 private:
  const ScriptedH2ServerOptions options_;
  const H2Policy policy_;
  ResponseLog log_;
  // Declared last: its destructor joins the handlers that use the above.
  LoopbackServer server_;
};

// Deadline for a request whose response body the server deliberately
// stalls; the test expects the deadline to fire.
constexpr absl::Duration kStallDeadline = absl::Milliseconds(500);

IssueRequestOptions H2Options(absl::Duration timeout) {
  return IssueRequestOptions()
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp2PriorKnowledge)
      .SetRequestTimeout(timeout);
}

IssueRequestOptions H2Options(absl::Cord payload, absl::Duration timeout) {
  return IssueRequestOptions(std::move(payload))
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp2PriorKnowledge)
      .SetRequestTimeout(timeout);
}

Http2TestOptions SingleConnectionPerHost() {
  Http2TestOptions options;
  options.max_connections_per_host = 1;
  return options;
}

// Verifies that a stream whose final response arrived while its upload was
// still blocked on flow control stops counting against the server's
// MAX_CONCURRENT_STREAMS, so a following request is not starved.
TEST_P(Http2TransportTest, EarlyFinalResponseReleasesStreamSlot) {
  if (!supports_early_final_response()) {
    GTEST_SKIP()
        << "Transport does not support early final response during upload";
  }
  ScriptedH2ServerOptions server_options;
  server_options.act_on_headers = true;
  server_options.max_concurrent_streams = 1;
  server_options.replenish_window = false;
  ScriptedH2Server server(server_options,
                          [](const H2StreamEvent& event, H2Actions& actions) {
                            if (event.method == "PUT") {
                              actions.Respond(413, "too big");
                            } else {
                              actions.Respond(200, "ok");
                            }
                          });
  auto transport = CreateTransport(SingleConnectionPerHost());

  // The 1 MiB body exceeds the 64 KiB window the server never replenishes.
  auto put = transport->IssueRequest(
      server.Request("PUT", "/upload"),
      H2Options(absl::Cord(std::string(1 << 20, 'x')), request_timeout()));
  // Wait until the server has answered the PUT, so that the GET can only
  // proceed if the transport releases the PUT's stream slot.
  ASSERT_TRUE(
      server.AwaitResponseCount("PUT", "/upload", 1, hang_detect_timeout()));

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/get"),
                                 H2Options(hang_detect_timeout()))
                  .result(),
              HasStatusAndPayload(200, "ok"));
}

class ReentrantBatchHandler final : public HttpResponseHandler {
 public:
  ReentrantBatchHandler(std::shared_ptr<HttpTransport> transport,
                        HttpRequest follow_up_req,
                        IssueRequestOptions follow_up_opts)
      : transport_(std::move(transport)),
        follow_up_req_(std::move(follow_up_req)),
        follow_up_opts_(std::move(follow_up_opts)) {}

  void OnStatus(int32_t status_code) override {
    events_.push_back(absl::StrCat("status:", status_code));
  }
  void OnResponseHeader(std::string_view name,
                        std::string_view value) override {}
  void OnHeaderBlockDone() override {}
  void OnResponseBody(std::string_view data) override {
    received_bytes_ += data.size();
    if (!issued_reentrant_) {
      issued_reentrant_ = true;
      reentrant_future_ =
          transport_->IssueRequest(follow_up_req_, follow_up_opts_);
    }
  }
  void OnComplete() override {
    events_.push_back("complete");
    done_.Notify();
  }
  void OnFailure(absl::Status status) override {
    ABSL_LOG(ERROR) << "Test failed with status: " << status;
    status_ = std::move(status);
    done_.Notify();
  }

  void Wait() { done_.WaitForNotification(); }
  size_t received_bytes() const { return received_bytes_; }
  const std::vector<std::string>& events() const { return events_; }
  Future<HttpResponse>& reentrant_future() { return reentrant_future_; }

 private:
  std::shared_ptr<HttpTransport> transport_;
  HttpRequest follow_up_req_;
  IssueRequestOptions follow_up_opts_;
  bool issued_reentrant_ = false;
  size_t received_bytes_ = 0;
  std::vector<std::string> events_;
  absl::Status status_;
  Future<HttpResponse> reentrant_future_;
  absl::Notification done_;
};

TEST_P(Http2TransportTest,
       BatchedDrainReplenishesDownloadWindowAndPreservesReentrantOrder) {
  if (!supports_reentrant_requests()) {
    GTEST_SKIP()
        << "Transport does not support reentrant requests on a multiplexed "
           "connection";
  }
  constexpr size_t kLargeBodyBytes =
      20 * 1024 * 1024;  // > 16 MiB stream window
  const std::string large_body(kLargeBodyBytes, 'Z');

  ScriptedH2ServerOptions server_options;
  // Use default replenish_window = true so large download succeeds.
  ScriptedH2Server server(server_options,
                          [&](const H2StreamEvent& event, H2Actions& actions) {
                            if (event.path == "/large") {
                              actions.Respond(200, large_body);
                            } else {
                              actions.Respond(200, "reentrant-ok");
                            }
                          });

  auto transport = CreateTransport(SingleConnectionPerHost());

  ReentrantBatchHandler handler(transport, server.Request("GET", "/follow"),
                                H2Options(request_timeout()));
  transport->IssueRequestWithHandler(server.Request("GET", "/large"),
                                     H2Options(request_timeout()), &handler);
  handler.Wait();

  EXPECT_THAT(handler.received_bytes(), ::testing::Eq(kLargeBodyBytes));
  EXPECT_THAT(handler.events(),
              ::testing::ElementsAre("status:200", "complete"));
  EXPECT_THAT(handler.reentrant_future().result(),
              ::testing::Optional(::testing::Field(
                  &HttpResponse::payload, absl::Cord("reentrant-ok"))));
}

// Answers the first stream on each connection, then sends GOAWAY naming that
// stream as the last one processed, leaving every later stream unprocessed.
void RespondOnceThenGoaway(const H2StreamEvent& event, H2Actions& actions) {
  if (event.stream_index_on_connection != 0) return;
  actions.Respond(200, event.path);
  actions.Goaway(event.stream_id, NGHTTP2_NO_ERROR);
}

TEST_P(Http2TransportTest,
       DecompressionErrorsDoNotLeakConnectionFlowControlWindow) {
  if (!supports_multiplexed_stream_reset()) {
    GTEST_SKIP() << "Transport drains connection on stream decompression error";
  }
  ScriptedH2Server server(
      {}, [&](const H2StreamEvent& event, H2Actions& actions) {
        if (event.path == "/corrupt") {
          H2Response resp;
          resp.status = 200;
          resp.headers = {{"content-encoding", "gzip"}};
          // 64 KiB corrupt body across 520 streams (> 32 MiB connection
          // window, while staying below nghttp2's 1000-reset flood limit).
          resp.body = std::string(65536, 'X');
          actions.Respond(std::move(resp));
        } else {
          actions.Respond(200, std::string(32 * 1024 * 1024, 'A'));
        }
      });

  constexpr int kFailedStreams = 520;
  auto transport = CreateTransport(SingleConnectionPerHost());

  constexpr int kBatchSize = 64;
  for (int base = 0; base < kFailedStreams; base += kBatchSize) {
    const int count = std::min(kBatchSize, kFailedStreams - base);
    std::vector<Future<HttpResponse>> futures;
    futures.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto req = server.Request("GET", "/corrupt");
      req.accept_encoding = true;
      futures.push_back(transport->IssueRequest(std::move(req),
                                                H2Options(request_timeout())));
    }
    for (auto& f : futures) {
      EXPECT_THAT(f.result(), StatusIs(AnyOf(absl::StatusCode::kUnavailable,
                                             absl::StatusCode::kUnknown)));
    }
  }

  auto req = server.Request("GET", "/ok");
  req.accept_encoding = true;
  EXPECT_THAT(
      transport->IssueRequest(std::move(req), H2Options(absl::Seconds(10)))
          .result(),
      ::testing::Optional(AllOf(
          ::testing::Field(&HttpResponse::status_code, 200),
          ::testing::Field(&HttpResponse::payload,
                           absl::Cord(std::string(32 * 1024 * 1024, 'A'))))));
  EXPECT_THAT(server.accepted_connections(), ::testing::Eq(1));
}

constexpr int kGoawayRequests = 4;

// Issues `kGoawayRequests` concurrent GETs to /r0../rN and waits for all.
std::vector<Result<HttpResponse>> IssueConcurrentGets(
    HttpTransport& transport, const ScriptedH2Server& server,
    absl::Duration timeout) {
  std::vector<Future<HttpResponse>> futures;
  for (int i = 0; i < kGoawayRequests; ++i) {
    futures.push_back(transport.IssueRequest(
        server.Request("GET", absl::StrCat("/r", i)), H2Options(timeout)));
  }
  std::vector<Result<HttpResponse>> results;
  for (auto& future : futures) {
    results.push_back(future.result());
  }
  return results;
}

// Verifies that requests left unprocessed by a GOAWAY are retried on a new
// connection, including when the replacement connection is also drained.
TEST_P(Http2TransportTest, QueuedRequestsSurviveRepeatedGoaway) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 retry on GOAWAY";
  }
  ScriptedH2Server server({}, &RespondOnceThenGoaway);
  auto transport = CreateTransport(SingleConnectionPerHost());

  EXPECT_THAT(IssueConcurrentGets(*transport, server, request_timeout()),
              Each(HasStatusCode(200)));
}

// Verifies that the GOAWAY retry path sends each request exactly once.
TEST_P(Http2TransportTest, RepeatedGoawayDoesNotDuplicateRequests) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 retry on GOAWAY";
  }
  ScriptedH2Server server({}, &RespondOnceThenGoaway);
  auto transport = CreateTransport(SingleConnectionPerHost());
  (void)IssueConcurrentGets(*transport, server, request_timeout());

  std::vector<int> response_counts;
  for (int i = 0; i < kGoawayRequests; ++i) {
    response_counts.push_back(
        server.ResponseCount("GET", absl::StrCat("/r", i)));
  }
  // Delivery is asserted by QueuedRequestsSurviveRepeatedGoaway; this test
  // only checks that no request reached the server more than once.
  EXPECT_THAT(response_counts, Each(Le(1)));
}

// Answers GET /ping with 200 so tests can warm the connection pool, and
// delegates every other stream to `policy`.
H2Policy WithPing(H2Policy policy) {
  return [policy = std::move(policy)](const H2StreamEvent& event,
                                      H2Actions& actions) {
    if (event.path == "/ping") {
      actions.Respond(200, "pong");
      return;
    }
    policy(event, actions);
  };
}

// Leaves `transport` holding a pooled connection that the next request
// reuses. CurlTransport only retries REFUSED_STREAM on reused connections.
void WarmConnection(HttpTransport& transport, const ScriptedH2Server& server,
                    absl::Duration timeout) {
  ASSERT_THAT(
      transport.IssueRequest(server.Request("GET", "/ping"), H2Options(timeout))
          .result(),
      HasStatusCode(200));
}

// Resets every stream other than /ping with `error_code`.
H2Policy AlwaysReset(uint32_t error_code) {
  return WithPing([error_code](const H2StreamEvent&, H2Actions& actions) {
    actions.RstStream(error_code);
  });
}

// Resets the first stream other than /ping with `error_code` and answers
// every later one with 200 "retry_succeeded", so a transparent retry is
// observable as success.
H2Policy ResetFirstAttempt(uint32_t error_code) {
  auto attempts = std::make_shared<std::atomic<int>>(0);
  return WithPing(
      [error_code, attempts](const H2StreamEvent&, H2Actions& actions) {
        if (attempts->fetch_add(1) == 0) {
          actions.RstStream(error_code);
        } else {
          actions.Respond(200, "retry_succeeded");
        }
      });
}

// Verifies transparent request retry when server resets stream with REFUSED.
TEST_P(Http2TransportTest, StreamResetRefusedTriggersTransparentRetry) {
  if (!supports_stream_retry()) {
    GTEST_SKIP() << "Transport does not support transparent HTTP/2 retry on "
                    "REFUSED_STREAM";
  }
  ScriptedH2Server server({}, ResetFirstAttempt(NGHTTP2_REFUSED_STREAM));
  auto transport = CreateTransport();
  WarmConnection(*transport, server, request_timeout());

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/refused"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("retry_succeeded")));
}

// Verifies server stream reset with CANCEL propagates error status.
TEST_P(Http2TransportTest, StreamResetCancelReturnsErrorStatus) {
  ScriptedH2Server server({}, AlwaysReset(NGHTTP2_CANCEL));
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/cancel"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(AnyOf(absl::StatusCode::kCancelled,
                             absl::StatusCode::kUnavailable)));
}

// Verifies server reset with ENHANCE_YOUR_CALM propagates quota/error status.
TEST_P(Http2TransportTest, StreamResetEnhanceYourCalmReturnsErrorStatus) {
  ScriptedH2Server server({}, AlwaysReset(NGHTTP2_ENHANCE_YOUR_CALM));
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/calm"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(AnyOf(absl::StatusCode::kResourceExhausted,
                             absl::StatusCode::kUnavailable)));
}

// Verifies a stream reset other than REFUSED_STREAM on a reused connection
// is not transparently retried, since the server may have processed it.
TEST_P(Http2TransportTest,
       StreamResetInternalErrorOnReusedConnectionNotRetried) {
  ScriptedH2Server server({}, ResetFirstAttempt(NGHTTP2_INTERNAL_ERROR));
  auto transport = CreateTransport();
  WarmConnection(*transport, server, request_timeout());

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("POST", "/internal"),
                                 H2Options(absl::Cord("non-idempotent"),
                                           request_timeout()))
                  .result(),
              Not(IsOk()));
}

// Verifies REFUSED_STREAM is retried for a non-idempotent request with a
// body, since the server guarantees it did not process the stream.
TEST_P(Http2TransportTest, StreamResetRefusedRetriesPostWithBody) {
  if (!supports_stream_retry()) {
    GTEST_SKIP() << "Transport does not support transparent HTTP/2 retry on "
                    "REFUSED_STREAM";
  }
  ScriptedH2Server server({}, ResetFirstAttempt(NGHTTP2_REFUSED_STREAM));
  auto transport = CreateTransport();
  WarmConnection(*transport, server, request_timeout());

  EXPECT_THAT(
      transport
          ->IssueRequest(server.Request("POST", "/refused"),
                         H2Options(absl::Cord("body"), request_timeout()))
          .result(),
      HasStatusAndPayload(200, absl::Cord("retry_succeeded")));
}

// Answers on request HEADERS with 413 and never grants upload credit, so
// the client's upload stays blocked after the final response.
ScriptedH2ServerOptions EarlyResponseOptions() {
  ScriptedH2ServerOptions options;
  options.act_on_headers = true;
  options.replenish_window = false;
  return options;
}

void Reject413(const H2StreamEvent&, H2Actions& actions) {
  actions.Respond(413, "too big");
}

// Verifies a final response that arrives while the upload is blocked by
// flow control completes instead of hanging until the deadline.
TEST_P(Http2TransportTest, EarlyFinalResponseDuringBlockedUploadCompletes) {
  ScriptedH2Server server(EarlyResponseOptions(), &Reject413);
  auto transport = CreateTransport();

  // 1 MiB exceeds the 65,535-byte initial window.
  EXPECT_THAT(
      transport
          ->IssueRequest(server.Request("PUT", "/blocked"),
                         H2Options(absl::Cord(std::string(1 << 20, 'x')),
                                   hang_detect_timeout()))
          .result(),
      HasStatusAndPayload(413, absl::Cord("too big")));
}

// Verifies a final response to an Expect: 100-continue request completes
// even though the server never sends 100 or RST_STREAM.
TEST_P(Http2TransportTest, EarlyFinalResponseToExpectContinueCompletes) {
  ScriptedH2Server server(EarlyResponseOptions(), &Reject413);
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("PUT", "/expect",
                                                {{"expect", "100-continue"}}),
                                 H2Options(absl::Cord(std::string(1024, 'x')),
                                           hang_detect_timeout()))
                  .result(),
              HasStatusAndPayload(413, "too big"));
}

// Verifies graceful GOAWAY drains current stream and next request redials.
TEST_P(Http2TransportTest,
       GracefulGoawayDrainsConnectionAndDialsNewConnection) {
  ScriptedH2Server server({},
                          [](const H2StreamEvent& event, H2Actions& actions) {
                            if (event.path == "/goaway") {
                              actions.Respond(200, "graceful_goaway_ok");
                              actions.Goaway(event.stream_id, NGHTTP2_NO_ERROR);
                            } else {
                              actions.Respond(404, "");
                            }
                          });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/goaway"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("graceful_goaway_ok")));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/missing"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusCode(404));
  EXPECT_THAT(server.accepted_connections(), Eq(2));
}

// Verifies 100 Continue informational header is handled before final status.
TEST_P(Http2TransportTest, Interim100ContinueDeliversFinalStatusOnly) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    actions.SendInterim(100);
    actions.Respond(200, "continue_ok");
  });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/interim"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("continue_ok")));
}

// Verifies multiple 1xx header blocks (100, 103) are ignored before final 200.
TEST_P(Http2TransportTest, MultipleInterim1xxHeadersDeliverFinalStatusOnly) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    actions.SendInterim(100, {{"x-interim-1", "ignore-me"}});
    actions.SendInterim(103, {{"link", "</style.css>; rel=preload"}});
    actions.Respond(200, "multi_1xx_ok");
  });
  auto transport = CreateTransport();

  EXPECT_THAT(
      transport
          ->IssueRequest(server.Request("GET", "/interim"),
                         H2Options(request_timeout()))
          .result(),
      AllOf(HasStatusAndPayload(200, absl::Cord("multi_1xx_ok")),
            LacksResponseHeader("x-interim-1"), LacksResponseHeader("link")));
}

// Verifies 204 No Content with END_STREAM on HEADERS and zero DATA frames.
TEST_P(Http2TransportTest, NoContent204WithoutDataFrames) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    H2Response response;
    response.status = 204;
    response.headers = {{"x-no-content", "true"}};
    actions.Respond(std::move(response));
  });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/no_content"),
                                 H2Options(request_timeout()))
                  .result(),
              AllOf(HasStatusAndPayload(204, absl::Cord("")),
                    HasResponseHeader("x-no-content", "true")));
}

// Verifies uploads exceeding the 65,535-byte initial HTTP/2 window pause and
// resume across small 4 KiB WINDOW_UPDATE frames, and arrive intact.
TEST_P(Http2TransportTest, UploadFlowControlWindowDribble) {
  ScriptedH2ServerOptions server_options;
  server_options.window_update_increment = 4096;
  ScriptedH2Server server(server_options,
                          [](const H2StreamEvent& event, H2Actions& actions) {
                            actions.Respond(200, event.request_body);
                          });
  auto transport = CreateTransport();

  absl::Cord payload;
  for (int i = 0; i < 4; ++i) {
    payload.Append(std::string(32768, 'A' + i));
  }
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("PUT", "/upload"),
                                 H2Options(payload, request_timeout()))
                  .result(),
              HasStatusAndPayload(200, payload));
}

// Verifies trailing headers after response body do not disrupt payload.
TEST_P(Http2TransportTest, TrailersDoNotDisruptResponseDelivery) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    H2Response response;
    response.body = "payload-with-trailers";
    response.trailers = {{"x-checksum", "sha256-abc"}};
    actions.Respond(std::move(response));
  });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/trailers"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("payload-with-trailers")));
}

// Verifies request timeout fires when server delivers headers but stalls body.
TEST_P(Http2TransportTest, StalledBodyTriggersRequestTimeout) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    H2Response response;
    response.body = std::string(1000, 'x');
    response.stall_body = true;
    actions.Respond(std::move(response));
  });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/stall"),
                                 H2Options(kStallDeadline))
                  .result(),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

// Verifies safe teardown when handler resets transport reference in OnComplete.
TEST_P(Http2TransportTest, HandlerDestroysTransportInOnComplete) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    actions.Respond(200, "ok");
  });
  auto transport = CreateTransport();

  auto [promise, future] = PromiseFuturePair<int32_t>::Make();
  TransportDroppingHandler handler(&transport, std::move(promise));
  transport->IssueRequestWithHandler(server.Request("GET", "/drop"),
                                     H2Options(request_timeout()), &handler);

  EXPECT_THAT(future.result(), IsOkAndHolds(200));
}

// Verifies a multi-MiB upload against a server advertising a 16 MiB initial
// stream and connection window completes with the full payload intact.
TEST_P(Http2TransportTest,
       LargeUploadWithLargeInitialWindowStreamsInBoundedChunks) {
  ScriptedH2ServerOptions server_options;
  server_options.initial_window_size = 16 << 20;
  server_options.replenish_window = false;
  ScriptedH2Server server(server_options, [](const H2StreamEvent& event,
                                             H2Actions& actions) {
    actions.Respond(200, absl::StrCat("received:", event.request_body.size()));
  });
  auto transport = CreateTransport();

  constexpr size_t kUploadSize = 4 << 20;
  absl::Cord payload(std::string(kUploadSize, 'U'));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("PUT", "/large_upload"),
                                 H2Options(payload, request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("received:4194304")));
}

std::string MakeMultiMemberGzipBomb(int num_members) {
  absl::Cord out;
  zlib::Options opts;
  opts.use_gzip_header = true;
  std::string member_plain(1024, 'A');
  zlib::Encode(absl::Cord(member_plain), &out, opts);
  std::string member = std::string(out);
  std::string result;
  for (int i = 0; i < num_members; ++i) {
    result += member;
  }
  return result;
}

TEST_P(Http2TransportTest, MultiMemberGzipBombRejected) {
  std::string bomb = MakeMultiMemberGzipBomb(/*num_members=*/64);
  ScriptedH2Server server({},
                          [&bomb](const H2StreamEvent&, H2Actions& actions) {
                            H2Response response;
                            response.status = 200;
                            response.headers = {{"content-encoding", "gzip"}};
                            response.body = bomb;
                            actions.Respond(std::move(response));
                          });
  auto transport = CreateTransport();
  HttpRequest request = server.Request("GET", "/gzip_bomb");
  request.accept_encoding = true;

  EXPECT_THAT(
      transport->IssueRequest(request, H2Options(request_timeout())).result(),
      ::testing::Not(IsOk()));
}

TEST_P(Http2TransportTest,
       StreamResetInternalErrorDoesNotDrainMultiplexedConnection) {
  if (!supports_multiplexed_stream_reset()) {
    GTEST_SKIP()
        << "Transport does not support reusing multiplexed connection after "
           "stream reset";
  }
  ScriptedH2Server server({},
                          [](const H2StreamEvent& event, H2Actions& actions) {
                            if (event.path == "/fail") {
                              actions.RstStream(NGHTTP2_INTERNAL_ERROR);
                            } else {
                              actions.Respond(200, "ok");
                            }
                          });
  auto transport = CreateTransport(SingleConnectionPerHost());

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/fail"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(AnyOf(absl::StatusCode::kInternal,
                             absl::StatusCode::kUnavailable)));

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/ok"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("ok")));
  EXPECT_THAT(server.accepted_connections(), ::testing::Eq(1));
}

TEST_P(Http2TransportTest,
       CorruptContentEncodingOnOneStreamDoesNotDrainConnection) {
  if (!supports_multiplexed_stream_reset()) {
    GTEST_SKIP()
        << "Transport does not support reusing multiplexed connection after "
           "stream decompression error";
  }
  ScriptedH2Server server({},
                          [](const H2StreamEvent& event, H2Actions& actions) {
                            H2Response resp;
                            resp.status = 200;
                            if (event.path == "/bad-gzip") {
                              resp.headers = {{"content-encoding", "gzip"}};
                              resp.body = "not-a-valid-gzip-stream";
                            } else {
                              resp.body = "healthy-stream";
                            }
                            actions.Respond(std::move(resp));
                          });
  auto transport = CreateTransport(SingleConnectionPerHost());

  HttpRequest bad_req = server.Request("GET", "/bad-gzip");
  bad_req.accept_encoding = true;
  EXPECT_THAT(
      transport->IssueRequest(bad_req, H2Options(request_timeout())).result(),
      StatusIs(AnyOf(absl::StatusCode::kDataLoss, absl::StatusCode::kInternal,
                     absl::StatusCode::kUnavailable,
                     absl::StatusCode::kUnknown)));

  HttpRequest ok_req = server.Request("GET", "/ok");
  ok_req.accept_encoding = true;
  EXPECT_THAT(
      transport->IssueRequest(ok_req, H2Options(request_timeout())).result(),
      HasStatusAndPayload(200, absl::Cord("healthy-stream")));
  EXPECT_THAT(server.accepted_connections(), ::testing::Eq(1));
}

struct HandlerSequenceResult {
  bool status_seen = false;
  bool body_before_status = false;
  absl::Status final_status;
};

void PrintTo(const HandlerSequenceResult& r, std::ostream* os) {
  *os << "{status_seen=" << r.status_seen
      << ", body_before_status=" << r.body_before_status
      << ", final_status=" << r.final_status << "}";
}

MATCHER(FailedWithoutBodyBeforeStatus, "") {
  return ExplainMatchResult(
      AllOf(
          ::testing::Field(&HandlerSequenceResult::body_before_status,
                           Eq(false)),
          ::testing::Field(&HandlerSequenceResult::final_status, Not(IsOk()))),
      arg, result_listener);
}

TEST_P(Http2TransportTest,
       StatusBelow100RejectedWithoutDeliveringBodyBeforeStatus) {
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    actions.Respond(99, "bad-body");
  });
  auto transport = CreateTransport();

  class SequenceHandler final : public HttpResponseHandler {
   public:
    explicit SequenceHandler(HandlerSequenceResult* out,
                             absl::Notification* done)
        : out_(out), done_(done) {}
    void OnStatus(int32_t) override { out_->status_seen = true; }
    void OnResponseHeader(std::string_view, std::string_view) override {}
    void OnHeaderBlockDone() override {}
    void OnResponseBody(std::string_view) override {
      if (!out_->status_seen) out_->body_before_status = true;
    }
    void OnComplete() override { done_->Notify(); }
    void OnFailure(absl::Status s) override {
      out_->final_status = std::move(s);
      done_->Notify();
    }

   private:
    HandlerSequenceResult* out_;
    absl::Notification* done_;
  };

  HandlerSequenceResult result;
  absl::Notification done;
  SequenceHandler handler(&result, &done);
  transport->IssueRequestWithHandler(server.Request("GET", "/status99"),
                                     H2Options(request_timeout()), &handler);
  ASSERT_TRUE(done.WaitForNotificationWithTimeout(request_timeout()));
  EXPECT_THAT(result, FailedWithoutBodyBeforeStatus());
}

TEST_P(Http2TransportTest, RejectsDecompressionBombOverHttp2) {
  if (!supports_decompressed_size_limit()) {
    GTEST_SKIP()
        << "Transport does not support custom max_decompressed_response_bytes";
  }
  absl::Cord compressed_cord;
  zlib::Options z_opts;
  z_opts.use_gzip_header = true;
  zlib::Encode(absl::Cord(std::string(256 * 1024, '\0')), &compressed_cord,
               z_opts);
  const std::string bomb_payload(compressed_cord);

  ScriptedH2Server server(
      {}, [&bomb_payload](const H2StreamEvent&, H2Actions& actions) {
        H2Response resp;
        resp.status = 200;
        resp.headers = {{"content-encoding", "gzip"}};
        resp.body = bomb_payload;
        actions.Respond(std::move(resp));
      });

  Http2TestOptions t_opts;
  t_opts.max_decompressed_response_bytes = 64 * 1024;
  auto transport = CreateTransport(t_opts);

  HttpRequest req = server.Request("GET", "/bomb");
  req.accept_encoding = true;
  EXPECT_THAT(
      transport->IssueRequest(req, H2Options(request_timeout())).result(),
      StatusIs(absl::StatusCode::kResourceExhausted,
               ::testing::HasSubstr(
                   "HTTP response decompression exceeded maximum size")));
}

TEST_P(Http2TransportTest,
       SettingsMaxConcurrentStreamsZeroRoutesToNewConnection) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support stream retry / max streams 0 routing";
  }
  ScriptedH2ServerOptions server_options;
  server_options.max_concurrent_streams = 1;
  ScriptedH2Server server(
      server_options, [](const H2StreamEvent& event, H2Actions& actions) {
        if (event.connection_index == 0) {
          actions.Respond(200, "from-conn-1");
          actions.SendSettings(/*max_concurrent_streams=*/0);
        } else {
          actions.Respond(200, "from-conn-2");
        }
      });

  auto transport = CreateTransport();
  ASSERT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/test"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("from-conn-1")));
  ASSERT_TRUE(server.AwaitSettingsAckCount(2, hang_detect_timeout()));

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/test"),
                                 H2Options(hang_detect_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("from-conn-2")));
  EXPECT_THAT(server.accepted_connections(), Eq(2));
}

TEST_P(Http2TransportTest, EarlyErrorResponseAbortsInFlightUpload) {
  if (!supports_early_final_response()) {
    GTEST_SKIP()
        << "Transport does not support early final response during upload";
  }
  ScriptedH2ServerOptions server_options;
  server_options.act_after_request_bytes = 65535;
  server_options.replenish_window = false;
  ScriptedH2Server server(server_options,
                          [](const H2StreamEvent&, H2Actions& actions) {
                            actions.Respond(413, "too large");
                            actions.GrantWindow(256 * 1024);
                          });

  auto transport = CreateTransport();
  // 256 KiB payload (> 65535-byte default HTTP/2 initial window) so upload
  // remains paused on flow control when the early 413 response arrives.
  absl::Cord large_body(std::string(256 * 1024, 'x'));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("POST", "/upload"),
                                 H2Options(large_body, request_timeout()))
                  .result(),
              HasStatusAndPayload(413, absl::Cord("too large")));
  EXPECT_THAT(server.bytes_after_response(), Eq(0u));
}

TEST_P(Http2TransportTest,
       EnforcesH2HeaderLimitsAnd1xxCountWithoutDroppingConnection) {
  if (!enforces_response_header_and_1xx_limits()) {
    GTEST_SKIP()
        << "Transport does not enforce response header and 1xx count limits";
  }
  ScriptedH2Server server(
      {}, [](const H2StreamEvent& event, H2Actions& actions) {
        if (event.path == "/too_many_headers") {
          H2Response resp;
          resp.status = 200;
          for (int i = 0; i < 260; ++i) {
            resp.headers.emplace_back(absl::StrCat("x-hdr-", i), "v");
          }
          actions.Respond(std::move(resp));
        } else if (event.path == "/too_many_1xx") {
          for (int i = 0; i < 6; ++i) {
            actions.SendInterim(100);
          }
          actions.Respond(200, "unreachable");
        } else {
          actions.Respond(200, "ok");
        }
      });

  auto transport = CreateTransport(SingleConnectionPerHost());
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/too_many_headers"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kUnavailable,
                       ::testing::HasSubstr(
                           "HTTP/2 response headers exceeded maximum limit")));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/too_many_1xx"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kUnavailable,
                       ::testing::HasSubstr(
                           "Exceeded maximum HTTP/2 1xx interim responses")));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/ok"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("ok")));
  EXPECT_THAT(server.accepted_connections(), Eq(1));
}

TEST_P(Http2TransportTest,
       GoawayAndConnectionFailurePreserveDeterministicFifoStreamOrder) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 retry on GOAWAY";
  }
  constexpr int kNumStreams = 6;
  constexpr int kDefaultMaxConcurrentStreams = 4;
  absl::Mutex mu;
  std::vector<std::string> conn2_retry_paths;

  ScriptedH2Server server(
      {}, [&](const H2StreamEvent& event, H2Actions& actions) {
        if (event.connection_index == 0) {
          if (event.stream_index_on_connection ==
              kDefaultMaxConcurrentStreams - 1) {
            actions.RespondOnStream(/*stream_id=*/1, 200, "ok");
            actions.Goaway(/*last_stream_id=*/1, NGHTTP2_NO_ERROR);
          }
        } else {
          {
            absl::MutexLock lock(mu);
            conn2_retry_paths.push_back(event.path);
          }
          actions.Respond(200, "ok");
        }
      });

  auto transport = CreateTransport(SingleConnectionPerHost());
  std::vector<Future<HttpResponse>> futures;
  futures.reserve(kNumStreams);
  for (int i = 0; i < kNumStreams; ++i) {
    futures.push_back(
        transport->IssueRequest(server.Request("GET", absl::StrCat("/req/", i)),
                                H2Options(request_timeout())));
  }
  for (auto& f : futures) {
    EXPECT_THAT(f.result(), IsOk());
  }

  absl::MutexLock lock(mu);
  if (preserves_fifo_stream_retry_order()) {
    EXPECT_THAT(conn2_retry_paths,
                ::testing::ElementsAre("/req/1", "/req/2", "/req/3", "/req/4",
                                       "/req/5"));
  } else {
    EXPECT_THAT(conn2_retry_paths,
                ::testing::UnorderedElementsAre("/req/1", "/req/2", "/req/3",
                                                "/req/4", "/req/5"));
  }
}

TEST_P(Http2TransportTest, H2RefusedStreamAndGoawayTransparentlyRetry) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 stream retry";
  }
  ScriptedH2Server server(
      {}, [](const H2StreamEvent& event, H2Actions& actions) {
        if (event.connection_index == 0) {
          if (event.stream_index_on_connection == 1) {
            actions.RespondOnStream(/*stream_id=*/1, 200, "stream1_ok");
            actions.Goaway(/*last_stream_id=*/1, NGHTTP2_NO_ERROR);
          }
        } else if (event.connection_index == 1 &&
                   event.stream_index_on_connection == 0) {
          actions.RstStream(NGHTTP2_REFUSED_STREAM);
        } else {
          EXPECT_THAT(event.request_body, Eq("post_payload"));
          actions.Respond(200, "stream3_ok");
        }
      });

  auto transport = CreateTransport(SingleConnectionPerHost());
  auto f1 = transport->IssueRequest(server.Request("GET", "/s1"),
                                    H2Options(request_timeout()));
  auto f2 = transport->IssueRequest(
      server.Request("POST", "/s3"),
      H2Options(absl::Cord("post_payload"), request_timeout()));

  EXPECT_THAT(f1.result(), HasStatusAndPayload(200, absl::Cord("stream1_ok")));
  EXPECT_THAT(f2.result(), HasStatusAndPayload(200, absl::Cord("stream3_ok")));
  EXPECT_THAT(server.accepted_connections(), ::testing::Ge(2));
}

TEST_P(Http2TransportTest, H2ZeroLastStreamIdGoawayRespectsRetryLimit) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 retry on GOAWAY";
  }
  ScriptedH2Server server({}, [](const H2StreamEvent&, H2Actions& actions) {
    actions.Goaway(/*last_stream_id=*/0, NGHTTP2_NO_ERROR);
  });

  auto transport = CreateTransport();
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/zero_goaway"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kUnavailable));
  EXPECT_THAT(server.accepted_connections(), ::testing::Ge(2));
}

TEST_P(Http2TransportTest, TransparentRetryExhaustionBoundaryAtFiveRetries) {
  if (!supports_stream_retry()) {
    GTEST_SKIP()
        << "Transport does not support transparent HTTP/2 stream retry";
  }
  std::atomic<int> attempts{0};
  std::atomic<int> refuse_budget{0};

  ScriptedH2Server server({},
                          [&](const H2StreamEvent& event, H2Actions& actions) {
                            if (event.path == "/ping") {
                              actions.Respond(200, "pong");
                              return;
                            }
                            const int attempt = attempts.fetch_add(1) + 1;
                            if (attempt <= refuse_budget.load()) {
                              actions.RstStream(NGHTTP2_REFUSED_STREAM);
                            } else {
                              actions.Respond(200, "retry_boundary_ok");
                            }
                          });

  auto transport = CreateTransport();
  WarmConnection(*transport, server, request_timeout());

  attempts.store(0);
  refuse_budget.store(5);
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/retry_boundary"),
                                 H2Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, absl::Cord("retry_boundary_ok")));
  EXPECT_THAT(attempts.load(), Eq(6));

  attempts.store(0);
  refuse_budget.store(100);
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/retry_boundary"),
                                 H2Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kUnavailable));
}

}  // namespace
}  // namespace internal_http
}  // namespace tensorstore
