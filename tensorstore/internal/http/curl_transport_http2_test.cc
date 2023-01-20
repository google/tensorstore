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

#ifdef _WIN32
#undef UNICODE
#define WIN32_LEAN_AND_MEAN
#pragma comment(lib, "ws2_32.lib")
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <optional>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/http/curl_transport.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/transport_test_utils.h"
#include "tensorstore/internal/thread.h"

using ::tensorstore::internal_http::HttpRequestBuilder;
using ::tensorstore::transport_test_utils::AcceptNonBlocking;
using ::tensorstore::transport_test_utils::AssertSend;
using ::tensorstore::transport_test_utils::CloseSocket;
using ::tensorstore::transport_test_utils::CreateBoundSocket;
using ::tensorstore::transport_test_utils::FormatSocketAddress;
using ::tensorstore::transport_test_utils::get_socket_errno;
using ::tensorstore::transport_test_utils::ReceiveAvailable;
using ::tensorstore::transport_test_utils::socket_t;
using ::tensorstore::transport_test_utils::WaitForRead;
using ::testing::HasSubstr;

// Platform specific defines.
#ifdef _WIN32
using ssize_t = ptrdiff_t;
#else  // _WIN32
using ssize_t = ::ssize_t;
#endif  // _WIN32

#include <nghttp2/nghttp2.h>

namespace {

// See nghttp2_frame_type
static constexpr const char* kFrameName[] = {
    "NGHTTP2_DATA",           // 0
    "NGHTTP2_HEADERS",        // 0x01,
    "NGHTTP2_PRIORITY",       // 0x02,
    "NGHTTP2_RST_STREAM",     // 0x03,
    "NGHTTP2_SETTINGS",       // 0x04,
    "NGHTTP2_PUSH_PROMISE",   // 0x05,
    "NGHTTP2_PING",           // 0x06,
    "NGHTTP2_GOAWAY",         // 0x07,
    "NGHTTP2_WINDOW_UPDATE",  // 0x08,
    "NGHTTP2_CONTINUATION",   // 0x09,
    "NGHTTP2_ALTSVC",         // 0x0a,
    "",                       // 0x0b
    "NGHTTP2_ORIGIN",         // 0x0c
};

class Http2Session {
  socket_t client_fd_;
  nghttp2_session* session_;
  int32_t last_stream_id_;

 public:
  struct Stream {
    std::vector<std::pair<std::string, std::string>> headers;
    std::string data;
  };

  absl::flat_hash_map<int32_t, Stream> streams;
  absl::flat_hash_map<int32_t, Stream> completed;

  void OnStreamHeader(int32_t stream_id, std::string key, std::string value) {
    ABSL_LOG(INFO) << "http2 header <" << stream_id << ">: " << key << " "
                   << value;
    streams[stream_id].headers.emplace_back(std::move(key), std::move(value));
  }

  void OnStreamData(int32_t stream_id, const char* data, size_t len) {
    streams[stream_id].data.append(data, len);
  }

  void OnStreamDone(int32_t stream_id) {
    if (streams.count(stream_id) != 0) {
      last_stream_id_ = stream_id;
      completed[stream_id] = std::move(streams[stream_id]);
      streams.erase(stream_id);
    }
  }

  size_t Send(const char* data, size_t length) {
    int err = send(client_fd_, data, length, 0);
    if (err < 0) {
      ABSL_LOG(INFO) << "send error:" << get_socket_errno();
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
    ABSL_CHECK_GT(err, 0);
    return err;
  }

  // Callbacks for nghttp2_session:
  static ssize_t Send(nghttp2_session* session, const uint8_t* data,
                      size_t length, int flags, void* user_data) {
    ABSL_LOG(INFO) << "http2 send " << length << ":"
                   << absl::BytesToHexString(std::string_view(
                          reinterpret_cast<const char*>(data), length));
    return static_cast<Http2Session*>(user_data)->Send(
        reinterpret_cast<const char*>(data), length);
  }

  static int OnFrameSend(nghttp2_session* session, const nghttp2_frame* frame,
                         void* user_data) {
    const auto stream_id = frame->hd.stream_id;
    const auto type = frame->hd.type;
    ABSL_LOG(INFO) << "http2 frame send <" << stream_id
                   << ">: " << kFrameName[type];
    return 0;
  }

  static int OnFrameRecv(nghttp2_session* session, const nghttp2_frame* frame,
                         void* user_data) {
    const auto stream_id = frame->hd.stream_id;
    const auto type = frame->hd.type;
    ABSL_LOG(INFO) << "http2 frame recv <" << stream_id
                   << ">: " << kFrameName[type];

    if ((type == NGHTTP2_DATA || type == NGHTTP2_HEADERS) &&
        (frame->hd.flags & NGHTTP2_FLAG_END_STREAM)) {
      // The request is done.
      ABSL_LOG(INFO) << "http2 stream done <" << stream_id << ">";

      // Update local stream tracking.
      static_cast<Http2Session*>(user_data)->OnStreamDone(stream_id);
    }
    return 0;
  }

  static int OnInvalidFrameRecv(nghttp2_session* session,
                                const nghttp2_frame* frame, int lib_error_code,
                                void* user_data) {
    const auto stream_id = frame->hd.stream_id;
    const auto type = frame->hd.type;
    ABSL_LOG(INFO) << "http2 frame recv invalid <" << stream_id
                   << ">: " << kFrameName[type] << " code=" << lib_error_code;
    return 0;
  }

  static int OnHeader(nghttp2_session* session, const nghttp2_frame* frame,
                      const uint8_t* name, size_t namelen, const uint8_t* value,
                      size_t valuelen, uint8_t flags, void* user_data) {
    static_cast<Http2Session*>(user_data)->OnStreamHeader(
        frame->hd.stream_id,
        std::string(reinterpret_cast<const char*>(name), namelen),
        std::string(reinterpret_cast<const char*>(value), valuelen));
    return 0;
  }

  static int OnDataChunkRecv(nghttp2_session* session, uint8_t flags,
                             int32_t stream_id, const uint8_t* data, size_t len,
                             void* user_data) {
    ABSL_LOG(INFO) << "http2 recv chunk <" << stream_id << ">";
    static_cast<Http2Session*>(user_data)->OnStreamData(
        stream_id, reinterpret_cast<const char*>(data), len);
    return 0;
  }

  static int OnStreamClose(nghttp2_session* session, int32_t stream_id,
                           uint32_t error_code, void* user_data) {
    ABSL_LOG(INFO) << "http2 stream close  <" << stream_id << ">";
    return 0;
  }

  struct StringViewDataSource {
    std::string_view view;
  };

  static ssize_t StringViewRead(nghttp2_session* session, int32_t stream_id,
                                uint8_t* buf, size_t length,
                                uint32_t* data_flags,
                                nghttp2_data_source* source, void*) {
    auto my_data = reinterpret_cast<StringViewDataSource*>(source->ptr);
    if (length > my_data->view.size()) {
      length = my_data->view.size();
      memcpy(buf, my_data->view.data(), length);
      *data_flags |= NGHTTP2_DATA_FLAG_EOF;
      delete my_data;
      return length;
    }
    memcpy(buf, my_data->view.data(), length);
    my_data->view.remove_prefix(length);
    return length;
  }

  Http2Session(socket_t client, std::string_view settings)
      : client_fd_(client) {
    nghttp2_session_callbacks* callbacks;
    ABSL_CHECK_EQ(0, nghttp2_session_callbacks_new(&callbacks));
    nghttp2_session_callbacks_set_send_callback(callbacks, &Http2Session::Send);
    nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                     &Http2Session::OnHeader);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
        callbacks, &Http2Session::OnDataChunkRecv);
    nghttp2_session_callbacks_set_on_stream_close_callback(
        callbacks, &Http2Session::OnStreamClose);
    nghttp2_session_callbacks_set_on_frame_recv_callback(
        callbacks, &Http2Session::OnFrameRecv);
    nghttp2_session_callbacks_set_on_frame_send_callback(
        callbacks, &Http2Session::OnFrameSend);
    nghttp2_session_callbacks_set_on_invalid_frame_recv_callback(
        callbacks, &Http2Session::OnInvalidFrameRecv);

    nghttp2_session_server_new2(&session_, callbacks, this, nullptr);
    nghttp2_session_callbacks_del(callbacks);

    // The initial stream id is 1.
    auto result = nghttp2_session_upgrade2(
        session_, reinterpret_cast<const uint8_t*>(settings.data()),
        settings.size(), false, nullptr);
    ABSL_CHECK_EQ(0, result);

    // Queue a settings
    result = nghttp2_submit_settings(session_, NGHTTP2_FLAG_NONE, nullptr, 0);
    ABSL_CHECK_EQ(0, result);
  }

  ~Http2Session() { nghttp2_session_del(session_); }

  void GoAway() { nghttp2_session_terminate_session(session_, 0); }

  bool Done() {
    return !nghttp2_session_want_write(session_) &&
           !nghttp2_session_want_read(session_);
  }

  bool TrySendReceive() {
    constexpr size_t kBufferSize = 4096;
    char buf[kBufferSize];

    nghttp2_session_send(session_);

    // Read as long as there is data available.
    for (; WaitForRead(client_fd_);) {
      int r = recv(client_fd_, buf, kBufferSize, 0);
      if (r < 0) {
        // WaitForRead calls select() on the socket, so we
        // should not see EAGAIN nor EWOULDBLOCK.
        ABSL_LOG(INFO) << "recv error: " << get_socket_errno();
        return false;
      }
      // No data here would be unexpected since WaitForRead calls select().
      ABSL_CHECK_GT(r, 0);
      ABSL_LOG(INFO) << "socket recv: " << r;
      auto result = nghttp2_session_mem_recv(
          session_, reinterpret_cast<const uint8_t*>(buf), r);
      ABSL_CHECK_GE(result, 0);
    }
    return true;
  }

  void SendResponse(int32_t stream_id,
                    std::vector<std::pair<std::string, std::string>> headers,
                    std::string_view data) {
    ABSL_CHECK_GE(stream_id, 0);
    ABSL_LOG(INFO) << "http2 respond <" << stream_id
                   << ">: " << absl::BytesToHexString(data);

    const size_t num_headers = headers.size();
    std::unique_ptr<nghttp2_nv[]> nvs(new nghttp2_nv[num_headers]);
    for (size_t i = 0; i < num_headers; ++i) {
      auto& header = headers[i];
      nghttp2_nv* nv = &nvs[i];
      nv->name = reinterpret_cast<uint8_t*>(header.first.data());
      nv->value = reinterpret_cast<uint8_t*>(header.second.data());
      nv->namelen = header.first.size();
      nv->valuelen = header.second.size();
      nv->flags = NGHTTP2_NV_FLAG_NONE;
    }
    nghttp2_data_provider data_provider;
    if (!data.empty()) {
      data_provider.source.ptr = new StringViewDataSource{data};
      data_provider.read_callback = &Http2Session::StringViewRead;
    }
    auto result =
        nghttp2_submit_response(session_, stream_id, nvs.get(), num_headers,
                                data.empty() ? nullptr : &data_provider);
    ABSL_CHECK_EQ(0, result);
  }
};

class CurlTransportTest : public ::testing::Test {
 public:
  static void SetUpTestCase() {
#ifdef _WIN32
    WSADATA wsaData;
    ABSL_CHECK_EQ(0, WSAStartup(MAKEWORD(2, 2), &wsaData));
#endif
  }
  static void TearDownTestCase() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
};

TEST_F(CurlTransportTest, Http2) {
  auto transport = ::tensorstore::internal_http::GetDefaultHttpTransport();

  // This test sets up a simple single-request tcp/ip service which allows
  // us to mock a simple HTTP/2 server.
  auto socket = CreateBoundSocket();
  ABSL_CHECK(socket.has_value());

  auto hostport = FormatSocketAddress(*socket);
  ABSL_CHECK(!hostport.empty());

  static constexpr char kSwitchProtocols[] =  // 69
      "HTTP/1.1 101 Switching Protocols\r\n"  // 35
      "Connection: Upgrade\r\n"               //
      "Upgrade: h2c\r\n"                      //
      "\r\n";

  // AAMAAABkAAQCAAAAAAIAAAAA
  static constexpr char kSettings[] = "\0\3\0\0\0\x64\0\4\2\0\0\0\0\2\0\0\0\0";

  std::string initial_request;
  std::string second_request;

  tensorstore::internal::Thread serve_thread({"serve_thread"}, [&] {
    auto client_fd = AcceptNonBlocking(*socket);
    ABSL_CHECK(client_fd.has_value());
    initial_request = ReceiveAvailable(*client_fd);

    // Manually upgrade the h2c to HTTP/2
    AssertSend(*client_fd, kSwitchProtocols);

    Http2Session session(*client_fd, std::string_view(kSettings, 18));
    session.SendResponse(
        1, {{":status", "200"}, {"content-type", "text/html"}},
        "<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n");
    session.TrySendReceive();

    // After that has been sent, we have an additional request to
    // handle,, but it is sent asynchronously on another thread,
    // so loop here. 10 is somewhat arbitrary, though several send/recv
    // calls are required to transmit various frames, since each stream
    // likely needs to send/recv HEADER, DATA, & WINDOW, and then the
    // GOAWAY is triggered below.
    for (int i = 0; i < 10; i++) {
      if (!session.TrySendReceive()) {
        // fd closed prematurely.
        CloseSocket(*client_fd);
        return;
      }

      if (!session.completed.empty()) {
        auto it = session.completed.begin();
        session.SendResponse(
            session.completed.begin()->first,
            {{":status", "200"}, {"content-type", "text/html"}}, "McFly");
        session.completed.erase(it);
      }
    }
    session.TrySendReceive();
    session.GoAway();
    session.TrySendReceive();

    // We have not sent a shutdown message.
    CloseSocket(*client_fd);
  });

  // Issue request 1.
  {
    auto response = transport->IssueRequest(
        HttpRequestBuilder("POST", absl::StrCat("http://", hostport, "/"))
            .AddUserAgentPrefix("test")
            .AddHeader("X-foo: bar")
            .AddQueryParameter("name", "dragon")
            .AddQueryParameter("age", "1234")
            .EnableAcceptEncoding()
            .BuildRequest(),
        absl::Cord("Hello"));

    // Waits for the response.
    ABSL_LOG(INFO) << response.status();

    EXPECT_THAT(initial_request, HasSubstr("/?name=dragon&age=1234"));
    EXPECT_THAT(initial_request,
                HasSubstr(absl::StrCat("Host: ", hostport, "\r\n")));

    // User-Agent versions change based on zlib, nghttp2, and curl versions.
    EXPECT_THAT(initial_request, HasSubstr("User-Agent: testtensorstore/0.1 "));

    EXPECT_THAT(initial_request, HasSubstr("Accept: */*\r\n"));
    EXPECT_THAT(initial_request, HasSubstr("X-foo: bar\r\n"));
    EXPECT_THAT(initial_request, HasSubstr("Content-Length: 5"));
    EXPECT_THAT(
        initial_request,
        HasSubstr("Content-Type: application/x-www-form-urlencoded\r\n"));
    EXPECT_THAT(initial_request, HasSubstr("Hello"));

    EXPECT_EQ(200, response.value().status_code);
    EXPECT_EQ("<html>\n<body>\n<h1>Hello, World!</h1>\n</body>\n</html>\n",
              response.value().payload);
  }

  {
    auto response = transport->IssueRequest(
        HttpRequestBuilder("GET", absl::StrCat("http://", hostport, "/boo"))
            .BuildRequest(),
        absl::Cord());

    // Waits for the response.
    ABSL_LOG(INFO) << response.status();
  }

  serve_thread.Join();
  CloseSocket(*socket);
}

}  // namespace
