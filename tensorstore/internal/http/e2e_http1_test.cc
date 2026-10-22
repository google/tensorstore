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

#include "tensorstore/internal/http/e2e_http1_test.h"

#ifndef _WIN32
#include <sys/socket.h>
#endif

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
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

using ::tensorstore::IsOkAndHolds;
using ::tensorstore::transport_test_utils::LoopbackServer;
using ::tensorstore::transport_test_utils::SendAll;
using ::tensorstore::transport_test_utils::ShutdownSocket;
using ::tensorstore::transport_test_utils::socket_t;
using ::tensorstore::transport_test_utils::WaitForRead;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;

// A request as seen by an `Http1Policy`.
struct Http1Request {
  // Zero-based index of the connection in accept order.
  int connection_index = 0;
  // Zero-based index of this request among requests on its connection.
  int request_index_on_connection = 0;
  std::string method;
  std::string target;
  // Header fields in arrival order, with lowercased names.
  std::vector<std::pair<std::string, std::string>> headers;
  // The request body. Always empty when the policy acts on headers.
  std::string body;

  // Returns every value of header `name` (lowercase), in arrival order.
  std::vector<std::string_view> HeaderValues(std::string_view name) const {
    std::vector<std::string_view> values;
    for (const auto& [n, v] : headers) {
      if (n == name) values.push_back(v);
    }
    return values;
  }

  // Returns the first value of header `name` (lowercase), if any.
  std::optional<std::string_view> Header(std::string_view name) const {
    for (const auto& [n, v] : headers) {
      if (n == name) return v;
    }
    return std::nullopt;
  }
};

std::string_view ReasonPhrase(int status) {
  switch (status) {
    case 100:
      return "Continue";
    case 200:
      return "OK";
    case 417:
      return "Expectation Failed";
    default:
      return "Unknown";
  }
}

// Actions an `Http1Policy` may take on the connection of a request.
class Http1Actions {
 public:
  explicit Http1Actions(socket_t fd) : fd_(fd) {}

  // Writes `raw` to the connection verbatim.
  void Send(std::string_view raw) {
    if (!closed_ && !SendAll(fd_, raw)) closed_ = true;
  }

  // Sends a complete response framed by Content-Length.
  void Respond(int status, std::string_view body) {
    Send(absl::StrCat("HTTP/1.1 ", status, " ", ReasonPhrase(status),
                      "\r\nContent-Length: ", body.size(), "\r\n\r\n", body));
  }

  // Delays the next action, e.g. so that consecutive sends reach the client
  // in separate reads.
  void Pause(absl::Duration duration) { absl::SleepFor(duration); }

  // Closes the connection once the policy returns, without reading the rest
  // of the request or any later request.
  void Close() { closed_ = true; }

  bool closed() const { return closed_; }

  socket_t fd() const { return fd_; }

 private:
  socket_t fd_;
  bool closed_ = false;
};

// Decides how the server answers each request.
//
// The policy is invoked on the thread serving the request's connection, so
// it may run concurrently for different connections; any state shared
// across connections must be synchronized.
using Http1Policy = std::function<void(const Http1Request&, Http1Actions&)>;

struct Http1ServerOptions {
  // Invoke the policy as soon as the request head arrives, before the body
  // is read, so the server can respond before the upload ends. The server
  // still reads and discards the declared body afterwards.
  bool act_on_headers = false;
};

// Answers 200 with "METHOD target body=BODY".
void Echo(const Http1Request& request, Http1Actions& actions) {
  actions.Respond(200, absl::StrCat(request.method, " ", request.target,
                                    " body=", request.body));
}

// Parses a request head (without the terminating blank line) into
// `request`. Returns the declared Content-Length, or nullopt after
// recording a test failure if the head is malformed or uses a framing this
// server does not support.
std::optional<size_t> ParseRequestHead(std::string_view head,
                                       Http1Request& request) {
  std::vector<std::string_view> lines = absl::StrSplit(head, "\r\n");
  std::vector<std::string_view> request_line = absl::StrSplit(lines[0], ' ');
  if (request_line.size() != 3) {
    ADD_FAILURE() << "Malformed request line: " << lines[0];
    return std::nullopt;
  }
  request.method = std::string(request_line[0]);
  request.target = std::string(request_line[1]);

  size_t content_length = 0;
  for (std::string_view line : absl::MakeConstSpan(lines).subspan(1)) {
    size_t colon = line.find(':');
    if (colon == std::string_view::npos || colon == 0) {
      ADD_FAILURE() << "Malformed header line: " << line;
      return std::nullopt;
    }
    std::string name = absl::AsciiStrToLower(line.substr(0, colon));
    std::string_view value = absl::StripAsciiWhitespace(line.substr(colon + 1));
    if (name == "transfer-encoding") {
      ADD_FAILURE() << "Unsupported request Transfer-Encoding: " << value;
      return std::nullopt;
    }
    if (name == "content-length" && !absl::SimpleAtoi(value, &content_length)) {
      ADD_FAILURE() << "Malformed Content-Length: " << value;
      return std::nullopt;
    }
    request.headers.emplace_back(std::move(name), std::string(value));
  }
  return content_length;
}

// Buffered reader for the requests on one connection.
class Http1RequestReader {
 public:
  Http1RequestReader(socket_t fd, const std::atomic<bool>& stop)
      : fd_(fd), stop_(stop) {}

  // Reads the next request head into `request` and returns its declared
  // Content-Length, or nullopt if the peer disconnected or the head is
  // malformed.
  std::optional<size_t> ReadHead(Http1Request& request) {
    size_t head_end;
    while ((head_end = buf_.find("\r\n\r\n")) == std::string::npos) {
      if (!Fill()) return std::nullopt;
    }
    auto content_length =
        ParseRequestHead(std::string_view(buf_).substr(0, head_end), request);
    buf_.erase(0, head_end + 4);
    return content_length;
  }

  // Reads `length` body bytes into `body`, or discards them if `body` is
  // null. Returns false if the peer disconnected first.
  bool ReadBody(size_t length, std::string* body) {
    while (buf_.size() < length) {
      if (!Fill()) return false;
    }
    if (body != nullptr) body->assign(buf_, 0, length);
    buf_.erase(0, length);
    return true;
  }

 private:
  bool Fill() {
    if (!WaitForRead(fd_, stop_)) return false;
    char tmp[8192];
    auto n = ::recv(fd_, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;
    buf_.append(tmp, static_cast<size_t>(n));
    return true;
  }

  socket_t fd_;
  const std::atomic<bool>& stop_;
  std::string buf_;
};

// Loopback HTTP/1.1 server whose per-request behavior is scripted by an
// `Http1Policy`.
class Http1LoopbackServer {
 public:
  Http1LoopbackServer() : Http1LoopbackServer({}, &Echo) {}

  Http1LoopbackServer(Http1ServerOptions options, Http1Policy policy)
      : options_(options),
        policy_(std::move(policy)),
        server_([this](socket_t fd, int connection_index,
                       const std::atomic<bool>& stop) {
          Serve(fd, connection_index, stop);
        }) {}

  std::string base_url() const {
    return absl::StrCat("http://", server_.hostport());
  }

  HttpRequest Request(std::string_view method, std::string_view path,
                      RequestHeaders headers = {}) const {
    return MakeRequest(base_url(), method, path, headers);
  }

  int accepted_connections() const { return server_.accepted_connections(); }

 private:
  void Serve(socket_t fd, int connection_index,
             const std::atomic<bool>& stop) const {
    Http1RequestReader reader(fd, stop);
    for (int request_index = 0;; ++request_index) {
      Http1Request request;
      request.connection_index = connection_index;
      request.request_index_on_connection = request_index;
      std::optional<size_t> content_length = reader.ReadHead(request);
      if (!content_length) break;

      Http1Actions actions(fd);
      if (options_.act_on_headers) {
        policy_(request, actions);
        if (actions.closed()) break;
        if (!reader.ReadBody(*content_length, nullptr)) break;
      } else {
        if (!reader.ReadBody(*content_length, &request.body)) break;
        policy_(request, actions);
        if (actions.closed()) break;
      }
    }
    ShutdownSocket(fd);
  }

  const Http1ServerOptions options_;
  const Http1Policy policy_;
  // Declared last: its destructor joins the handlers that use the above.
  LoopbackServer server_;
};

IssueRequestOptions Http1Options(absl::Duration timeout) {
  return IssueRequestOptions()
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp1)
      .SetRequestTimeout(timeout);
}

IssueRequestOptions Http1Options(absl::Cord payload, absl::Duration timeout) {
  return IssueRequestOptions(std::move(payload))
      .SetHttpVersion(IssueRequestOptions::HttpVersion::kHttp1)
      .SetRequestTimeout(timeout);
}

// Verifies basic HTTP/1.1 GET request and response body delivery.
TEST_P(Http1TransportTest, GetRequestReturnsExpectedStatusAndPayload) {
  Http1LoopbackServer server;
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/resource?query=val"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "GET /resource?query=val body="));
}

// Verifies persistent keep-alive connection reuse across sequential requests.
TEST_P(Http1TransportTest, SequentialRequestsReusePersistentConnection) {
  Http1LoopbackServer server;
  auto transport = CreateTransport();

  std::vector<Result<HttpResponse>> results;
  for (int i = 0; i < 3; ++i) {
    results.push_back(
        transport
            ->IssueRequest(server.Request("GET", absl::StrCat("/item/", i)),
                           Http1Options(request_timeout()))
            .result());
  }

  EXPECT_THAT(results,
              ElementsAre(HasStatusAndPayload(200, "GET /item/0 body="),
                          HasStatusAndPayload(200, "GET /item/1 body="),
                          HasStatusAndPayload(200, "GET /item/2 body=")));
  EXPECT_THAT(server.accepted_connections(), Eq(1));
}

// Verifies POST request body transmission with Content-Length header.
TEST_P(Http1TransportTest, PostRequestSendsBodyAndContentLength) {
  Http1LoopbackServer server;
  auto transport = CreateTransport();
  std::string payload(2048, 'Z');

  EXPECT_THAT(
      transport
          ->IssueRequest(server.Request("POST", "/upload"),
                         Http1Options(absl::Cord(payload), request_timeout()))
          .result(),
      HasStatusAndPayload(200, absl::StrCat("POST /upload body=", payload)));
}

// Verifies decoding of chunked transfer-encoded response bodies and trailers.
TEST_P(Http1TransportTest, ChunkedResponseDecodedCorrectly) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               actions.Send(
                                   "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/plain\r\n"
                                   "Transfer-Encoding: chunked\r\n"
                                   "\r\n"
                                   "5\r\nhello\r\n"
                                   "6;ext=val\r\n world\r\n"
                                   "0\r\n"
                                   "X-Trailer: trailer-value\r\n"
                                   "\r\n");
                             });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/chunked"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "hello world"));
}

// Verifies HEAD request returns headers without waiting for a response body.
TEST_P(Http1TransportTest, HeadRequestReturnsHeadersWithoutBody) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               // The Content-Length describes the body a GET
                               // would return; a HEAD response has none.
                               actions.Send(
                                   "HTTP/1.1 200 OK\r\n"
                                   "Content-Length: 1024\r\n"
                                   "X-Head-Header: verified\r\n"
                                   "\r\n");
                             });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("HEAD", "/head"),
                                 Http1Options(request_timeout()))
                  .result(),
              AllOf(HasStatusAndPayload(200, ""),
                    HasResponseHeader("x-head-header", "verified")));
}

// Verifies Connection: close header prevents connection reuse for next request.
TEST_P(Http1TransportTest, ConnectionCloseHeaderTerminatesConnection) {
  Http1LoopbackServer server(
      {}, [](const Http1Request& request, Http1Actions& actions) {
        if (request.target != "/close") return Echo(request, actions);
        actions.Send(
            "HTTP/1.1 200 OK\r\n"
            "Content-Length: 7\r\n"
            "Connection: close\r\n"
            "\r\n"
            "closing");
        actions.Close();
      });
  auto transport = CreateTransport();

  ASSERT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/close"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "closing"));
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/new_item"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "GET /new_item body="));
  EXPECT_THAT(server.accepted_connections(), Eq(2));
}

// Verifies 100 Continue interim status is skipped and 200 OK delivered.
TEST_P(Http1TransportTest, Informational100ContinueSkippedBeforeFinalStatus) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               actions.Send("HTTP/1.1 100 Continue\r\n\r\n");
                               actions.Respond(200, "final");
                             });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/100_continue"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "final"));
}

// Verifies handler destroying transport inside callback does not crash.
TEST_P(Http1TransportTest, HandlerDestroysTransportInOnComplete) {
  Http1LoopbackServer server;
  auto transport = CreateTransport();

  auto [promise, future] = PromiseFuturePair<int32_t>::Make();
  TransportDroppingHandler handler(&transport, std::move(promise));
  transport->IssueRequestWithHandler(server.Request("GET", "/quick"),
                                     Http1Options(request_timeout()), &handler);

  EXPECT_THAT(future.result(), IsOkAndHolds(200));
}

// Verifies a custom User-Agent request header replaces the transport's
// default rather than being sent alongside it.
TEST_P(Http1TransportTest, DuplicateUserAgentHeaderHandling) {
  // Reports every User-Agent value received, so a duplicate is visible.
  Http1LoopbackServer server(
      {}, [](const Http1Request& request, Http1Actions& actions) {
        actions.Respond(200,
                        absl::StrJoin(request.HeaderValues("user-agent"), "|"));
      });
  auto transport = CreateTransport();

  EXPECT_THAT(
      transport
          ->IssueRequest(server.Request("GET", "/ua",
                                        {{"user-agent", "custom_agent/2.0"}}),
                         Http1Options(request_timeout()))
          .result(),
      HasStatusAndPayload(200, "custom_agent/2.0"));
}

// Closes the connection without responding to the second request on each
// connection, and answers every other request with `Echo`. This simulates a
// server closing an idle connection just as the client reuses it.
void DropSecondRequestOnConnection(const Http1Request& request,
                                   Http1Actions& actions) {
  if (request.request_index_on_connection == 1) {
    actions.Close();
    return;
  }
  Echo(request, actions);
}

// Verifies an idempotent GET is transparently retried when a reused
// connection drops before any response bytes arrive.
TEST_P(Http1TransportTest, StaleConnectionRetriesIdempotentGet) {
  Http1LoopbackServer server({}, &DropSecondRequestOnConnection);
  auto transport = CreateTransport();
  HttpRequest request = server.Request("GET", "/item");

  ASSERT_THAT(transport->IssueRequest(request, Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "GET /item body="));
  EXPECT_THAT(transport->IssueRequest(request, Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "GET /item body="));
}

// Pins curl's current behavior: a request of any method on a reused
// connection that fails before any response bytes arrive is resent on a new
// connection, so this POST succeeds.
//
// Open policy question: the server may have processed a non-idempotent
// request before closing, so whether such a POST should be retried is not
// decided. This test records the curl contract, not a chosen policy.
TEST_P(Http1TransportTest, StaleConnectionRetriesPostWithoutResponseBytes) {
  Http1LoopbackServer server({}, &DropSecondRequestOnConnection);
  auto transport = CreateTransport();
  HttpRequest request = server.Request("POST", "/upload");

  ASSERT_THAT(transport
                  ->IssueRequest(request, Http1Options(absl::Cord("Hello"),
                                                       request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "POST /upload body=Hello"));
  EXPECT_THAT(transport
                  ->IssueRequest(request, Http1Options(absl::Cord("Hello"),
                                                       request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "POST /upload body=Hello"));
  EXPECT_THAT(server.accepted_connections(), Eq(2));
}

// Rejects Expect: 100-continue requests with 417 before their body arrives,
// and answers every other request with `Echo`. When `split` is set, the 417
// head and body are sent in separate writes so that the client parses the
// final status in its own read.
//
// Like a compliant server, the server still consumes the declared body
// afterwards, which swallows the bytes of any request the client sends on
// the wrongly reused connection.
Http1Policy Reject417(bool split) {
  return [split](const Http1Request& request, Http1Actions& actions) {
    if (request.Header("expect") != "100-continue") {
      return Echo(request, actions);
    }
    constexpr std::string_view kHead =
        "HTTP/1.1 417 Expectation Failed\r\nContent-Length: 4\r\n\r\n";
    if (split) {
      actions.Send(kHead);
      actions.Pause(absl::Milliseconds(50));
      actions.Send("nope");
    } else {
      actions.Send(absl::StrCat(kHead, "nope"));
    }
  };
}

Http1ServerOptions ActOnHeaders() {
  Http1ServerOptions options;
  options.act_on_headers = true;
  return options;
}

// Issues an Expect: 100-continue PUT, which `server` rejects with 417, then
// returns the result of a following GET on the same transport.
Result<HttpResponse> GetAfterRejectedExpectContinue(
    HttpTransport& transport, const Http1LoopbackServer& server,
    absl::Duration timeout) {
  (void)transport
      .IssueRequest(
          server.Request("PUT", "/upload", {{"expect", "100-continue"}}),
          Http1Options(absl::Cord(std::string(1024, 'x')), timeout))
      .result();
  return transport
      .IssueRequest(server.Request("GET", "/next"), Http1Options(timeout))
      .result();
}

// Verifies a connection whose Expect: 100-continue upload was rejected is
// not reused while the server may still expect the declared body.
TEST_P(Http1TransportTest, RejectedExpectContinueDoesNotReuseConnection) {
  Http1LoopbackServer server(ActOnHeaders(), Reject417(/*split=*/false));
  auto transport = CreateTransport();

  EXPECT_THAT(
      GetAfterRejectedExpectContinue(*transport, server, hang_detect_timeout()),
      HasStatusAndPayload(200, "GET /next body="));
}

// Same as above, but the 417 head and body arrive in separate reads.
TEST_P(Http1TransportTest,
       RejectedExpectContinueSplitAcrossReadsDoesNotReuseConnection) {
  Http1LoopbackServer server(ActOnHeaders(), Reject417(/*split=*/true));
  auto transport = CreateTransport();

  EXPECT_THAT(
      GetAfterRejectedExpectContinue(*transport, server, hang_detect_timeout()),
      HasStatusAndPayload(200, "GET /next body="));
}

// Verifies an immediate 200 OK to an Expect: 100-continue upload (without 100
// Continue) does not reuse the connection while the body was withheld.
TEST_P(Http1TransportTest, Immediate200ToExpectContinueDoesNotReuseConnection) {
  if (!supports_early_final_response()) {
    GTEST_SKIP()
        << "Transport does not support early final response during upload";
  }
  Http1LoopbackServer server(
      ActOnHeaders(), [](const Http1Request& request, Http1Actions& actions) {
        if (request.Header("expect") != "100-continue") {
          return Echo(request, actions);
        }
        actions.Respond(200, "early-ok");
      });
  auto transport = CreateTransport();

  EXPECT_THAT(
      GetAfterRejectedExpectContinue(*transport, server, hang_detect_timeout()),
      HasStatusAndPayload(200, "GET /next body="));
}

TEST_P(Http1TransportTest, TestInvalidRequestRejection) {
  auto transport = CreateTransport();

  // 1. CRLF in authority.
  EXPECT_THAT(
      transport
          ->IssueRequest(MakeRequest("http://example.com\r\n/a", "GET", "", {}),
                         Http1Options(request_timeout()))
          .result(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. CRLF in port.
  EXPECT_THAT(transport
                  ->IssueRequest(
                      MakeRequest("http://example.com:80\r\n/a", "GET", "", {}),
                      Http1Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 3. Invalid port.
  EXPECT_THAT(
      transport
          ->IssueRequest(MakeRequest("http://example.com:80a/a", "GET", "", {}),
                         Http1Options(request_timeout()))
          .result(),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // 4. Malformed bracketed IPv6 literal.
  EXPECT_THAT(transport
                  ->IssueRequest(MakeRequest("http://[:::]/a", "GET", "", {}),
                                 Http1Options(request_timeout()))
                  .result(),
              StatusIs(absl::StatusCode::kInvalidArgument));
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

TEST_P(Http1TransportTest, MultiMemberGzipBombRejected) {
  std::string bomb = MakeMultiMemberGzipBomb(/*num_members=*/64);
  Http1LoopbackServer server(
      {}, [&bomb](const Http1Request&, Http1Actions& actions) {
        actions.Send(
            absl::StrCat("HTTP/1.1 200 OK\r\n"
                         "Content-Encoding: gzip\r\n"
                         "Content-Length: ",
                         bomb.size(), "\r\n\r\n", bomb));
      });
  auto transport = CreateTransport();
  HttpRequest request = server.Request("GET", "/gzip_bomb");
  request.accept_encoding = true;

  EXPECT_THAT(transport->IssueRequest(request, Http1Options(request_timeout()))
                  .result(),
              ::testing::Not(IsOk()));
}

TEST_P(Http1TransportTest, UnsupportedContentEncodingRejected) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               actions.Send(
                                   "HTTP/1.1 200 OK\r\n"
                                   "Content-Encoding: br\r\n"
                                   "Content-Length: 10\r\n\r\n"
                                   "0123456789");
                             });
  auto transport = CreateTransport();
  HttpRequest request = server.Request("GET", "/br_encoded");
  request.accept_encoding = true;

  EXPECT_THAT(transport->IssueRequest(request, Http1Options(request_timeout()))
                  .result(),
              ::testing::Not(IsOk()));
}

TEST_P(Http1TransportTest, InvalidChunkTrailerRejected) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               actions.Send(
                                   "HTTP/1.1 200 OK\r\n"
                                   "Transfer-Encoding: chunked\r\n"
                                   "\r\n"
                                   "4\r\ntest\r\n"
                                   "0\r\n"
                                   "Bad Trailer: value\r\n"
                                   "\r\n");
                             });
  auto transport = CreateTransport();

  // A transport may either fail the request when a chunk trailer is malformed
  // on the wire or ignore the malformed trailer without exposing it in
  // `response.headers`.
  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/chunked_bad_trailer"),
                                 Http1Options(request_timeout()))
                  .result(),
              ::testing::AnyOf(::testing::Not(IsOk()),
                               AllOf(HasStatusAndPayload(200, "test"),
                                     LacksResponseHeader("bad trailer"))));
}

TEST_P(Http1TransportTest, UntilCloseBodyTcpRstFailsInsteadOfTruncating) {
  Http1LoopbackServer server({}, [](const Http1Request&,
                                    Http1Actions& actions) {
    actions.Send(
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/plain\r\n"
        "\r\n"
        "partial-body-before-rst");
    struct linger so_linger = {1, 0};
    setsockopt(actions.fd(), SOL_SOCKET, SO_LINGER,
               reinterpret_cast<const char*>(&so_linger), sizeof(so_linger));
    tensorstore::transport_test_utils::CloseSocket(actions.fd());
    actions.Close();
  });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/until-close-rst"),
                                 Http1Options(request_timeout()))
                  .result(),
              ::testing::Not(IsOk()));
}

TEST_P(Http1TransportTest, UntilCloseBodyCleanFinSucceeds) {
  Http1LoopbackServer server({},
                             [](const Http1Request&, Http1Actions& actions) {
                               actions.Send(
                                   "HTTP/1.1 200 OK\r\n"
                                   "Content-Type: text/plain\r\n"
                                   "\r\n"
                                   "complete-until-close-body");
                               actions.Close();
                             });
  auto transport = CreateTransport();

  EXPECT_THAT(transport
                  ->IssueRequest(server.Request("GET", "/until-close-fin"),
                                 Http1Options(request_timeout()))
                  .result(),
              HasStatusAndPayload(200, "complete-until-close-body"));
}

}  // namespace
}  // namespace internal_http
}  // namespace tensorstore
