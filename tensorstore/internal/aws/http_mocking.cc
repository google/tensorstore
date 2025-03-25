// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/internal/aws/http_mocking.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include <aws/auth/credentials.h>
#include <aws/auth/private/credentials_utils.h>
#include <aws/common/allocator.h>
#include <aws/common/byte_buf.h>
#include <aws/common/clock.h>
#include <aws/common/error.h>
#include <aws/common/zero.h>
#include <aws/http/connection_manager.h>
#include <aws/http/request_response.h>
#include "tensorstore/internal/aws/string_view.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/log/verbose_flag.h"

namespace tensorstore {
namespace internal_aws {
namespace {

// Hook AWS logging into absl logging.
ABSL_CONST_INIT internal_log::VerboseFlag aws_logging("aws");

std::atomic<bool> g_use_mock{false};

struct ConnectionManager {
  std::string connection_string;
  void *shutdown_complete_user_data;
  aws_http_connection_manager_shutdown_complete_fn *shutdown_complete_callback;
  std::atomic<int> refcount{1};
};

struct Mock {
  absl::Mutex mutex;
  std::vector<std::pair<std::string, internal_http::HttpResponse>> responses;
};

Mock &GetMock() {
  static Mock *mock = absl::IgnoreLeak(new Mock());
  return *mock;
}

aws_http_connection_manager *s_aws_http_connection_manager_new_mock(
    struct aws_allocator *allocator,
    const struct aws_http_connection_manager_options *options) {
  (void)allocator;
  (void)options;

  // This constructs the http://host:port string for the connection,
  // storing it in the mock for continued reuse.
  std::string connection_string = absl::StrCat(
      options->tls_connection_options ? "https" : "http", "://",
      AwsByteCursorToStringView(options->host), ":", options->port, "/");
  ABSL_LOG_IF(INFO, aws_logging)
      << "s_aws_http_connection_manager_new_mock " << connection_string;

  ConnectionManager *cm = new ConnectionManager();
  cm->connection_string = std::move(connection_string);
  cm->shutdown_complete_user_data = options->shutdown_complete_user_data;
  cm->shutdown_complete_callback = options->shutdown_complete_callback;

  return (struct aws_http_connection_manager *)cm;
}

void s_aws_http_connection_manager_release_mock(
    struct aws_http_connection_manager *manager) {
  auto *cm = (ConnectionManager *)manager;
  if (cm->refcount.fetch_sub(1) == 1) {
    cm->shutdown_complete_callback(cm->shutdown_complete_user_data);
    delete cm;
  }
}

void s_aws_http_connection_manager_acquire_connection_mock(
    struct aws_http_connection_manager *manager,
    aws_http_connection_manager_on_connection_setup_fn *callback,
    void *user_data) {
  (void)manager;
  (void)callback;
  (void)user_data;
  auto *cm = (ConnectionManager *)manager;
  cm->refcount.fetch_add(1);
  ABSL_LOG_IF(INFO, aws_logging.Level(1))
      << "s_aws_http_connection_manager_acquire_connection_mock "
      << cm->connection_string;
  callback((struct aws_http_connection *)cm->connection_string.c_str(),
           AWS_OP_SUCCESS, user_data);
}

int s_aws_http_connection_manager_release_connection_mock(
    struct aws_http_connection_manager *manager,
    struct aws_http_connection *connection) {
  (void)manager;
  (void)connection;
  auto *cm = (ConnectionManager *)manager;
  if (cm->refcount.fetch_sub(1) == 1) {
    cm->shutdown_complete_callback(cm->shutdown_complete_user_data);
    delete cm;
  }
  return AWS_OP_SUCCESS;
}

int s_aws_http_stream_activate_mock(struct aws_http_stream *stream) {
  (void)stream;
  return AWS_OP_SUCCESS;
}

static struct aws_http_connection *s_aws_http_stream_get_connection_mock(
    const struct aws_http_stream *stream) {
  (void)stream;
  return (struct aws_http_connection *)1;
}

int s_aws_http_stream_get_incoming_response_status_mock(
    const struct aws_http_stream *stream, int *out_status_code) {
  (void)stream;

  *out_status_code = (int)(uintptr_t)stream;
  return AWS_OP_SUCCESS;
}

void s_aws_http_stream_release_mock(struct aws_http_stream *stream) {
  (void)stream;
}

void s_aws_http_connection_close_mock(struct aws_http_connection *connection) {
  (void)connection;
}

internal_http::HttpRequest BuildHttpRequest(
    struct aws_http_connection *client_connection,
    const struct aws_http_make_request_options *options) {
  internal_http::HttpRequest request;

  // method
  struct aws_byte_cursor method_cursor;
  AWS_ZERO_STRUCT(method_cursor);
  aws_http_message_get_request_method(options->request, &method_cursor);
  request.method = AwsByteCursorToStringView(method_cursor);

  // uri
  struct aws_byte_cursor path_cursor;
  AWS_ZERO_STRUCT(path_cursor);
  aws_http_message_get_request_path(options->request, &path_cursor);
  auto path = AwsByteCursorToStringView(path_cursor);
  if (!path.empty() && path.front() == '/') path = path.substr(1);
  request.url = absl::StrCat(reinterpret_cast<char *>(client_connection), path);

  // user_agent

  // Add all headers.
  aws_http_headers *headers = aws_http_message_get_headers(options->request);
  for (size_t i = 0, end = aws_http_headers_count(headers); i < end; ++i) {
    aws_http_header header;
    AWS_ZERO_STRUCT(header);
    aws_http_headers_get_index(headers, i, &header);
    request.headers.SetHeader(AwsByteCursorToStringView(header.name),
                              AwsByteCursorToStringView(header.value));
  }
  return request;
}

std::string ComposeRequestPathAndQuery(
    struct aws_http_connection *client_connection,
    const struct aws_http_make_request_options *options) {
  struct aws_byte_cursor method_cursor;
  AWS_ZERO_STRUCT(method_cursor);
  aws_http_message_get_request_method(options->request, &method_cursor);

  struct aws_byte_cursor path_cursor;
  AWS_ZERO_STRUCT(path_cursor);
  aws_http_message_get_request_path(options->request, &path_cursor);
  auto path = AwsByteCursorToStringView(path_cursor);
  if (!path.empty() && path.front() == '/') path = path.substr(1);
  return absl::StrCat(AwsByteCursorToStringView(method_cursor), " ",
                      reinterpret_cast<char *>(client_connection), path);
}

aws_http_stream *s_aws_http_connection_make_request_mock(
    struct aws_http_connection *client_connection,
    const struct aws_http_make_request_options *options) {
  (void)client_connection;
  (void)options;

  // This is the workhorse for the mocking path.
  std::string request_path_and_query =
      ComposeRequestPathAndQuery(client_connection, options);
  ABSL_LOG_IF(INFO, aws_logging)
      << "s_aws_http_connection_make_request_mock " << request_path_and_query;

  std::optional<internal_http::HttpResponse> response;
  {
    auto &mock = GetMock();
    absl::MutexLock lock(&mock.mutex);
    for (size_t i = 0; i < mock.responses.size(); ++i) {
      if (mock.responses[i].first == request_path_and_query) {
        response = mock.responses[i].second;
        mock.responses[i].first = "";
        break;
      }
    }
  }
  if (!response) {
    ABSL_LOG(INFO) << "No response for "
                   << BuildHttpRequest(client_connection, options);
    response = internal_http::HttpResponse{
        404, absl::Cord(request_path_and_query), {}};
  }

  auto stream = (struct aws_http_stream *)(uintptr_t)response->status_code;

  // Respond with headers.
  for (const auto &kv : response->headers) {
    struct aws_http_header headers[1];
    AWS_ZERO_ARRAY(headers);
    headers[0].name = StringViewToAwsByteCursor(kv.first);
    headers[0].value = StringViewToAwsByteCursor(kv.second);
    options->on_response_headers(stream, AWS_HTTP_HEADER_BLOCK_MAIN, headers, 1,
                                 options->user_data);
  }
  if (options->on_response_header_block_done) {
    options->on_response_header_block_done(stream, AWS_HTTP_HEADER_BLOCK_MAIN,
                                           options->user_data);
  }

  // Respond with body.
  for (auto chunk : response->payload.Chunks()) {
    auto cursor = StringViewToAwsByteCursor(chunk);
    options->on_response_body(stream, &cursor, options->user_data);
  }
  options->on_complete(stream, AWS_ERROR_SUCCESS, options->user_data);

  return stream;
}

}  // namespace

aws_auth_http_system_vtable *GetAwsHttpMockingIfEnabled() {
  static aws_auth_http_system_vtable s_mock_function_table = []() {
    aws_auth_http_system_vtable s;
    s.aws_http_connection_manager_new = s_aws_http_connection_manager_new_mock;
    s.aws_http_connection_manager_release =
        s_aws_http_connection_manager_release_mock;
    s.aws_http_connection_manager_acquire_connection =
        s_aws_http_connection_manager_acquire_connection_mock;
    s.aws_http_connection_manager_release_connection =
        s_aws_http_connection_manager_release_connection_mock;
    s.aws_http_connection_make_request =
        s_aws_http_connection_make_request_mock;
    s.aws_http_stream_activate = s_aws_http_stream_activate_mock;
    s.aws_http_stream_get_connection = s_aws_http_stream_get_connection_mock;
    s.aws_http_stream_get_incoming_response_status =
        s_aws_http_stream_get_incoming_response_status_mock;
    s.aws_http_stream_release = s_aws_http_stream_release_mock;
    s.aws_http_connection_close = s_aws_http_connection_close_mock;
    s.aws_high_res_clock_get_ticks = aws_high_res_clock_get_ticks;
    return s;
  }();

  if (g_use_mock.load()) {
    GetMock();
    return &s_mock_function_table;
  }
  return nullptr;
}

void EnableAwsHttpMocking(
    std::vector<std::pair<std::string, internal_http::HttpResponse>>
        responses) {
  auto &mock = GetMock();
  absl::MutexLock lock(&mock.mutex);
  mock.responses = std::move(responses);
  g_use_mock.store(true);
}

void DisableAwsHttpMocking() { g_use_mock.store(false); }

}  // namespace internal_aws
}  // namespace tensorstore
