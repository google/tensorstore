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

#include "tensorstore/kvstore/s3/aws_api.h"

#include <stdint.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/log_severity.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/absl_log.h"
#include <aws/auth/auth.h>
#include <aws/common/allocator.h>
#include <aws/common/byte_buf.h>
#include <aws/common/common.h>
#include <aws/common/error.h>
#include <aws/common/logging.h>
#include <aws/common/zero.h>
#include <aws/io/channel_bootstrap.h>
#include <aws/io/event_loop.h>
#include <aws/io/host_resolver.h>
#include <aws/io/io.h>
#include <aws/io/tls_channel_handler.h>
#include "tensorstore/internal/log/verbose_flag.h"

namespace tensorstore {
namespace internal_kvstore_s3 {
namespace {

constexpr int kLogBufSize = 2000;

// Hook AWS logging into absl logging.
ABSL_CONST_INIT internal_log::VerboseFlag aws_logging("aws");

int absl_log(struct aws_logger *logger, enum aws_log_level log_level,
             aws_log_subject_t subject, const char *format, ...) {
  absl::LogSeverity severity = absl::LogSeverity::kInfo;
  if (log_level <= AWS_LL_FATAL) {
    severity = absl::LogSeverity::kFatal;
  } else if (log_level <= AWS_LL_ERROR) {
    severity = absl::LogSeverity::kError;
  } else if (log_level <= AWS_LL_WARN) {
    severity = absl::LogSeverity::kWarning;
  }
#ifdef ABSL_MIN_LOG_LEVEL
  if (severity < static_cast<absl::LogSeverity>(ABSL_MIN_LOG_LEVEL) &&
      severity < absl::LogSeverity::kFatal) {
    enabled = false;
  }
#endif

  // AWS Logging doesn't provide a way to get the filename or line number,
  // instead use the aws subject name as the filename and the subject itself as
  // the line number.
  const char *subject_name = aws_log_subject_name(subject);
  bool is_valid_subject =
      (subject_name != nullptr && strcmp(subject_name, "Unknown") != 0);

  char buffer[kLogBufSize];
  char *buf = buffer;
  size_t size = sizeof(buffer);

  va_list argptr;
  va_start(argptr, format);
  int n = vsnprintf(buf, size, format, argptr);
  va_end(argptr);
  if (n > 0 && n < size) {
    ABSL_LOG(LEVEL(severity))
            .AtLocation(is_valid_subject ? subject_name : "aws_api.cc", subject)
        << std::string_view(buf, n);
  }
  return AWS_OP_SUCCESS;
};

enum aws_log_level absl_get_log_level(struct aws_logger *logger,
                                      aws_log_subject_t subject) {
  uintptr_t lvl = reinterpret_cast<uintptr_t>(logger->p_impl);
  if (lvl != 0) {
    return static_cast<enum aws_log_level>(lvl - 1);
  }
  if (!aws_logging) {
    return AWS_LL_WARN;
  }
  // NOTE: AWS logging is quite verbose even at AWS_LL_INFO.
  if (aws_logging.Level(1)) {
    return aws_logging.Level(2) ? AWS_LL_TRACE : AWS_LL_DEBUG;
  }
  return AWS_LL_INFO;
}

int absl_set_log_level(struct aws_logger *logger, enum aws_log_level lvl) {
  if (lvl == AWS_LL_NONE) {
    reinterpret_cast<uintptr_t &>(logger->p_impl) = 0;
  } else {
    reinterpret_cast<uintptr_t &>(logger->p_impl) =
        1 + static_cast<uintptr_t>(lvl);
  }
  return AWS_OP_SUCCESS;
}

void absl_clean_up(struct aws_logger *logger) { (void)logger; }

static aws_logger_vtable s_absl_vtable = {
    /*log =*/absl_log,
    /*get_log_level =*/absl_get_log_level,
    /*clean_up =*/absl_clean_up,
    /*set_log_level =*/absl_set_log_level,
};

static aws_logger s_absl_logger = {
    /*vtable =*/&s_absl_vtable,
    /*.allocator =*/nullptr,
    /*.p_impl =*/0,
};

// AWS apis rely on global initialization; do that here.
absl::once_flag g_init_global;
aws_allocator *g_allocator = nullptr;

void InitAwsGlobal() {
  absl::LeakCheckDisabler disabler;
  aws_logger_set(&s_absl_logger);
  g_allocator = aws_default_allocator();

  // Initialize the AWS APIs used.
  aws_common_library_init(g_allocator);
  aws_io_library_init(g_allocator);
  aws_auth_library_init(g_allocator);
}

// Also initialize a global client bootstrap and TLS context.
absl::once_flag g_init_tls;
aws_event_loop_group *g_event_loop_group = nullptr;
aws_host_resolver *g_resolver = nullptr;
aws_client_bootstrap *g_client_bootstrap = nullptr;
aws_tls_ctx *g_tls_ctx = nullptr;

void InitAwsTls() {
  absl::LeakCheckDisabler disabler;
  /* event loop */
  g_event_loop_group =
      aws_event_loop_group_new_default(g_allocator, 0, nullptr);

  /* resolver */
  aws_host_resolver_default_options resolver_options;
  AWS_ZERO_STRUCT(resolver_options);
  resolver_options.el_group = g_event_loop_group;
  resolver_options.max_entries = 32;  // defaults to 8?
  g_resolver = aws_host_resolver_new_default(g_allocator, &resolver_options);

  /* client bootstrap */
  aws_client_bootstrap_options bootstrap_options;
  AWS_ZERO_STRUCT(bootstrap_options);
  bootstrap_options.event_loop_group = g_event_loop_group;
  bootstrap_options.host_resolver = g_resolver;
  g_client_bootstrap =
      aws_client_bootstrap_new(g_allocator, &bootstrap_options);
  if (g_client_bootstrap == nullptr) {
    ABSL_LOG(FATAL) << "ERROR initializing client bootstrap: "
                    << aws_error_debug_str(aws_last_error());
  }

  aws_tls_ctx_options tls_options;
  aws_tls_ctx_options_init_default_client(&tls_options, g_allocator);
  aws_tls_ctx *tls_ctx = aws_tls_client_ctx_new(g_allocator, &tls_options);
  aws_tls_ctx_options_clean_up(&tls_options);
  if (tls_ctx == nullptr) {
    ABSL_LOG(FATAL) << "ERROR initializing TLS context: "
                    << aws_error_debug_str(aws_last_error());
  }
  g_tls_ctx = tls_ctx;
}

}  // namespace

aws_allocator *GetAwsAllocator() {
  absl::call_once(g_init_global, InitAwsGlobal);
  return g_allocator;
}

aws_client_bootstrap *GetAwsClientBootstrap() {
  absl::call_once(g_init_global, InitAwsGlobal);
  absl::call_once(g_init_tls, InitAwsTls);
  return g_client_bootstrap;
}

aws_tls_ctx *GetAwsTlsCtx() {
  absl::call_once(g_init_global, InitAwsGlobal);
  absl::call_once(g_init_tls, InitAwsTls);
  return g_tls_ctx;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
