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

#include "tensorstore/internal/aws/aws_api.h"

#include <stddef.h>
#include <stdint.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "absl/base/attributes.h"
#include "absl/base/log_severity.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/debugging/leak_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include <aws/auth/auth.h>
#include <aws/cal/cal.h>
#include <aws/common/allocator.h>
#include <aws/common/common.h>
#include <aws/common/error.h>
#include <aws/common/logging.h>
#include <aws/common/zero.h>
#include <aws/http/http.h>
#include <aws/io/channel_bootstrap.h>
#include <aws/io/event_loop.h>
#include <aws/io/host_resolver.h>
#include <aws/io/io.h>
#include <aws/io/tls_channel_handler.h>
#include "tensorstore/internal/aws/tls_ctx.h"
#include "tensorstore/internal/log/verbose_flag.h"

namespace tensorstore {
namespace internal_aws {
namespace {

constexpr int kLogBufSize = 2000;

// Hook AWS logging into absl logging.
ABSL_CONST_INIT internal_log::VerboseFlag aws_logging("aws");

int absl_log(aws_logger *logger, aws_log_level log_level,
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

enum aws_log_level absl_get_log_level(aws_logger *logger,
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

int absl_set_log_level(aws_logger *logger, aws_log_level lvl) {
  if (lvl == AWS_LL_NONE) {
    reinterpret_cast<uintptr_t &>(logger->p_impl) = 0;
  } else {
    reinterpret_cast<uintptr_t &>(logger->p_impl) =
        1 + static_cast<uintptr_t>(lvl);
  }
  return AWS_OP_SUCCESS;
}

void absl_clean_up(aws_logger *logger) { (void)logger; }

// Some C++ compiler targets don't like designated initializers in C++, until
// they are supported static_assert() on the offsets
static_assert(offsetof(aws_logger_vtable, log) == 0);
static_assert(offsetof(aws_logger_vtable, get_log_level) == sizeof(void *));
static_assert(offsetof(aws_logger_vtable, clean_up) == 2 * sizeof(void *));
static_assert(offsetof(aws_logger_vtable, set_log_level) == 3 * sizeof(void *));

static_assert(offsetof(aws_logger, vtable) == 0);
static_assert(offsetof(aws_logger, allocator) == sizeof(void *));
static_assert(offsetof(aws_logger, p_impl) == 2 * sizeof(void *));

ABSL_CONST_INIT aws_logger_vtable s_absl_vtable{
    /*.log=*/absl_log,
    /*.get_log_level=*/absl_get_log_level,
    /*.clean_up=*/absl_clean_up,
    /*.set_log_level=*/absl_set_log_level,
};

aws_logger s_absl_logger{
    /*.vtable=*/&s_absl_vtable,
    /*.allocator=*/nullptr,
    /*.p_impl=*/nullptr,
};

// AWS apis rely on global initialization; do that here.
class AwsApi {
 public:
  AwsApi() : allocator_(aws_default_allocator()) {
    absl::LeakCheckDisabler disabler;

    /* Initialize AWS libraries.*/
    aws_common_library_init(allocator_);

    s_absl_logger.allocator = allocator_;
    aws_logger_set(&s_absl_logger);

    aws_cal_library_init(allocator_);
    aws_io_library_init(allocator_);
    aws_http_library_init(allocator_);
    aws_auth_library_init(allocator_);
  }

  aws_allocator *allocator() { return allocator_; }

  aws_client_bootstrap *client_bootstrap() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock l(&mutex_);
    init_client_bootstrap();
    return client_bootstrap_;
  }

  aws_tls_ctx *tls_ctx() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock l(&mutex_);
    init_tls_ctx();
    return tls_ctx_;
  }

 private:
  void init_event_loop_group() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (event_loop_group_ != nullptr) return;
    event_loop_group_ =
        aws_event_loop_group_new_default(allocator_, 0, nullptr);
  }

  void init_resolver() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (resolver_ != nullptr) return;
    init_event_loop_group();

    aws_host_resolver_default_options resolver_options;
    AWS_ZERO_STRUCT(resolver_options);
    resolver_options.el_group = event_loop_group_;
    resolver_options.max_entries = 32;  // defaults to 8?
    resolver_ = aws_host_resolver_new_default(allocator_, &resolver_options);
  }

  void init_client_bootstrap() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (client_bootstrap_ != nullptr) return;
    init_event_loop_group();
    init_resolver();

    aws_client_bootstrap_options bootstrap_options;
    AWS_ZERO_STRUCT(bootstrap_options);
    bootstrap_options.event_loop_group = event_loop_group_;
    bootstrap_options.host_resolver = resolver_;
    client_bootstrap_ =
        aws_client_bootstrap_new(allocator_, &bootstrap_options);
    if (client_bootstrap_ == nullptr) {
      ABSL_LOG(FATAL) << "ERROR initializing client bootstrap: "
                      << aws_error_debug_str(aws_last_error());
    }
  }

  void init_tls_ctx() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    if (tls_ctx_ != nullptr) return;
    auto my_tls_ctx = AwsTlsCtxBuilder(allocator_).Build();
    if (my_tls_ctx == nullptr) {
      ABSL_LOG(FATAL) << "ERROR initializing TLS context: "
                      << aws_error_debug_str(aws_last_error());
    }
    tls_ctx_ = my_tls_ctx.release();
  }

  absl::Mutex mutex_;
  aws_allocator *allocator_ = nullptr;
  aws_event_loop_group *event_loop_group_ ABSL_GUARDED_BY(mutex_) = nullptr;
  aws_host_resolver *resolver_ ABSL_GUARDED_BY(mutex_) = nullptr;
  aws_client_bootstrap *client_bootstrap_ ABSL_GUARDED_BY(mutex_) = nullptr;
  aws_tls_ctx *tls_ctx_ ABSL_GUARDED_BY(mutex_) = nullptr;
};

AwsApi &GetAwsApi() {
  static absl::NoDestructor<AwsApi> aws_api;
  return *aws_api;
}

}  // namespace

aws_allocator *GetAwsAllocator() { return GetAwsApi().allocator(); }

aws_client_bootstrap *GetAwsClientBootstrap() {
  return GetAwsApi().client_bootstrap();
}

aws_tls_ctx *GetAwsTlsCtx() { return GetAwsApi().tls_ctx(); }

}  // namespace internal_aws
}  // namespace tensorstore
