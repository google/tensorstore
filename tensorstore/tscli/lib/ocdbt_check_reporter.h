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

#ifndef TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_REPORTER_H_
#define TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_REPORTER_H_

#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <optional>
#include <ostream>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace cli {

class OcdbtCheckReporter {
 public:
  // ANSI escape code to move cursor to start of line and erase the entire line.
  constexpr static const char kEraseLine[] = "\r\33[2K";

  OcdbtCheckReporter(std::ostream& output, bool detailed)
      : output_(output), detailed_(detailed), is_tty_(IsTty(output)) {}

  bool detailed() const { return detailed_; }

  int error_count() const {
    absl::MutexLock lock(mutex_);
    return error_count_;
  }
  int warning_count() const {
    absl::MutexLock lock(mutex_);
    return warning_count_;
  }

  // Reports an error from an absl::Status.  The status must not be OK.
  void ReportError(absl::Status status) {
    absl::MutexLock lock(mutex_);
    if (is_tty_) output_ << kEraseLine;
    output_ << status.ToString() << std::endl;
    error_count_++;
  }

  template <typename... Args>
  void ReportWarning(const absl::FormatSpec<Args...>& format,
                     const Args&... args) {
    absl::MutexLock lock(mutex_);
    if (is_tty_) output_ << kEraseLine;
    output_ << absl::StreamFormat(format, args...) << std::endl;
    warning_count_++;
  }

  template <typename... Args>
  void ReportInfo(const absl::FormatSpec<Args...>& format,
                  const Args&... args) {
    absl::MutexLock lock(mutex_);
    if (is_tty_) output_ << kEraseLine;
    output_ << absl::StreamFormat(format, args...) << std::endl;
  }

  template <typename... Args>
  void PrintProgress(const absl::FormatSpec<Args...>& format,
                     const Args&... args) {
    absl::MutexLock lock(mutex_);
    if (is_tty_) output_ << kEraseLine;
    if (total_.has_value() && *total_ > 0) {
      output_ << absl::StrFormat("%06d/%06d ", progress_count_++, *total_);
    } else if (total_.has_value()) {
      output_ << absl::StrFormat("%06d ", progress_count_++);
    }
    output_ << absl::StreamFormat(format, args...) << (is_tty_ ? "\r" : "\n")
            << std::flush;
  }

  void ClearProgress() {
    absl::MutexLock lock(mutex_);
    total_ = std::nullopt;
  }

  void ResetProgress(std::optional<int64_t> total = 0) {
    absl::MutexLock lock(mutex_);
    progress_count_ = 0;
    total_ = total;
  }

 private:
  static bool IsTty(std::ostream& output);

  mutable absl::Mutex mutex_;
  std::ostream& output_;
  bool detailed_;
  bool is_tty_;

  int error_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int warning_count_ ABSL_GUARDED_BY(mutex_) = 0;
  int64_t progress_count_ ABSL_GUARDED_BY(mutex_) = 0;
  std::optional<int64_t> total_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace cli
}  // namespace tensorstore

#endif  // TENSORSTORE_TSCLI_LIB_OCDBT_CHECK_REPORTER_H_
