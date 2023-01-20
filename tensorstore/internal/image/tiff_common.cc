// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/image/tiff_common.h"

#include <stdarg.h>
#include <stdio.h>

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"

// Include libtiff last.
// See: http://www.libtiff.org/man/index.html
#include <tiffio.h>

namespace tensorstore {
namespace internal_image {
namespace {

/// libtiff has static globals for the warning handler; best to lock and
/// only allow a single thread.
struct LibTIFFErrorHook {
  absl::Mutex mutex_;
  TIFFErrorHandlerExt error_handler_ = nullptr;
  TIFFErrorHandlerExt warning_handler_ = nullptr;

  absl::flat_hash_set<thandle_t> live_;

  void MaybeInstall(LibTiffErrorBase* handle);
  void MaybeUninstall(LibTiffErrorBase* handle);
};

LibTIFFErrorHook* GetLibTIFFErrorHook() {
  static auto* hook = new LibTIFFErrorHook();
  return hook;
}

void TensorstoreTiffWarningHandler(thandle_t data, const char* file,
                                   const char* format, va_list ap) {
  char buf[128];
  vsnprintf(buf, sizeof(buf), format, ap);
  buf[sizeof(buf) - 1] = 0;
  ABSL_LOG(WARNING) << "libtiff warn " << file << ": " << buf;
}

void TensorstoreTiffErrorHandler(thandle_t data, const char* file,
                                 const char* format, va_list ap) {
  char buf[128];
  vsnprintf(buf, sizeof(buf), format, ap);
  buf[sizeof(buf) - 1] = 0;

  ABSL_LOG(ERROR) << "libtiff error " << file << ": " << buf;

  if (data) {
    auto* hook = GetLibTIFFErrorHook();
    absl::MutexLock l(&hook->mutex_);
    if (auto it = hook->live_.find(data); it != hook->live_.end()) {
      reinterpret_cast<LibTiffErrorBase*>(data)->error_.Update(
          absl::InvalidArgumentError(std::string(buf)));
    }
  }
}

void LibTIFFErrorHook::MaybeInstall(LibTiffErrorBase* handle) {
  absl::MutexLock l(&mutex_);
  if (live_.empty()) {
    warning_handler_ = TIFFSetWarningHandlerExt(TensorstoreTiffWarningHandler);
    error_handler_ = TIFFSetErrorHandlerExt(TensorstoreTiffErrorHandler);
  }
  live_.insert(handle);
}

void LibTIFFErrorHook::MaybeUninstall(LibTiffErrorBase* handle) {
  absl::MutexLock l(&mutex_);
  live_.erase(handle);
  if (live_.empty()) {
    TIFFSetWarningHandlerExt(warning_handler_);
    TIFFSetErrorHandlerExt(error_handler_);
    warning_handler_ = nullptr;
    error_handler_ = nullptr;
  }
}

}  // namespace

LibTiffErrorBase::LibTiffErrorBase() {
  GetLibTIFFErrorHook()->MaybeInstall(this);
}
LibTiffErrorBase::~LibTiffErrorBase() {
  GetLibTIFFErrorHook()->MaybeUninstall(this);
}

}  // namespace internal_image
}  // namespace tensorstore
