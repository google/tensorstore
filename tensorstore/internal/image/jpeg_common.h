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

#ifndef TENSORSTORE_INTERNAL_IMAGE_JPEG_COMMON_H_
#define TENSORSTORE_INTERNAL_IMAGE_JPEG_COMMON_H_

#include <csetjmp>
#include <string>

#include "absl/status/status.h"

// Include libjpeg last
#include <jerror.h>
#include <jpeglib.h>

namespace tensorstore {
namespace internal_image {

/// jpeg_error_mgr implementation.
// Because JpegSourceRiegeli is standard layout, we can safely
// `reinterpret_cast` from pointer to first member.
struct JpegError {
  ::jpeg_error_mgr jerr_;
  std::jmp_buf jmpbuf;
  absl::Status last_error;

  // ::jpeg_error_mgr handlers.
  [[noreturn]] static void ErrorExit(::jpeg_common_struct* cinfo);
  static void EmitMessage(::jpeg_common_struct* cinfo, int msg_level);

  // "Constructor"
  void Construct(::jpeg_common_struct* cinfo);
};

}  // namespace internal_image
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_IMAGE_JPEG_COMMON_H_
