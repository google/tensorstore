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

#include "tensorstore/internal/image/jpeg_common.h"

#include <cassert>

#include "absl/status/status.h"

namespace tensorstore {
namespace internal_image {

void JpegError::ErrorExit(::jpeg_common_struct* cinfo) {
  char buffer[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message)(cinfo, buffer);

  auto& self = *reinterpret_cast<JpegError*>(cinfo->err);
  self.last_error = absl::InternalError(buffer);
  std::longjmp(self.jmpbuf, 1);
}

void JpegError::EmitMessage(::jpeg_common_struct* cinfo, int msg_level) {
  if (msg_level > 0) {
    // Ignore trace messages.
    return;
  }
  cinfo->err->num_warnings++;
  // Warnings (which can indicate corrupt data) are treated as errors.
  JpegError::ErrorExit(cinfo);
}

void JpegError::Construct(::jpeg_common_struct* cinfo) {
  assert(cinfo);
  // Only `emit_message` and `error_exit` are called by the rest of the
  // library.
  ::jpeg_std_error(&jerr_);
  jerr_.emit_message = &JpegError::EmitMessage;
  jerr_.error_exit = &JpegError::ErrorExit;
  cinfo->err = &jerr_;
}

}  // namespace internal_image
}  // namespace tensorstore
