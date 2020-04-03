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

#ifndef TENSORSTORE_INTERNAL_LOG_MESSAGE_H_
#define TENSORSTORE_INTERNAL_LOG_MESSAGE_H_

/// \file Internal low-level logging facilities.

#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal {
/// Logs a message.
void LogMessage(const char* message,
                SourceLocation loc TENSORSTORE_LOC_CURRENT_DEFAULT_ARG);

/// Logs a message and terminates the program.
[[noreturn]] void LogMessageFatal(const char* message,
                                  SourceLocation loc
                                      TENSORSTORE_LOC_CURRENT_DEFAULT_ARG);
}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_LOG_MESSAGE_H_
