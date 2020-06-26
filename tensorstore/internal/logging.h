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

#ifndef TENSORSTORE_INTERNAL_LOGGING_H_
#define TENSORSTORE_INTERNAL_LOGGING_H_

#include <string>

#include "tensorstore/internal/log_message.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/str_cat.h"

/// Logs the concatenation of all arguments converted to strings along with the
/// filename and line number.
#define TENSORSTORE_LOG(...)           \
  ::tensorstore::internal::LogMessage( \
      ::tensorstore::StrCat(__VA_ARGS__).c_str(), TENSORSTORE_LOC)

#define TENSORSTORE_LOG_FATAL(...)          \
  ::tensorstore::internal::LogMessageFatal( \
      ::tensorstore::StrCat(__VA_ARGS__).c_str(), TENSORSTORE_LOC)

/// Same as `TENSORSTORE_LOG` if `conditional`, which must be a constexpr value,
/// is `true`.  Otherwise, does nothing.
#define TENSORSTORE_DEBUG_LOG(conditional, ...) \
  if constexpr (!(conditional)) {               \
  } else                                        \
    TENSORSTORE_LOG(__VA_ARGS__)

#endif  // TENSORSTORE_INTERNAL_LOGGING_H_
