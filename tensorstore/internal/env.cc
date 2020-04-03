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

#include "tensorstore/internal/env.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace tensorstore {
namespace internal {

std::optional<std::string> GetEnv(char const* variable) {
#if _WIN32
  // On Windows, std::getenv() is not thread-safe. It returns a pointer that
  // can be invalidated by _putenv_s(). We must use the thread-safe alternative,
  // which unfortunately allocates the buffer using malloc():
  char* buffer;
  std::size_t size;
  _dupenv_s(&buffer, &size, variable);
  std::unique_ptr<char, decltype(&free)> release(buffer, &free);
#else
  char* buffer = std::getenv(variable);
#endif  // _WIN32
  if (buffer == nullptr) {
    return std::optional<std::string>();
  }
  return std::optional<std::string>(std::string{buffer});
}

void SetEnv(const char* variable, const char* value) {
#if _WIN32
  ::_putenv_s(variable, value);
#else
  ::setenv(variable, value, 1);
#endif
}

void UnsetEnv(const char* variable) {
#if _WIN32
  ::_putenv_s(variable, "");
#else
  ::unsetenv(variable);
#endif
}

}  // namespace internal
}  // namespace tensorstore
