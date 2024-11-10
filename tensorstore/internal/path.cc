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

#include "tensorstore/internal/path.h"

#include <stddef.h>

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace {

#ifdef _WIN32
constexpr inline bool IsDirSeparator(char c) { return c == '\\' || c == '/'; }
#else
constexpr inline bool IsDirSeparator(char c) { return c == '/'; }
#endif

}  // namespace
namespace tensorstore {
namespace internal_path {

std::string JoinPathImpl(std::initializer_list<std::string_view> paths) {
  size_t s = 0;
  for (std::string_view path : paths) {
    s += path.size() + 1;
  }

  std::string result;
  result.reserve(s);
  for (std::string_view path : paths) {
    internal::AppendPathComponent(result, path);
  }
  return result;
}

}  // namespace internal_path
namespace internal {

// Splits a path into the pair {dirname, basename}
std::pair<std::string_view, std::string_view> PathDirnameBasename(
    std::string_view path) {
  size_t pos = path.size();
  while (pos != 0 && !IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  size_t basename = pos;
  --pos;
  if (pos == std::string_view::npos) {
    return {"", path};
  }
  while (pos != 0 && IsDirSeparator(path[pos - 1])) {
    --pos;
  }
  if (pos == 0) {
    return {"/", path.substr(basename)};
  }
  return {path.substr(0, pos), path.substr(basename)};
}

void EnsureDirectoryPath(std::string& path) {
  if (path.size() == 1 && path[0] == '/') {
    path.clear();
  } else if (!path.empty() && path.back() != '/') {
    path += '/';
  }
}

void EnsureNonDirectoryPath(std::string& path) {
  size_t size = path.size();
  while (size > 0 && path[size - 1] == '/') {
    --size;
  }
  path.resize(size);
}

void AppendPathComponent(std::string& path, std::string_view component) {
  if (!path.empty() && path.back() != '/' && !component.empty() &&
      component.front() != '/') {
    absl::StrAppend(&path, "/", component);
  } else {
    path += component;
  }
}

std::string LexicalNormalizePath(std::string path) {
  if (path.empty()) return path;

  const char* src = path.c_str();
  auto dst = path.begin();

  // Remove initial '/' characters.
  // Assume that Windows paths do not begin with '/'.
  const bool is_absolute_path = (*src == '/');
  if (is_absolute_path) {
    dst++;
    src++;
    while (*src == '/') ++src;
  }
  auto limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || src[1] == '\\' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' &&
                 (src[2] == '/' || src[2] == '\\' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != limit) {
          // We can backtrack the previous part
          for (--dst; dst != limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/' && *src != '\\') {
        *dst++ = *src++;
      }
      if (*src) {  // Always convert to /
        *dst++ = '/';
        src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/' || *src == '\\') {
      ++src;
    }
  }

  // The path may be empty.
  path.resize(dst - path.begin());
  return path;
}

}  // namespace internal
}  // namespace tensorstore
