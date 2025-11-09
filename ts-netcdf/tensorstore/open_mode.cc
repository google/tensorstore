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

#include "tensorstore/open_mode.h"

#include <ostream>

#include "absl/status/status.h"

namespace tensorstore {

std::string_view to_string(ReadWriteMode mode) {
  switch (mode) {
    case ReadWriteMode::dynamic:
      return "dynamic";
    case ReadWriteMode::read:
      return "read";
    case ReadWriteMode::write:
      return "write";
    case ReadWriteMode::read_write:
      return "read_write";
    default:
      return "<unknown>";
  }
}

std::ostream& operator<<(std::ostream& os, ReadWriteMode mode) {
  return os << to_string(mode);
}

std::ostream& operator<<(std::ostream& os, OpenMode mode) {
  const char* sep = "";
  constexpr const char* kSep = "|";
  if (!!(mode & OpenMode::open)) {
    os << "open";
    sep = kSep;
  }
  if (!!(mode & OpenMode::create)) {
    os << sep << "create";
    sep = kSep;
  }
  if (!!(mode & OpenMode::delete_existing)) {
    os << sep << "delete_existing";
    sep = kSep;
  }
  if (!!(mode & OpenMode::assume_metadata)) {
    os << sep << "assume_metadata";
    sep = kSep;
  }
  return os;
}

namespace internal {
absl::Status ValidateSupportsRead(ReadWriteMode mode) {
  return !(mode & ReadWriteMode::read)
             ? absl::InvalidArgumentError("Source does not support reading.")
             : absl::Status();
}

absl::Status ValidateSupportsWrite(ReadWriteMode mode) {
  return !(mode & ReadWriteMode::write)
             ? absl::InvalidArgumentError(
                   "Destination does not support writing.")
             : absl::Status();
}

absl::Status ValidateSupportsModes(ReadWriteMode mode,
                                   ReadWriteMode required_modes) {
  if ((mode & required_modes) != required_modes) {
    if (!!(required_modes & ReadWriteMode::read) &&
        !(mode & ReadWriteMode::read)) {
      return absl::InvalidArgumentError("Read mode not supported");
    }
    if (!!(required_modes & ReadWriteMode::write) &&
        !(mode & ReadWriteMode::write)) {
      return absl::InvalidArgumentError("Write mode not supported");
    }
  }
  return absl::OkStatus();
}

}  // namespace internal

}  // namespace tensorstore
