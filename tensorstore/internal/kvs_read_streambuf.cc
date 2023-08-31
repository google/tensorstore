// Copyright 2023 The TensorStore Authors
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

// Based off of google-cloud-cpp object_read_stream
// Copyright 2021 Google LLC

#include "tensorstore/internal/kvs_read_streambuf.h"

#include "tensorstore/kvstore/driver.h"

namespace tensorstore {
namespace internal {
KvsReadStreambuf::KvsReadStreambuf(kvstore::DriverPtr kvstore_driver,
                                   kvstore::Key key, size_t buffer_size,
                                   std::streamoff pos_in_stream)
    : kvstore_driver_(std::move(kvstore_driver)),
      key_(std::move(key)),
      source_pos_(pos_in_stream),
      buffer_size_(buffer_size) {}

KvsReadStreambuf::pos_type KvsReadStreambuf::seekpos(
    pos_type sp, std::ios_base::openmode which) {
  return seekoff(sp - pos_type(off_type(0)), std::ios_base::beg, which);
}

KvsReadStreambuf::pos_type KvsReadStreambuf::seekoff(
    off_type off, std::ios_base::seekdir way, std::ios_base::openmode which) {
  // We don't know the total size of the object so we can't seek relative
  // to the end.
  if (which != std::ios_base::in || way == std::ios_base::end) return -1;
  if (way == std::ios_base::cur) {  // Convert relative to absolute position.
    off = source_pos_ - in_avail() + off;
  }

  if (off < 0) return -1;

  long buff_size = static_cast<long>(current_buffer_.size());
  long rel_off = off - (source_pos_ - buff_size);

  char* data = current_buffer_.data();
  if (rel_off < 0 || rel_off > buff_size) {
    setg(data, data + buff_size, data + buff_size);
    source_pos_ = off;
    underflow();
  } else {
    setg(data, data + rel_off, data + buff_size);
  }
  return source_pos_ - in_avail();
}

KvsReadStreambuf::int_type KvsReadStreambuf::underflow() {
  std::vector<char> buffer(buffer_size_);
  auto const offset = xsgetn(buffer.data(), buffer_size_);
  if (offset == 0) return traits_type::eof();
  buffer.resize(static_cast<std::size_t>(offset));
  buffer.swap(current_buffer_);
  char* data = current_buffer_.data();
  setg(data, data, data + current_buffer_.size());
  return traits_type::to_int_type(*data);
}

std::streamsize KvsReadStreambuf::xsgetn(char* s, std::streamsize count) {
  std::streamsize offset = 0;
  auto from_internal = (std::min)(count, in_avail());
  if (from_internal > 0) {
    std::memcpy(s, gptr(), static_cast<std::size_t>(from_internal));
  }
  gbump(static_cast<int>(from_internal));
  offset += from_internal;
  if (offset >= count) return offset;

  kvstore::ReadOptions options;
  options.staleness_bound = absl::Now();
  options.if_not_equal = StorageGeneration::NoValue();
  options.byte_range =
      ByteRange{static_cast<int64_t>(source_pos_),
                static_cast<int64_t>(count + source_pos_ - offset)};

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto result, kvstore_driver_->Read(key_, options).result(), offset);
  auto data = result.value.Flatten();
  std::memcpy(s + offset, data.data(), data.size());

  offset += static_cast<std::streamsize>(data.size());
  source_pos_ += static_cast<std::streamoff>(data.size());
  return offset;
}

}  // namespace internal
}  // namespace tensorstore