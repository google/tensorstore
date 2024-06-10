// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/kvstore/s3_sdk/cord_streambuf.h"

#include <ios>
#include <numeric>
#include <string_view>

#include "absl/strings/cord.h"

using absl::Cord;
using absl::CordBuffer;
using std::streamsize;
using std::streampos;
using std::streamoff;
using std::ios_base;

namespace tensorstore {
namespace internal_kvstore_s3 {

CordStreamBuf::CordStreamBuf() :
  CordStreamBuf(Cord()) {}

CordStreamBuf::CordStreamBuf(Cord && cord) :
    mode_(cord.size() == 0 ? ios_base::out : ios_base::in),
    cord_(std::move(cord)),
    read_chunk_(cord_.Chunks().begin()) {

  // Set up the get area, if the Cord has data
  if(read_chunk_ != cord_.Chunks().end()) {
    char * data = const_cast<char *>(read_chunk_->data());
    setg(data, data, data + read_chunk_->size());
  }

}

Cord CordStreamBuf::GetCord() {
  Cord result;
  std::swap(result, cord_);
  read_chunk_ = cord_.Chunks().begin();
  char dummy;
  setg(&dummy, &dummy, &dummy + 1);
  setp(&dummy, &dummy + 1);
  return result;
}

// Bulk put operation
streamsize CordStreamBuf::xsputn(const char * s, streamsize count) {
  if(!(mode_ & ios_base::out)) return 0;
  streamsize n = count;
  bool first = true;
  streamsize p = 0;

  while (n > 0) {
    CordBuffer buffer = first ? cord_.GetAppendBuffer(n)
                              : CordBuffer::CreateWithDefaultLimit(n);

    auto span = buffer.available_up_to(n);
    for(int i = 0; i < span.size(); ++i, ++p) span[i] = s[p];
    buffer.IncreaseLengthBy(span.size());
    cord_.Append(std::move(buffer));
    n -= span.size();
    first = false;
  }

  return p;
}

// Handle buffer overflow.
CordStreamBuf::int_type CordStreamBuf::overflow(int_type ch) {
  // Not writing or eof received
  if(!(mode_ & ios_base::out)) return traits_type::eof();
  if(traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::eof();
  auto c = traits_type::to_char_type(ch);
  cord_.Append(absl::string_view(&c, 1));
  return ch;
}

// Bulk get operation
streamsize CordStreamBuf::xsgetn(char * s, streamsize count) {
  // Not reading or no more Cord data
  if(!(mode_ & ios_base::in)) return 0;
  if(read_chunk_ == cord_.Chunks().end()) return 0;
  auto bytes_to_read = std::min<streamsize>(read_chunk_->size(), count);
  for(streamsize i = 0; i < bytes_to_read; ++i) s[i] = read_chunk_->operator[](i);
  assert(gptr() + bytes_to_read <= egptr());
  if(gptr() + bytes_to_read < egptr()) {
    setg(eback(), gptr() + bytes_to_read, egptr());
  } else {
    if(++read_chunk_ != cord_.Chunks().end()) {
      char_type * data = const_cast<char_type *>(read_chunk_->data());
      setg(data, data, data + read_chunk_->size());
    }
  }
  return bytes_to_read;
}

// Handle buffer underflow.
CordStreamBuf::int_type CordStreamBuf::underflow() {
  // Not reading or no more Cord data
  if(!(mode_ & ios_base::in)) return traits_type::eof();
  if(read_chunk_ == cord_.Chunks().end()) return traits_type::eof();
  if(gptr() < egptr()) {
    return traits_type::to_int_type(*gptr());
  }
  if(++read_chunk_ == cord_.Chunks().end()) return traits_type::eof();
  char_type * data = const_cast<char_type *>(read_chunk_->data());
  setg(data, data, data + read_chunk_->size());
  return traits_type::to_int_type(*data);
}

streampos CordStreamBuf::seekoff(streamoff off, ios_base::seekdir way, ios_base::openmode which) {
  if (which == ios_base::in) {
    if (way == ios_base::beg) {
      // Seek from the beginning of the cord
      if(off >= cord_.size()) return traits_type::eof();
      auto n = off;
      read_chunk_ = cord_.Chunks().begin();
      while(n > read_chunk_->size()) {
        n -= read_chunk_->size();
        ++read_chunk_;
      }
      char_type * data = const_cast<char_type *>(read_chunk_->data());
      setg(data, data + n, data + read_chunk_->size());
      return off;
    } else if (way == ios_base::cur) {
      if(read_chunk_ == cord_.Chunks().end()) return traits_type::eof();
      auto n = off;
      // Advance forward by Cord chunk,
      // consuming any remaining characters
      // in the chunk
      while(n >= remaining()) {
        n -= remaining();
        if(++read_chunk_ == cord_.Chunks().end()) return traits_type::eof();
        char_type * data = const_cast<char_type *>(read_chunk_->data());
        setg(data, data, data + read_chunk_->size());
      }
      setg(eback(), gptr() + n, egptr());
      return std::accumulate(cord_.Chunks().begin(), read_chunk_, consumed(),
                             [](auto i, auto c) { return i + c.size(); });
    } else if (way == ios_base::end) {
      // Seeks past the stream end are unsupported
      return traits_type::eof();
    }
  } else if (which == ios_base::out) {
    // Only support appends
    return traits_type::eof();
  }
  return traits_type::eof();
}

std::streamsize CordStreamBuf::consumed() const {
  return gptr() - eback();
};
std::streamsize CordStreamBuf::remaining() const {
  return egptr() - gptr();
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
