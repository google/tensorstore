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
    mode_(ios_base::out | ios_base::in),
    cord_(std::move(cord)),
    read_chunk_(cord_.Chunks().begin()) {

  assert(eback() == nullptr);
  assert(gptr() == nullptr);
  assert(egptr() == nullptr);
  assert(pbase() == nullptr);
  assert(pptr() == nullptr);
  assert(epptr() == nullptr);

  // Set up the get area to point at the first chunk,
  // if the Cord has data
  if(read_chunk_ != cord_.Chunks().end()) {
    char_type * data = const_cast<char_type *>(read_chunk_->data());
    setg(data, data, data + read_chunk_->size());
  }
}

Cord CordStreamBuf::MoveCord() {
  Cord result;
  std::swap(result, cord_);
  read_chunk_ = cord_.Chunks().begin();
  assert(read_chunk_ == cord_.Chunks().end());
  setg(nullptr, nullptr, nullptr);
  setp(nullptr, nullptr);
  return result;
}

void CordStreamBuf::TakeCord(Cord && cord) {
  setg(nullptr, nullptr, nullptr);
  setp(nullptr, nullptr);

  cord_ = std::move(cord);
  read_chunk_ = cord_.Chunks().begin();

  if(read_chunk_ != cord_.Chunks().end()) {
    char_type * data = const_cast<char_type *>(read_chunk_->data());
    setg(data, data, data + read_chunk_->size());
  }
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

  MaybeSetGetArea();
  return p;
}

// Handle buffer overflow.
CordStreamBuf::int_type CordStreamBuf::overflow(int_type ch) {
  // Not writing or eof received
  if(!(mode_ & ios_base::out)) return traits_type::eof();
  if(traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::eof();
  auto c = traits_type::to_char_type(ch);
  cord_.Append(absl::string_view(&c, 1));
  MaybeSetGetArea();
  return ch;
}

// Bulk get operation
streamsize CordStreamBuf::xsgetn(char * s, streamsize count) {
  // Not reading
  if(!(mode_ & ios_base::in)) return 0;
  streamsize bytes_read = 0;

  while(bytes_read < count && read_chunk_ != cord_.Chunks().end()) {
    assert(read_chunk_->size() == egptr() - eback());   // invariant
    auto bytes_to_read = std::min<streamsize>(gremaining(), count - bytes_read);
    for(streamsize i = 0, consumed = gconsumed(); i < bytes_to_read; ++i) {
      s[bytes_read + i] = read_chunk_->operator[](consumed + i);
    }
    if(gptr() + bytes_to_read < egptr()) {
      // Data remains in the get area
      setg(eback(), gptr() + bytes_to_read, egptr());
    } else if(++read_chunk_ != cord_.Chunks().end()) {
      // Initialise get area for next iteration
      char_type * data = const_cast<char_type *>(read_chunk_->data());
      setg(data, data, data + read_chunk_->size());
    }

    bytes_read += bytes_to_read;
  };

  return bytes_read;
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
      if(off > cord_.size()) return traits_type::eof();
      // Seek from the beginning of the cord
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
      auto current = gconsumed();
      for(auto c = cord_.Chunks().begin(); c != read_chunk_; ++c) {
        current += c->size();
      }

      auto n = off;
      // Advance forward in the current chunk
      if(n > 0 && gremaining() > 0) {
        auto bytes_to_remove = std::min<streamsize>(n, gremaining());
        n -= bytes_to_remove;
        gbump(bytes_to_remove);
      }
      // Advance forward by Cord chunks,
      // consuming any remaining characters
      // in the chunk
      while(n > 0) {
        if(++read_chunk_ == cord_.Chunks().end()) return traits_type::eof();
        auto bytes_to_advance = std::min<streamsize>(n, read_chunk_->size());
        char_type * data = const_cast<char_type *>(read_chunk_->data());
        setg(data, data + bytes_to_advance, data + read_chunk_->size());
        n -= bytes_to_advance;
      }

      return current + off;
    } else if (way == ios_base::end) {
      // Seeks past the stream end are unsupported
      if(off > 0) return traits_type::eof();
      auto n = cord_.size() + off;
      read_chunk_ = cord_.Chunks().begin();
      while(n > read_chunk_->size()) {
        n -= read_chunk_->size();
        ++read_chunk_;
      }
      char_type * data = const_cast<char_type *>(read_chunk_->data());
      setg(data, data + n, data + read_chunk_->size());
      return cord_.size() + off;
    }
  } else if (which == ios_base::out) {
    // This streambuf only supports appends.
    // Don't respect the off argument, always return
    // the append position
    return cord_.size();
  }
  return traits_type::eof();
}

streamsize CordStreamBuf::gconsumed() const {
  return gptr() - eback();
};

streamsize CordStreamBuf::gremaining() const {
  return egptr() - gptr();
}

void CordStreamBuf::MaybeSetGetArea() {
  if(read_chunk_ == cord_.Chunks().end()) {
    read_chunk_ = cord_.Chunks().begin();
    if(read_chunk_ == cord_.Chunks().end()) return;
    char_type * data = const_cast<char_type *>(read_chunk_->data());
    setg(data, data, data + read_chunk_->size());
  }
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
