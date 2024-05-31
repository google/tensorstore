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

#ifndef TENSORSTORE_KVSTORE_S3_STREAMBUF_H_
#define TENSORSTORE_KVSTORE_S3_STREAMBUF_H_

#include <ios>
#include <streambuf>

#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

/// Basic implementation of a std::basic_streambuf<char>
/// backed by an abseil Cord.
/// It should be used in two modes
/// (1) Append-only writing mode, where data is appended to the underlying Cord
/// (2) Read mode, where data is read from the Cord. Seeking is supported
///     within the Stream Buffer.
class CordStreamBuf : public std::basic_streambuf<char> {
public:
  // Creates a stream buffer for writing
  CordStreamBuf();
  // Creates a stream buffer for reading from the supplied Cord
  CordStreamBuf(absl::Cord && cord);

  // Obtain read access to the backing Cord
  const absl::Cord & GetCord() const { return cord_; }

  // Returns the underlying Cord, resetting the underlying stream
  absl::Cord GetCord();

protected:
  // Bulk put operation
  virtual std::streamsize xsputn(const char * s, std::streamsize count) override;
  // Bulk get operation
  virtual std::streamsize xsgetn(char * s, std::streamsize count) override;
  // Handle buffer overflow.
  virtual int_type overflow(int_type ch) override;
  // Handle buffer underflow.
  virtual int_type underflow() override;

  // Seek within the underlying Cord (only seeks in the get area are supported)
  virtual std::streampos seekoff(
    std::streamoff off,
    std::ios_base::seekdir way,
    std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override;

  // Seek within the underlying Cord (only seeks in the get area are supported)
  virtual std::streampos seekpos(
    std::streampos sp,
    std::ios_base::openmode which = std::ios_base::in | std::ios_base::out) override {
      return seekoff(sp - std::streampos(0), std::ios_base::beg, which);
  }

private:
  std::ios_base::openmode mode_;
  absl::Cord cord_;
  absl::Cord::CharIterator get_iterator_;
};

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif // TENSORSTORE_KVSTORE_S3_STREAMBUF_H_