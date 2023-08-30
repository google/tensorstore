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

#ifndef TENSORSTORE_INTERNAL_KVS_READ_STREAMBUF_H_
#define TENSORSTORE_INTERNAL_KVS_READ_STREAMBUF_H_

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/spec.h"

namespace tensorstore {
namespace internal {

class KvsReadStreambuf : public std::basic_streambuf<char> {
 public:
  KvsReadStreambuf(kvstore::DriverPtr kvstore_driver, kvstore::Key key,
                   std::streamoff pos_in_stream = 0);

  ~KvsReadStreambuf() override = default;

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override;
  pos_type seekoff(off_type off, std::ios_base::seekdir way,
                   std::ios_base::openmode which) override;

 private:
  int_type underflow() override;
  std::streamsize xsgetn(char* s, std::streamsize count) override;

  kvstore::DriverPtr kvstore_driver_;
  kvstore::Key key_;
  std::streamoff source_pos_;
  std::vector<char> current_buffer_;
};

}  // namespace internal
}  // namespace tensorstore

#endif
