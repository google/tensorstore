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

#include "tensorstore/internal/digest/md5.h"

#include <string_view>

#include "absl/strings/cord.h"
#include <openssl/md5.h>

namespace tensorstore {
namespace internal {

void MD5Digester::Write(const absl::Cord& cord) {
  for (std::string_view chunk : cord.Chunks()) {
    Write(chunk);
  }
}

MD5Digester::DigestType MD5Digester::Digest() {
  DigestType digest;
  MD5_Final(digest.data(), &ctx_);
  return digest;
}

}  // namespace internal
}  // namespace tensorstore
