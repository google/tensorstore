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

#ifndef TENSORSTORE_KVSTORE_SUPPORTED_FEATURES_H_
#define TENSORSTORE_KVSTORE_SUPPORTED_FEATURES_H_

#include <stdint.h>

namespace tensorstore {
namespace kvstore {

enum class SupportedFeatures : uint64_t {
  kNone = 0,

  /// Indicates if there is race-free support for `WriteOptions::if_equal` set
  /// to `StorageGeneration::NoValue()`.  This is a subset of the guarantee
  /// provided by `kSingleKeyAtomicReadModifyWrite`.
  kAtomicWriteWithoutOverwrite = 4,

  /// Indicates if single-key atomic read-modify-write operations are supported,
  /// i.e. `WriteOptions::if_equal` is handled race-free.  This implies
  /// `kSingleKeyAtomicReadModifyWrite`.
  kSingleKeyAtomicReadModifyWrite = 8,
};

constexpr inline SupportedFeatures operator&(SupportedFeatures a,
                                             SupportedFeatures b) {
  return static_cast<SupportedFeatures>(static_cast<uint64_t>(a) &
                                        static_cast<uint64_t>(b));
}

constexpr inline SupportedFeatures operator|(SupportedFeatures a,
                                             SupportedFeatures b) {
  return static_cast<SupportedFeatures>(static_cast<uint64_t>(a) |
                                        static_cast<uint64_t>(b));
}

}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_SUPPORTED_FEATURES_H_
