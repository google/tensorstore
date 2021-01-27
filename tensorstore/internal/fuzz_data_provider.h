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

#ifndef TENSORSTORE_INTERNAL_FUZZ_DATA_PROVIDER_H_
#define TENSORSTORE_INTERNAL_FUZZ_DATA_PROVIDER_H_

/// \file
///
/// Defines `FuzzDataProvider`, an interface which is used in random/fuzz
/// tests to supply parameters.

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

namespace tensorstore {
namespace internal {

/// Interface for providing fuzz/random data to KvsRandomOperationTester and
/// other random test generators.
class FuzzDataProvider {
 public:
  virtual ~FuzzDataProvider() = default;

  // Returns a set of ascii string keys.
  virtual std::vector<std::string> GenerateKeys() = 0;

  // Returns a Uniform value between [lower, upper]
  virtual size_t Uniform(size_t lower, size_t upper) = 0;

  // Returns a Uniform value between [lower, upper]
  virtual int64_t UniformInt(int64_t lower, int64_t upper) = 0;

  // Returns true with a probability of p
  virtual bool Bernoulli(double p) = 0;
};

/// Returns a default FuzzDataProvider implementation.
std::unique_ptr<FuzzDataProvider> MakeDefaultFuzzDataProvider();

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_KVS_BACKED_CACHE_TESTUTIL_H_
