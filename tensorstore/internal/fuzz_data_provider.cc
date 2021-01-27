#include "tensorstore/internal/fuzz_data_provider.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "tensorstore/internal/test_util.h"

namespace tensorstore {
namespace internal {
namespace {

/// Default implementation of fuzzer supported by a PRNG.
class DefaultFuzzDataProvider : public FuzzDataProvider {
 public:
  ~DefaultFuzzDataProvider() override = default;

  std::vector<std::string> GenerateKeys() override {
    std::vector<std::string> keys;
    for (size_t i = 0; i < 10; ++i) {
      keys.push_back(std::string{static_cast<char>('a' + i)});
    }
    return keys;
  }

  size_t Uniform(size_t lower, size_t upper) override {
    return absl::Uniform(absl::IntervalClosedClosed, bitgen_, lower, upper);
  }

  int64_t UniformInt(int64_t lower, int64_t upper) override {
    return absl::Uniform(absl::IntervalClosedClosed, bitgen_, lower, upper);
  }

  bool Bernoulli(double p) override {  //
    return absl::Bernoulli(bitgen_, p);
  }

 private:
  std::minstd_rand bitgen_{internal::GetRandomSeedForTest(
      "TENSORSTORE_INTERNAL_FUZZ_DATA_PROVIDER")};
};

}  // namespace

std::unique_ptr<FuzzDataProvider> MakeDefaultFuzzDataProvider() {
  return std::make_unique<DefaultFuzzDataProvider>();
}

}  // namespace internal
}  // namespace tensorstore
