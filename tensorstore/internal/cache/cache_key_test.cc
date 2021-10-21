#include "tensorstore/internal/cache/cache_key.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace {

TEST(CacheKeyTest, CacheKeyTest) {
  int x = 1;
  float y = 2;
  std::string q("q");
  absl::Duration d = absl::Seconds(1);
  std::optional<int> o = 2;

  std::string key;
  tensorstore::internal::EncodeCacheKey(&key, x, y, q, d, o);

  {
    std::string key2;
    tensorstore::internal::EncodeCacheKey(&key2, x, y, q, d,
                                          std::optional<int>{});
    EXPECT_NE(key, key2);
  }
  {
    std::string key3;
    tensorstore::internal::EncodeCacheKey(&key3, x, y, q,
                                          absl::InfiniteDuration(), o);
    EXPECT_NE(key, key3);
  }
  {
    std::string key4;
    tensorstore::internal::EncodeCacheKey(
        &key4, x, y, q, d, tensorstore::internal::CacheKeyExcludes{o});
    EXPECT_NE(key, key4);
  }
}

}  // namespace
