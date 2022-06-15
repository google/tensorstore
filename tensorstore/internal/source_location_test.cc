#include "tensorstore/internal/source_location.h"

#include <gtest/gtest.h>

namespace {

using tensorstore::SourceLocation;

std::uint64_t TakesSourceLocation(
    SourceLocation loc TENSORSTORE_LOC_CURRENT_DEFAULT_ARG) {
  return loc.line();
}

TEST(SourceLocationTest, Basic) {
  constexpr tensorstore::SourceLocation loc = TENSORSTORE_LOC;
  EXPECT_NE(0, loc.line());

  EXPECT_NE(0, TakesSourceLocation(TENSORSTORE_LOC));
}

}  // namespace
