#include "tensorstore/internal/absl_time_json_binder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/json_gtest.h"

namespace {

namespace jb = tensorstore::internal_json_binding;

TEST(AbslTimeJsonBinder, Roundtrips) {
  const absl::TimeZone utc = absl::UTCTimeZone();
  const absl::CivilSecond cs(2015, 2, 3, 4, 5, 6);

  tensorstore::TestJsonBinderRoundTrip<absl::Time>(
      {
          {absl::FromCivil(cs, utc), "2015-02-03T04:05:06+00:00"},
          {absl::FromCivil(absl::CivilMinute(cs), utc),
           "2015-02-03T04:05:00+00:00"},
          {absl::FromCivil(absl::CivilHour(cs), utc),
           "2015-02-03T04:00:00+00:00"},
          {absl::FromCivil(absl::CivilDay(cs), utc),
           "2015-02-03T00:00:00+00:00"},
          {absl::FromCivil(absl::CivilMonth(cs), utc),
           "2015-02-01T00:00:00+00:00"},
          {absl::FromCivil(absl::CivilYear(cs), utc),
           "2015-01-01T00:00:00+00:00"},
      },
      jb::Rfc3339TimeBinder);
}

}  // namespace
