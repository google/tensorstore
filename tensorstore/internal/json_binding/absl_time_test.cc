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

#include "tensorstore/internal/json_binding/absl_time.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/time/civil_time.h"
#include "absl/time/time.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"

namespace jb = tensorstore::internal_json_binding;

namespace {

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
