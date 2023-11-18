// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_THREAD_SCHEDULE_AT_H_
#define TENSORSTORE_INTERNAL_THREAD_SCHEDULE_AT_H_

#include "absl/functional/any_invocable.h"
#include "absl/time/time.h"
#include "tensorstore/util/stop_token.h"

namespace tensorstore {
namespace internal {

/// Schedule an executor tast to run near a target time.
/// Long-running tasks should use WithExecutor() to avoid blocking the thread.
///
/// \ingroup async
void ScheduleAt(absl::Time target_time, absl::AnyInvocable<void() &&> task,
                const StopToken& stop_token = {});

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_SCHEDULE_AT_H_
