#ifndef TENSORSTORE_INTERNAL_SCHEDULE_AT_H_
#define TENSORSTORE_INTERNAL_SCHEDULE_AT_H_

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/time/time.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

/// Schedule an executor tast to run near a target time.
/// Long-running tasks should use WithExecutor() to avoid blocking the thread.
///
/// The return value of the `ExecutorTask` is ignored.
///
/// \ingroup async
void ScheduleAt(absl::Time target_time, ExecutorTask task);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SCHEDULE_AT_H_
