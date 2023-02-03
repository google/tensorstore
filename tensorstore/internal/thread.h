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

#ifndef TENSORSTORE_INTERNAL_THREAD_H_
#define TENSORSTORE_INTERNAL_THREAD_H_

#include <climits>
#include <cstring>
#include <functional>
#include <thread>  // NOLINT
#include <utility>

#include "absl/log/absl_check.h"

namespace tensorstore {
namespace internal {

/// Helper functions to set the thread name.
void TrySetCurrentThreadName(const char *name);

// Tensorstore-specific Thread class to be used instead of std::thread.
// This exposes a limited subset of the std::thread api.
class Thread {
 public:
  using Id = std::thread::id;

  struct Options {
    const char *name = nullptr;
  };

  Thread() = default;

  // Creates a joinable thread with a functor (std::function or function
  // pointer) and optional arguments.
  template <class Function, class... Args>
  explicit Thread(Options options, Function &&f, Args &&...args)
      : Thread(private_t{}, options, f, args...) {}

  // Allow move, disallow copy.
  Thread(Thread &&other) noexcept = default;
  Thread &operator=(Thread &&other) = default;
  Thread(const Thread &other) = delete;
  Thread &operator=(const Thread &other) = delete;

  ~Thread() { ABSL_CHECK(!thread_.joinable()); }

  // Static method that starts a detached thread. Creates a thread without
  // returning externally visible Thread object. Allows execution to continue
  // independently of the caller thread. Any resources allocated by
  // StartDetached will be freed once the thread exits.
  template <class Function, class... Args>
  static void StartDetached(Options options, Function &&f, Args &&...args) {
    Thread(private_t{}, options, f, args...).thread_.detach();
  }

  // Joins the thread, blocking the current thread until the thread identified
  // by *this finishes execution. Not applicable to detached threads, since
  // StartDetach method does not return Thread object.
  void Join() {
    ABSL_CHECK_NE(this_thread_id(), get_id());
    thread_.join();
  }

  // Returns a unique id of the thread.
  Id get_id() const { return thread_.get_id(); }

  // Returns the current thread's id.
  static Id this_thread_id() { return std::this_thread::get_id(); }

 private:
  struct private_t {};

  // Private constructor creates a joinable or detachable thread with a functor
  // and optional arguments. Used by public constructor and by StartDetached
  // factory method.
  template <class Function, class... Args>
  Thread(private_t, Options options, Function &&f, Args &&...args)
      : thread_(
            [name = options.name, fn = std::bind(std::forward<Function>(f),
                                                 std::forward<Args>(args)...)] {
              TrySetCurrentThreadName(name);
              std::move(fn)();
            }) {}

  std::thread thread_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  //  TENSORSTORE_INTERNAL_THREAD_H_
