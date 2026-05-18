// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_TESTING_TEST_HOOK_H_
#define TENSORSTORE_INTERNAL_TESTING_TEST_HOOK_H_

#if defined(TENSORSTORE_INTERNAL_TEST_HOOKS)

#include <functional>
#include <utility>

#include "absl/base/no_destructor.h"

namespace tensorstore {
namespace internal_testing {

/// TestHook provides a global registry for operation-specific hooks.
/// It is used to intercept low-level operations in tests (e.g., file I/O)
/// to simulate errors or specific behaviors.
///
/// The template parameter `Tag` must be a struct that defines a `HookFunc`
/// type, which is the signature of the hook function.
///
/// Example:
/// struct CloseOpTag {
///   using HookFunc = std::optional<absl::Status>(FileDescriptor);
/// };
///
/// In production code:
/// if (auto& hook = TestHook<CloseOpTag>::Get()) {
///   if (auto result = hook(fd)) return *result;
/// }
template <typename Tag>
struct TestHook {
  using HookFunc = typename Tag::HookFunc;

  static std::function<HookFunc>& Get() {
    static absl::NoDestructor<std::function<HookFunc>> instance;
    return *instance;
  }
};

/// ScopedTestHook is an RAII class to manage setting and restoring a hook.
/// It ensures that the hook is cleared when the object goes out of scope.
///
/// Example in a test:
/// {
///   ScopedTestHook<CloseOpTag> hook([](FileDescriptor fd) {
///     return absl::DataLossError("Simulated error");
///   });
///   // Operations here will trigger the hook.
/// }
/// // Hook is restored to its previous state here.
template <typename Tag>
class ScopedTestHook {
 public:
  using HookFunc = typename Tag::HookFunc;

  explicit ScopedTestHook(std::function<HookFunc> hook) {
    old_hook_ = std::move(TestHook<Tag>::Get());
    TestHook<Tag>::Get() = std::move(hook);
  }
  ~ScopedTestHook() { TestHook<Tag>::Get() = std::move(old_hook_); }

  ScopedTestHook(const ScopedTestHook&) = delete;
  ScopedTestHook& operator=(const ScopedTestHook&) = delete;

 private:
  std::function<HookFunc> old_hook_;
};

}  // namespace internal_testing
}  // namespace tensorstore

#define TENSORSTORE_INVOKE_TEST_HOOK(Tag, ...)                                \
  do {                                                                        \
    if (auto& hook = ::tensorstore::internal_testing::TestHook<Tag>::Get()) { \
      if (auto result = hook(__VA_ARGS__)) return *result;                    \
    }                                                                         \
  } while (0)

#else

#define TENSORSTORE_INVOKE_TEST_HOOK(Tag, ...) \
  do {                                         \
  } while (0)

#endif

#endif  // TENSORSTORE_INTERNAL_TESTING_TEST_HOOK_H_
