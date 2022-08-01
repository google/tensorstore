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

#ifndef TENSORSTORE_UTIL_STOP_TOKEN_H_
#define TENSORSTORE_UTIL_STOP_TOKEN_H_

#include <atomic>
#include <type_traits>
#include <utility>

#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/stop_token_impl.h"

/// \file
///
/// Implementation of cancellation mechanism similar to the C++20 `stop_token`
/// library.
///
/// Example:
///
///     tensorstore::StopSource source;
///     bool called = false;
///     tensorstore::StopCallback callback{source.get_token(), [&] {
///       called = true;
///     });
///     EXPECT_FALSE(called);
///     EXPECT_TRUE(source.request_stop());
///     EXPECT_TRUE(called);
///
namespace tensorstore {

class StopSource;
template <typename Callback>
class StopCallback;

/// Provides a mechanism to test whether a stop has been requested on an
/// asynchronous operation.
class StopToken {
 public:
  StopToken() noexcept = default;
  ~StopToken() noexcept = default;

  StopToken(const StopToken&) noexcept = default;
  StopToken(StopToken&&) noexcept = default;
  StopToken& operator=(const StopToken&) noexcept = default;
  StopToken& operator=(StopToken&&) noexcept = default;

  /// Detects whether a stop may be requested
  [[nodiscard]] bool stop_possible() const noexcept {
    return state_ != nullptr;
  }

  /// Detects whether a stop has been requested on the assoicated StopState.
  [[nodiscard]] bool stop_requested() const noexcept {
    return state_ != nullptr && state_->stop_requested();
  }

  friend bool operator==(const StopToken& a, const StopToken& b) {
    return a.state_ == b.state_;
  }
  friend bool operator!=(const StopToken& a, const StopToken& b) {
    return !(a == b);
  }

 private:
  friend class StopSource;
  template <typename Callback>
  friend class StopCallback;

  StopToken(internal::IntrusivePtr<internal_stop_token::StopState> state)
      : state_(std::move(state)) {}

  internal::IntrusivePtr<internal_stop_token::StopState> state_{nullptr};
};

/// Provides a mechanism to stop an asynchronous request.
/// Once a stop has been requested, it cannot be withdrawn, and all future
/// StopCallbacks registered with the stop_token() will be immediately invoked.
class StopSource {
 public:
  StopSource() noexcept
      : state_(internal::MakeIntrusivePtr<internal_stop_token::StopState>()) {}

  explicit StopSource(std::nullptr_t) noexcept : state_(nullptr) {}

  ~StopSource() noexcept = default;

  StopSource(const StopSource& b) noexcept = default;
  StopSource(StopSource&&) noexcept = default;
  StopSource& operator=(const StopSource& b) noexcept = default;
  StopSource& operator=(StopSource&&) noexcept = default;

  [[nodiscard]] bool stop_possible() const noexcept {
    return state_ != nullptr;
  }

  [[nodiscard]] bool stop_requested() const noexcept {
    return state_ != nullptr && state_->stop_requested();
  }

  /// Requests a stop. Once a stop has been requested, it cannot be withdrawn.
  /// The first call to request_stop() will invoke all StopCallbacks registered
  /// with the StopSource, and will return true.
  /// Subsequent (or concurrent) calls to request_stop() will return false and
  /// the callbacks may still be in-flight.
  bool request_stop() const noexcept {
    if (state_ != nullptr) {
      return state_->RequestStop();
    }
    return false;
  }

  [[nodiscard]] StopToken get_token() const noexcept {
    return StopToken(state_);
  }

 private:
  internal::IntrusivePtr<internal_stop_token::StopState> state_;
};

/// Registers a callback to run when a stop is requested on the StopSource
/// associated with `token`.
///
/// If the callback throws an exception when called, `std::terminate` is
/// called.
///
/// If a stop has already been requested on `token`, the callback is invoked
/// in the current thread before the constructor returns.  Otherwise, if a
/// stop is requested on the stop state associated with `token` before this
/// `StopCallback` is destroyed, the callback will be invoked synchronously
/// from the thread that requested the stop.
///
template <typename Callback>
class StopCallback : private internal_stop_token::StopCallbackBase {
  static_assert(std::is_invocable_v<Callback>);

 public:
  using callback_type = Callback;

  StopCallback(const StopCallback&) = delete;
  StopCallback& operator=(const StopCallback&) = delete;
  StopCallback(StopCallback&&) = delete;
  StopCallback& operator=(StopCallback&&) = delete;

  template <
      typename... Args,
      std::enable_if_t<std::is_constructible_v<Callback, Args...>, int> = 0>
  explicit StopCallback(const StopToken& token, Args&&... args)
      : callback_(std::forward<Args>(args)...) {
    internal_stop_token::StopState* state = token.state_.get();
    if (state) {
      invoker_ = &StopCallback::Invoker;
      state->RegisterImpl(*this);
    }  /// else null state, callback never invoked.
  }

  ~StopCallback() {
    /// state_ will be nullptr if the callback was never registered, or if it
    /// has been invoked (via request_stop).
    internal_stop_token::StopState* state =
        state_.exchange(nullptr, std::memory_order_acq_rel);
    if (state != nullptr) {
      /// UnregisterImpl decrements state reference count.
      state->UnregisterImpl(*this);
    }
  }

 private:
  static void Invoker(internal_stop_token::StopCallbackBase& self) noexcept {
    static_cast<Callback&&>(static_cast<StopCallback&&>(self).callback_)();
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Callback callback_;
};

template <typename Callback>
StopCallback(StopToken token, Callback callback) -> StopCallback<Callback>;

}  // namespace tensorstore

#endif
