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

#include "tensorstore/internal/rate_limiter/rate_limiter.h"

#include <cassert>

namespace tensorstore {
namespace internal {

void RateLimiter::RunStartFunction(RateLimiterNode* node) {
  // Next node gets a chance to run after clearing admission queue state.
  RateLimiterNode::StartFn fn = node->start_fn_;
  assert(fn != nullptr);
  node->next_ = nullptr;
  node->prev_ = nullptr;
  node->start_fn_ = nullptr;
  fn(node);
}

void NoRateLimiter::Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;
  RunStartFunction(node);
}

void NoRateLimiter::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
}

}  // namespace internal
}  // namespace tensorstore
