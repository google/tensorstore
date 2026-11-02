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

#include "tensorstore/kvstore/gcs_grpc/zonal_bucket_state.h"

#include <optional>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"

namespace {

using ::tensorstore::internal_gcs_grpc::GcsGrpcWriteMode;
using ::tensorstore::internal_gcs_grpc::ZonalBucketState;
using ProbeAction =
    ::tensorstore::internal_gcs_grpc::ZonalBucketState::ProbeAction;

TEST(ZonalBucketStateTest, DefaultStateAndProbeReleaseOnDestruction) {
  ZonalBucketState state;
  EXPECT_EQ(state.is_zonal(), std::nullopt);
  EXPECT_EQ(state.routing_token(), "");

  auto waiter = ZonalBucketState::ProbeResult{};
  {
    auto probe = state.AcquireProbe();
    ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
    EXPECT_TRUE(static_cast<bool>(probe.lease));

    waiter = state.AcquireProbe();
    ASSERT_EQ(waiter.action, ProbeAction::kWait);
    EXPECT_FALSE(waiter.wait_future.ready());
    // `probe.lease` goes out of scope without `Finish(true)`.
  }

  EXPECT_TRUE(waiter.wait_future.ready());
  EXPECT_EQ(state.is_zonal(), std::nullopt);

  // Because bucket mode is still unknown, the next caller becomes the new
  // probe.
  auto next_probe = state.AcquireProbe();
  EXPECT_EQ(next_probe.action, ProbeAction::kIsProbe);
}

TEST(ZonalBucketStateTest, ConfiguredModesIgnoreServiceSignals) {
  {
    ZonalBucketState standard_state;
    standard_state.set_configured_mode(GcsGrpcWriteMode::kStandard);
    EXPECT_EQ(standard_state.is_zonal(), false);
    EXPECT_EQ(standard_state.AcquireProbe().action, ProbeAction::kKnown);

    standard_state.set_is_zonal(true);
    EXPECT_EQ(standard_state.is_zonal(), false);
  }
  {
    ZonalBucketState zonal_state;
    zonal_state.set_configured_mode(GcsGrpcWriteMode::kZonal);
    EXPECT_EQ(zonal_state.is_zonal(), true);
    EXPECT_FALSE(zonal_state.is_non_atomic());

    auto probe = zonal_state.AcquireProbe();
    ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
    probe.lease.Finish(/*mark_done=*/true);
    EXPECT_EQ(zonal_state.AcquireProbe().action, ProbeAction::kKnown);

    zonal_state.set_is_zonal(false);
    EXPECT_EQ(zonal_state.is_zonal(), true);
  }
  {
    ZonalBucketState non_atomic_state;
    non_atomic_state.set_configured_mode(GcsGrpcWriteMode::kNonAtomic);
    EXPECT_EQ(non_atomic_state.is_zonal(), true);
    EXPECT_TRUE(non_atomic_state.is_non_atomic());

    auto probe = non_atomic_state.AcquireProbe();
    ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
    probe.lease.Finish(/*mark_done=*/true);
    EXPECT_EQ(non_atomic_state.AcquireProbe().action, ProbeAction::kKnown);

    non_atomic_state.set_is_zonal(false);
    EXPECT_EQ(non_atomic_state.is_zonal(), true);
  }
}

TEST(ZonalBucketStateTest, ContradictionPinningFromNonZonalToZonal) {
  ZonalBucketState state;

  auto probe = state.AcquireProbe();
  ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
  auto waiter = state.AcquireProbe();
  ASSERT_EQ(waiter.action, ProbeAction::kWait);

  state.set_is_zonal(false);
  EXPECT_EQ(state.is_zonal(), false);
  EXPECT_TRUE(waiter.wait_future.ready());
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);

  // Contradicting an earlier `false` signal pins the state to non-zonal so the
  // driver does not oscillate between write paths.
  state.set_is_zonal(true);
  EXPECT_EQ(state.is_zonal(), false);
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);

  state.set_is_zonal(true);
  EXPECT_EQ(state.is_zonal(), false);
}

TEST(ZonalBucketStateTest, ContradictionPinningFromZonalToNonZonalWakesWaiter) {
  ZonalBucketState state;
  state.set_is_zonal(true);
  EXPECT_EQ(state.is_zonal(), true);

  auto probe = state.AcquireProbe();
  ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
  auto waiter = state.AcquireProbe();
  ASSERT_EQ(waiter.action, ProbeAction::kWait);
  EXPECT_FALSE(waiter.wait_future.ready());

  // Contradicting an earlier `true` signal pins the state to non-zonal and
  // immediately wakes any waiting writes.
  state.set_is_zonal(false);
  EXPECT_EQ(state.is_zonal(), false);
  EXPECT_TRUE(waiter.wait_future.ready());
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);

  // Once pinned, subsequent `true` signals cannot flip the bucket back to
  // zonal.
  state.set_is_zonal(true);
  EXPECT_EQ(state.is_zonal(), false);
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);
}

TEST(ZonalBucketStateTest, RoutingTokenCompareAndClearAndEmptyGuards) {
  ZonalBucketState state;
  state.set_is_zonal(true);

  auto probe = state.AcquireProbe();
  ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
  auto waiter = state.AcquireProbe();
  ASSERT_EQ(waiter.action, ProbeAction::kWait);

  // An empty token is a no-op and must not prematurely complete the routing
  // probe.
  state.set_routing_token("");
  EXPECT_EQ(state.routing_token(), "");
  EXPECT_FALSE(waiter.wait_future.ready());

  // Recording a non-empty routing token completes the routing probe and wakes
  // waiting writes.
  state.set_routing_token("tok-1");
  EXPECT_EQ(state.routing_token(), "tok-1");
  EXPECT_TRUE(waiter.wait_future.ready());
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);

  // Empty or non-matching tokens do not clear the active routing token.
  state.set_routing_token("");
  EXPECT_EQ(state.routing_token(), "tok-1");
  state.clear_routing_token("");
  EXPECT_EQ(state.routing_token(), "tok-1");
  state.clear_routing_token("stale-tok");
  EXPECT_EQ(state.routing_token(), "tok-1");

  // Matching token clears the cached value.
  state.clear_routing_token("tok-1");
  EXPECT_EQ(state.routing_token(), "");
}

TEST(ZonalBucketStateTest,
     SetIsZonalTrueWithCachedRoutingTokenEntersReadyStateImmediately) {
  ZonalBucketState state;
  auto probe = state.AcquireProbe();
  ASSERT_EQ(probe.action, ProbeAction::kIsProbe);
  auto waiter = state.AcquireProbe();
  ASSERT_EQ(waiter.action, ProbeAction::kWait);

  // A redirect during the initial bucket-type probe caches the routing token,
  // but keeps waiters gated until `is_zonal` is known.
  state.set_routing_token("tok-early");
  EXPECT_EQ(state.routing_token(), "tok-early");
  EXPECT_FALSE(waiter.wait_future.ready());

  // Once the bucket is confirmed zonal, the already-cached routing token allows
  // transitioning directly to `kZonalReady` and waking waiters.
  state.set_is_zonal(true);
  EXPECT_EQ(state.is_zonal(), true);
  EXPECT_TRUE(waiter.wait_future.ready());
  EXPECT_EQ(state.AcquireProbe().action, ProbeAction::kKnown);
}

TEST(ZonalBucketStateTest, ProbeLeaseMoveAssignmentReleasesOverwrittenLease) {
  ZonalBucketState state1;
  state1.set_is_zonal(true);
  auto probe1 = state1.AcquireProbe();
  ASSERT_EQ(probe1.action, ProbeAction::kIsProbe);
  auto waiter1 = state1.AcquireProbe();
  ASSERT_EQ(waiter1.action, ProbeAction::kWait);

  ZonalBucketState state2;
  state2.set_is_zonal(true);
  auto probe2 = state2.AcquireProbe();
  ASSERT_EQ(probe2.action, ProbeAction::kIsProbe);
  auto waiter2 = state2.AcquireProbe();
  ASSERT_EQ(waiter2.action, ProbeAction::kWait);

  // Move-assigning `probe2.lease` over `probe1.lease` finishes `probe1` with
  // `mark_done = false`, waking `waiter1` while leaving `state1` in
  // `kZonalProbingRouting`.
  probe1.lease = std::move(probe2.lease);
  EXPECT_FALSE(static_cast<bool>(probe2.lease));
  EXPECT_TRUE(waiter1.wait_future.ready());
  EXPECT_EQ(state1.AcquireProbe().action, ProbeAction::kIsProbe);
  EXPECT_FALSE(waiter2.wait_future.ready());

  probe1.lease.Finish(/*mark_done=*/true);
  EXPECT_TRUE(waiter2.wait_future.ready());
  EXPECT_EQ(state2.AcquireProbe().action, ProbeAction::kKnown);
}

TEST(ZonalBucketStateTest, StaleProbeLeaseFinishTrueDoesNotCompleteNewProbe) {
  ZonalBucketState state;

  // Acquire the initial bucket-type probe (probe_id == 1).
  auto first_probe = state.AcquireProbe();
  ASSERT_EQ(first_probe.action, ProbeAction::kIsProbe);

  // Resetting the probe state (`set_is_zonal(std::nullopt)`) releases the
  // first probe slot while `first_probe.lease` is still alive, and setting
  // `is_zonal = true` enters `kZonalProbingRouting` so a new probe
  // (probe_id == 2) can be acquired.
  state.set_is_zonal(std::nullopt);
  state.set_is_zonal(true);
  auto second_probe = state.AcquireProbe();
  ASSERT_EQ(second_probe.action, ProbeAction::kIsProbe);

  // While `second_probe` is active, a third caller must wait.
  auto waiter = state.AcquireProbe();
  ASSERT_EQ(waiter.action, ProbeAction::kWait);
  EXPECT_FALSE(waiter.wait_future.ready());

  // Finishing the stale `first_probe` with `mark_done = true` must be ignored
  // and must not mark `second_probe` as done or wake `waiter`.
  first_probe.lease.Finish(/*mark_done=*/true);
  EXPECT_FALSE(waiter.wait_future.ready());

  // Finishing the active `second_probe` wakes `waiter` and transitions to
  // `kZonalReady`.
  second_probe.lease.Finish(/*mark_done=*/true);
  EXPECT_TRUE(waiter.wait_future.ready());
  auto after_done = state.AcquireProbe();
  EXPECT_EQ(after_done.action, ProbeAction::kKnown);
  EXPECT_FALSE(static_cast<bool>(after_done.lease));
}

TEST(ZonalBucketStateTest, WriteModeAbslStringify) {
  EXPECT_EQ(absl::StrFormat("%v", GcsGrpcWriteMode::kStandard), "standard");
  EXPECT_EQ(absl::StrFormat("%v", GcsGrpcWriteMode::kZonal), "zonal");
  EXPECT_EQ(absl::StrFormat("%v", GcsGrpcWriteMode::kNonAtomic), "non_atomic");
}

}  // namespace
