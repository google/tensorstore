// Copyright 2023 The TensorStore Authors
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

#include <stddef.h>

#include <deque>
#include <fstream>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <aws/auth/credentials.h>
#include <aws/common/error.h>
#include <aws/common/zero.h>
#include "tensorstore/internal/aws/aws_api.h"
#include "tensorstore/internal/aws/aws_credentials.h"
#include "tensorstore/internal/aws/credentials/common.h"
#include "tensorstore/internal/aws/http_mocking.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/testing/env_saver.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::StatusIs;
using ::tensorstore::internal::JoinPath;
using ::tensorstore::internal::SetEnv;
using ::tensorstore::internal::UnsetEnv;
using ::tensorstore::internal_aws::AwsCredentials;
using ::tensorstore::internal_aws::AwsCredentialsProvider;
using ::tensorstore::internal_aws::DisableAwsHttpMocking;
using ::tensorstore::internal_aws::EnableAwsHttpMocking;
using ::tensorstore::internal_aws::GetAwsAllocator;
using ::tensorstore::internal_aws::GetAwsCredentials;
using ::tensorstore::internal_aws::MakeAnonymous;
using ::tensorstore::internal_aws::MakeCache;
using ::tensorstore::internal_aws::MakeDefault;
using ::tensorstore::internal_aws::MakeDefaultWithAnonymous;
using ::tensorstore::internal_aws::MakeEnvironment;
using ::tensorstore::internal_aws::MakeFirstResolved;
using ::tensorstore::internal_aws::MakeImds;
using ::tensorstore::internal_testing::EnvSaver;
using ::testing::HasSubstr;

MATCHER(IsAnonymousCredentials, "") {
  if (arg.IsAnonymous()) return true;
  *result_listener << "has access_key_id=" << arg.GetAccessKeyId();
  return false;
}

MATCHER_P(HasAccessKeyId, expected, "") {
  if (!arg.IsAnonymous() && arg.GetAccessKeyId() == expected) return true;
  if (arg.IsAnonymous()) {
    *result_listener << "is anonymous";
  } else {
    *result_listener << "has access_key_id=" << arg.GetAccessKeyId();
  }
  return false;
}

// Controllable async/sync credentials provider for deterministic testing of
// coalescing, mid-flight shutdown, re-entrancy, and synchronous `AWS_OP_ERR`.
class ManualCredentialsProvider
    : public tensorstore::internal::AtomicReferenceCount<
          ManualCredentialsProvider> {
 public:
  AwsCredentialsProvider MakeProvider() {
    tensorstore::internal::IntrusivePtr<ManualCredentialsProvider> self(this);
    aws_credentials_provider_delegate_options options;
    AWS_ZERO_STRUCT(options);
    options.get_credentials = &GetCredentials;
    options.delegate_user_data = self.get();
    options.shutdown_options.shutdown_callback = &OnShutdown;
    options.shutdown_options.shutdown_user_data = self.get();
    aws_credentials_provider* raw =
        aws_credentials_provider_new_delegate(GetAwsAllocator(), &options);
    if (!raw) return nullptr;
    self.release();
    return AwsCredentialsProvider(raw, tensorstore::internal::adopt_object_ref);
  }

  void SetFailSynchronously(bool fail, int err = AWS_ERROR_INVALID_STATE) {
    absl::MutexLock lock(mutex_);
    fail_synchronously_ = fail;
    sync_error_code_ = err;
  }

  int call_count() const {
    absl::MutexLock lock(mutex_);
    return call_count_;
  }

  size_t pending_count() const {
    absl::MutexLock lock(mutex_);
    return pending_.size();
  }

  void CompleteNext(const AwsCredentials& creds) {
    Pending cb = PopNext();
    cb.callback(creds.get(), AWS_ERROR_SUCCESS, cb.user_data);
  }

  void FailNext(int err = AWS_ERROR_INVALID_STATE) {
    Pending cb = PopNext();
    cb.callback(nullptr, err, cb.user_data);
  }

 private:
  struct Pending {
    aws_on_get_credentials_callback_fn* callback;
    void* user_data;
  };

  Pending PopNext() {
    absl::MutexLock lock(mutex_);
    Pending cb = pending_.front();
    pending_.pop_front();
    return cb;
  }

  static void OnShutdown(void* user_data) {
    tensorstore::internal::IntrusivePtr<ManualCredentialsProvider>(
        static_cast<ManualCredentialsProvider*>(user_data),
        tensorstore::internal::adopt_object_ref);
  }

  static int GetCredentials(void* delegate_user_data,
                            aws_on_get_credentials_callback_fn* callback,
                            void* callback_user_data) {
    auto* self = static_cast<ManualCredentialsProvider*>(delegate_user_data);
    int sync_err = 0;
    {
      absl::MutexLock lock(self->mutex_);
      ++self->call_count_;
      if (self->fail_synchronously_) {
        sync_err = self->sync_error_code_;
      } else {
        self->pending_.push_back({callback, callback_user_data});
        return AWS_OP_SUCCESS;
      }
    }
    return aws_raise_error(sync_err);
  }

  mutable absl::Mutex mutex_;
  int call_count_ ABSL_GUARDED_BY(mutex_) = 0;
  bool fail_synchronously_ ABSL_GUARDED_BY(mutex_) = false;
  int sync_error_code_ ABSL_GUARDED_BY(mutex_) = AWS_ERROR_INVALID_STATE;
  std::deque<Pending> pending_ ABSL_GUARDED_BY(mutex_);
};

static constexpr char kAccessKeyId[] = "ASIA1234567890";
static constexpr char kSecretKey[] = "1234567890abcdef";
static constexpr char kSessionToken[] = "abcdef123456790";

class TestData
    : public tensorstore::internal_testing::ScopedTemporaryDirectory {
 public:
  std::string WriteCredentialsFile() {
    auto p = JoinPath(path(), "aws_credentials");
    std::ofstream ofs(p);
    ofs << "[default]\n"
           "aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n"
           "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
           "aws_session_token=abcdef1234567890 \n"
           "\n"
           "[alice]\n"
           "aws_access_key_id = AKIAIOSFODNN6EXAMPLE\n"
           "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCZEXAMPLEKEY\n"
           "\n";
    ofs.close();
    return p;
  }
  std::string WriteConfigFile() {
    auto p = JoinPath(path(), "aws_config");
    std::ofstream ofs(p);
    ofs << "[default]\n"
           "region=us-west-2\n"
           "output=json\n"
           "\n";
    ofs.close();
    return p;
  }
};

class DefaultCredentialProviderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    expiry = absl::Now() + absl::Seconds(200);
    EnableAwsHttpMocking({});

    for (const char* var : {
             "AWS_ACCESS_KEY_ID",
             "AWS_SECRET_ACCESS_KEY",
             "AWS_SESSION_TOKEN",
             "AWS_CONFIG_FILE",
             "AWS_SHARED_CREDENTIALS_FILE",
             "AWS_PROFILE",
             "AWS_CONTAINER_CREDENTIALS_FULL_URI",
             "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
         }) {
      UnsetEnv(var);
    }
  }
  void TearDown() override { DisableAwsHttpMocking(); }

  EnvSaver env_saver_;
  absl::Time expiry;
};

TEST_F(DefaultCredentialProviderTest, Anonymous) {
  AwsCredentialsProvider provider = MakeAnonymous();
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());
  EXPECT_TRUE(credentials_future.result()->IsAnonymous());
}

TEST_F(DefaultCredentialProviderTest, DefaultWithoutFallback) {
  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("aws-c-")));
}

TEST_F(DefaultCredentialProviderTest, DefaultWithFallback) {
  AwsCredentialsProvider provider = MakeDefaultWithAnonymous({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());
  EXPECT_TRUE(credentials_future.result()->IsAnonymous());
}

TEST_F(DefaultCredentialProviderTest, FromEnvironment) {
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);
  SetEnv("AWS_SESSION_TOKEN", kSessionToken);

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), kAccessKeyId);
  EXPECT_EQ(credentials.GetSecretAccessKey(), kSecretKey);
  EXPECT_EQ(credentials.GetSessionToken(), kSessionToken);
}

TEST_F(DefaultCredentialProviderTest, FromProfile) {
  TestData test_data;
  SetEnv("AWS_SHARED_CREDENTIALS_FILE",
         test_data.WriteCredentialsFile().c_str());
  SetEnv("AWS_CONFIG_FILE", test_data.WriteConfigFile().c_str());

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = GetAwsCredentials(provider.get());
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(credentials.GetSecretAccessKey(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}

TEST_F(DefaultCredentialProviderTest, FirstResolvedLatchesFallback) {
  AwsCredentialsProvider provider =
      MakeFirstResolved({MakeEnvironment(), MakeAnonymous()});
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              IsOkAndHolds(IsAnonymousCredentials()));

  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              IsOkAndHolds(IsAnonymousCredentials()));
}

TEST_F(DefaultCredentialProviderTest, FirstResolvedLatchesMiddleProvider) {
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);

  // `MakeImds()` fails under empty HTTP mocking, so `MakeEnvironment()` (the
  // middle provider) resolves first and is latched over `MakeAnonymous()`.
  AwsCredentialsProvider provider =
      MakeFirstResolved({MakeImds(), MakeEnvironment(), MakeAnonymous()});
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));

  // Subsequent failures of the latched provider surface an error rather than
  // advancing to `MakeAnonymous()`, and recover once it succeeds again.
  UnsetEnv("AWS_ACCESS_KEY_ID");
  UnsetEnv("AWS_SECRET_ACCESS_KEY");
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              StatusIs(absl::StatusCode::kInternal));

  SetEnv("AWS_ACCESS_KEY_ID", "AKIDROTATED");
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              IsOkAndHolds(HasAccessKeyId("AKIDROTATED")));
}

TEST_F(DefaultCredentialProviderTest, FirstResolvedAllFailRemainsUnlatched) {
  AwsCredentialsProvider provider =
      MakeFirstResolved({MakeImds(), MakeEnvironment()});
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              StatusIs(absl::StatusCode::kInternal));

  // Because no provider resolved on the first attempt, a later call retries the
  // sequence and latches once `MakeEnvironment()` succeeds.
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);
  EXPECT_THAT(GetAwsCredentials(provider.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));
}

TEST_F(DefaultCredentialProviderTest, MakeCacheZeroAndInfiniteDuration) {
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);

  AwsCredentialsProvider uncached =
      MakeCache(MakeEnvironment(), absl::ZeroDuration());
  AwsCredentialsProvider cached = MakeCache(MakeEnvironment());
  EXPECT_THAT(GetAwsCredentials(uncached.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));
  EXPECT_THAT(GetAwsCredentials(cached.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));

  SetEnv("AWS_ACCESS_KEY_ID", "AKIDROTATED");
  EXPECT_THAT(GetAwsCredentials(uncached.get()),
              IsOkAndHolds(HasAccessKeyId("AKIDROTATED")));
  EXPECT_THAT(GetAwsCredentials(cached.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));
}

TEST_F(DefaultCredentialProviderTest, MakeCacheFiniteDurationRefreshes) {
  SetEnv("AWS_ACCESS_KEY_ID", kAccessKeyId);
  SetEnv("AWS_SECRET_ACCESS_KEY", kSecretKey);

  AwsCredentialsProvider cached =
      MakeCache(MakeEnvironment(), absl::Milliseconds(1));
  EXPECT_THAT(GetAwsCredentials(cached.get()),
              IsOkAndHolds(HasAccessKeyId(kAccessKeyId)));

  SetEnv("AWS_ACCESS_KEY_ID", "AKIDROTATED");
  absl::SleepFor(absl::Milliseconds(10));
  EXPECT_THAT(GetAwsCredentials(cached.get()),
              IsOkAndHolds(HasAccessKeyId("AKIDROTATED")));
}

TEST_F(DefaultCredentialProviderTest, FirstResolvedMakeEdgeCases) {
  EXPECT_EQ(MakeFirstResolved({}).get(), nullptr);
  EXPECT_EQ(MakeFirstResolved({nullptr, nullptr}).get(), nullptr);

  AwsCredentialsProvider single = MakeAnonymous();
  EXPECT_EQ(MakeFirstResolved({nullptr, single, nullptr}).get(), single.get());
}

TEST_F(DefaultCredentialProviderTest, FirstResolvedCoalescesConcurrentCallers) {
  auto c0 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  auto c1 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  AwsCredentialsProvider provider =
      MakeFirstResolved({c0->MakeProvider(), c1->MakeProvider()});

  auto f1 = GetAwsCredentials(provider.get());
  auto f2 = GetAwsCredentials(provider.get());
  EXPECT_FALSE(f1.ready());
  EXPECT_FALSE(f2.ready());
  EXPECT_EQ(c0->call_count(), 1);
  EXPECT_EQ(c1->call_count(), 0);

  // Failing c0 advances the single active resolution to c1 without duplicating.
  c0->FailNext();
  auto f3 = GetAwsCredentials(provider.get());
  EXPECT_EQ(c0->call_count(), 1);
  EXPECT_EQ(c1->call_count(), 1);
  EXPECT_FALSE(f3.ready());

  c1->CompleteNext(AwsCredentials::Make("AKID_COALESCED", kSecretKey, ""));
  EXPECT_THAT(f1, IsOkAndHolds(HasAccessKeyId("AKID_COALESCED")));
  EXPECT_THAT(f2, IsOkAndHolds(HasAccessKeyId("AKID_COALESCED")));
  EXPECT_THAT(f3, IsOkAndHolds(HasAccessKeyId("AKID_COALESCED")));

  // Subsequent queries delegate straight to latched c1.
  auto f4 = GetAwsCredentials(provider.get());
  EXPECT_EQ(c0->call_count(), 1);
  EXPECT_EQ(c1->call_count(), 2);
  c1->CompleteNext(AwsCredentials::Make("AKID_NEXT", kSecretKey, ""));
  EXPECT_THAT(f4, IsOkAndHolds(HasAccessKeyId("AKID_NEXT")));
}

TEST_F(DefaultCredentialProviderTest,
       FirstResolvedSurvivesProviderReleaseMidResolution) {
  auto c0 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  auto c1 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  AwsCredentialsProvider outer =
      MakeFirstResolved({c0->MakeProvider(), c1->MakeProvider()});

  struct RawCapture {
    bool called = false;
    AwsCredentials creds;
    int error_code = -1;
    static void Callback(aws_credentials* c, int err, void* user_data) {
      auto* self = static_cast<RawCapture*>(user_data);
      self->called = true;
      self->creds = AwsCredentials(c);
      self->error_code = err;
    }
  } capture;

  // Query directly without `GetAwsCredentials`'s extra provider ref so
  // `outer.reset()` triggers `FirstResolvedState::OnShutdown` mid-resolution.
  ASSERT_EQ(aws_credentials_provider_get_credentials(
                outer.get(), &RawCapture::Callback, &capture),
            AWS_OP_SUCCESS);
  ASSERT_EQ(c0->pending_count(), 1u);

  outer.reset();
  EXPECT_FALSE(capture.called);

  c0->FailNext();
  ASSERT_EQ(c1->pending_count(), 1u);
  c1->CompleteNext(AwsCredentials::Make("AKID_SURVIVED", kSecretKey, ""));

  EXPECT_TRUE(capture.called);
  EXPECT_EQ(capture.error_code, AWS_ERROR_SUCCESS);
  EXPECT_THAT(capture.creds, HasAccessKeyId("AKID_SURVIVED"));
}

TEST_F(DefaultCredentialProviderTest,
       FirstResolvedHandlesSyncErrorAndReentrantRetry) {
  auto c0 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  auto c1 =
      tensorstore::internal::MakeIntrusivePtr<ManualCredentialsProvider>();
  c0->SetFailSynchronously(true, AWS_ERROR_INVALID_ARGUMENT);
  c1->SetFailSynchronously(true, AWS_ERROR_INVALID_STATE);

  AwsCredentialsProvider provider =
      MakeFirstResolved({c0->MakeProvider(), c1->MakeProvider()});

  // Test re-entrant retry directly from inside the all-failed callback, and
  // verify the last candidate's specific error code (not the fallback) arrives.
  struct ReentrantCapture {
    aws_credentials_provider* provider = nullptr;
    ManualCredentialsProvider* c1 = nullptr;
    bool retry_dispatched = false;
    static void Callback(aws_credentials* c, int err, void* user_data) {
      auto* self = static_cast<ReentrantCapture*>(user_data);
      EXPECT_EQ(c, nullptr);
      EXPECT_EQ(err, AWS_ERROR_INVALID_STATE);
      self->c1->SetFailSynchronously(false);
      EXPECT_EQ(aws_credentials_provider_get_credentials(self->provider,
                                                         &RetryCallback, self),
                AWS_OP_SUCCESS);
      self->retry_dispatched = true;
    }
    AwsCredentials retry_creds;
    static void RetryCallback(aws_credentials* c, int err, void* user_data) {
      auto* self = static_cast<ReentrantCapture*>(user_data);
      EXPECT_EQ(err, AWS_ERROR_SUCCESS);
      self->retry_creds = AwsCredentials(c);
    }
  } reentrant{provider.get(), c1.get()};

  ASSERT_EQ(aws_credentials_provider_get_credentials(
                provider.get(), &ReentrantCapture::Callback, &reentrant),
            AWS_OP_SUCCESS);
  EXPECT_TRUE(reentrant.retry_dispatched);
  ASSERT_EQ(c1->pending_count(), 1u);
  c1->CompleteNext(AwsCredentials::Make("AKID_REENTRANT", kSecretKey, ""));
  EXPECT_THAT(reentrant.retry_creds, HasAccessKeyId("AKID_REENTRANT"));

  // Also verify that synchronous `AWS_OP_ERR` on the latched fast path invokes
  // the callback with the real AWS error rather than hanging `MakeCache`.
  AwsCredentialsProvider cached = MakeCache(provider, absl::Milliseconds(1));
  auto f_ok = GetAwsCredentials(cached.get());
  c1->CompleteNext(AwsCredentials::Make("AKID_CACHED", kSecretKey, ""));
  EXPECT_THAT(f_ok, IsOkAndHolds(HasAccessKeyId("AKID_CACHED")));

  absl::SleepFor(absl::Milliseconds(10));
  c1->SetFailSynchronously(true, AWS_ERROR_INVALID_STATE);
  EXPECT_THAT(GetAwsCredentials(cached.get()),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("AWS_ERROR_INVALID_STATE")));
}

#if 0
// Default credentials provider chain does not include support for mocking
// the IMDS calls.
TEST_F(DefaultCredentialProviderTest, FromImds) {
  EnableAwsHttpMocking(DefaultImdsCredentialFlow(
      kApiToken, kAccessKey, kSecretKey, kSessionToken, expiry));

  AwsCredentialsProvider provider = MakeDefault({});
  auto credentials_future = provider.GetCredentials();
  ASSERT_THAT(credentials_future, IsOk());

  auto credentials = std::move(credentials_future.result()).value();
  EXPECT_EQ(credentials.GetAccessKeyId(), "AKIAIOSFODNN7EXAMPLE");
  EXPECT_EQ(credentials.GetSecretAccessKey(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
  EXPECT_EQ(credentials.GetSessionToken(), "abcdef1234567890");
}
#endif

}  // namespace
