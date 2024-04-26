
#include <memory>

#include <aws/core/Aws.h>
#include <aws/core/auth/AWSAuthSigner.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/http/HttpRequest.h>
#include <aws/core/utils/threading/Executor.h>

#include <aws/s3/S3Endpoint.h>
#include <aws/s3/S3Client.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tensorstore/kvstore/s3/s3_context.h"

using ::tensorstore::internal_kvstore_s3::GetAwsContext;
using ::tensorstore::internal_kvstore_s3::AwsContext;

namespace {

static constexpr char kAWSTag[] = "AWS";

TEST(S3ContextTest, Basic) {
  auto ctx = GetAwsContext();
  EXPECT_EQ(ctx.use_count(), 1);
  std::weak_ptr<AwsContext> wp = ctx;
  EXPECT_EQ(wp.use_count(), 1);

  auto ctx2 = GetAwsContext();
  EXPECT_EQ(ctx, ctx2);
  EXPECT_EQ(wp.use_count(), 2);

  ctx.reset();
  ctx2.reset();

  EXPECT_EQ(wp.use_count(), 0);
  EXPECT_EQ(wp.lock(), nullptr);

  ctx = GetAwsContext();
}

TEST(S3ContextTest, AWS4Signing) {
  auto ctx = GetAwsContext();
  auto signer = Aws::Client::AWSAuthV4Signer(ctx->cred_provider_, "s3", "us-east-2");
  //auto req = Aws::Http::HttpRequest();
}

TEST(S3ContextTest, Endpoint) {
  EXPECT_EQ(Aws::S3::S3Endpoint::ForRegion("us-east-2", false, false), "s3.us-east-2.amazonaws.com");
}

TEST(S3ContextTest, Client) {
  class OffloadExecutor : public Aws::Utils::Threading::Executor {
  protected:
    bool SubmitToThread(std::function<void()> && fn) {
      fn();
      return true;
    }
  };

  auto ctx = GetAwsContext();
  auto cfg = Aws::Client::ClientConfiguration();
  cfg.executor = Aws::MakeShared<OffloadExecutor>(kAWSTag);
  auto client = Aws::S3::S3Client();
}

} // namespace {