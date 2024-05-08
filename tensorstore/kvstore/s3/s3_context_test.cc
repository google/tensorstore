
#include <iostream>
#include <memory>


#include "absl/log/absl_log.h"

#include <aws/core/Aws.h>
#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/utils/threading/Executor.h>

#include <aws/s3/S3Endpoint.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/HeadBucketRequest.h>

#include "tensorstore/kvstore/s3/s3_context.h"
#include "tensorstore/internal/thread/thread_pool.h"
#include "tensorstore/util/executor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

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

  // sanity check basic credential retrieval
  //auto creds = ctx->cred_provider_->GetAWSCredentials();

  ctx.reset();
  ctx2.reset();

  EXPECT_EQ(wp.use_count(), 0);
  EXPECT_EQ(wp.lock(), nullptr);

  ctx = GetAwsContext();
}

TEST(S3ContextTest, Endpoint) {
  EXPECT_EQ(Aws::S3::S3Endpoint::ForRegion("us-east-2", false, false), "s3.us-east-2.amazonaws.com");
}

TEST(S3ContextTest, Client) {
  class AwsExecutorAdapter : public Aws::Utils::Threading::Executor {
  public:
    AwsExecutorAdapter(): executor_(::tensorstore::internal::DetachedThreadPool(4)) {}
  protected:
    bool SubmitToThread(std::function<void()> && fn) override {
      ::tensorstore::WithExecutor(executor_, std::move(fn));
      return true;
    }

  private:
    ::tensorstore::Executor executor_;
  };

  auto ctx = GetAwsContext();
  auto cfg = Aws::Client::ClientConfiguration();
  //cfg.executor = Aws::MakeShared<AwsExecutorAdapter>(kAWSTag);
  cfg.executor->Submit([msg = "Submission Works"] { ABSL_LOG(INFO) << msg; });
  auto client = Aws::S3::S3Client(cfg);
  auto head_bucket = Aws::S3::Model::HeadBucketRequest().WithBucket("ratt-public-data");
  auto outcome = client.HeadBucket(head_bucket);
  if(!outcome.IsSuccess()) {
    auto & err = outcome.GetError();
    std::cerr << "Error: " << err.GetExceptionName() << ": " << err.GetMessage() << std::endl;
  } else {
    std::cout << "Success" << std::endl;
  }
}

} // namespace {