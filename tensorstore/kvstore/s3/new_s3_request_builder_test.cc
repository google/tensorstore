#include <gtest/gtest.h>

#include "tensorstore/kvstore/s3_sdk/s3_context.h"
#include "tensorstore/kvstore/s3/new_s3_request_builder.h"


using ::tensorstore::internal_kvstore_s3::NewS3RequestBuilder;

namespace {

TEST(NewS3RequestBuilderTest, Basic) {
  auto ctx = tensorstore::internal_kvstore_s3::GetAwsContext();
  auto builder = NewS3RequestBuilder("get", "http://bucket")
                  .AddBody(absl::Cord{"foobar"})
                  .AddHeader("foo: bar")
                  .AddQueryParameter("qux", "baz");

  auto req = builder.BuildRequest(*ctx);
  EXPECT_TRUE(builder.request_.HasAuthorization());

  ABSL_LOG(INFO) << req;
}

}  // namespace
