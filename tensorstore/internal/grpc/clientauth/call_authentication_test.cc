// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/grpc/clientauth/call_authentication.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "grpcpp/client_context.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal_grpc::CreateAccessTokenAuthenticationStrategy;
using ::tensorstore::internal_grpc::CreateServiceAccountAuthenticationStrategy;
using ::tensorstore::internal_grpc::GrpcCallCredentialsAuthentication;

using ::testing::IsNull;
using ::testing::NotNull;

namespace {

TEST(GrpcAccessTokenAuthentication, Simple) {
  grpc::SslCredentialsOptions ssl_options;
  GrpcCallCredentialsAuthentication auth(
      grpc::SslCredentials(ssl_options),
      grpc::AccessTokenCredentials("token1"));

  grpc::ChannelArguments args;
  auto creds = auth.GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), NotNull());

  for (auto _ : {1, 2, 3}) {
    auto context = std::make_shared<grpc::ClientContext>();
    EXPECT_THAT(context->credentials(), IsNull());
    auto configured = auth.ConfigureContext(context).result();
    EXPECT_THAT(configured.status(), tensorstore::IsOk());
    EXPECT_THAT(configured.value()->credentials(), NotNull());
  }
}

TEST(GrpcAccessTokenAuthentication, AccessToken) {
  auto auth = CreateAccessTokenAuthenticationStrategy("token1", {});

  grpc::ChannelArguments args;
  auto creds = auth->GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), NotNull());

  for (auto _ : {1, 2, 3}) {
    auto context = std::make_shared<grpc::ClientContext>();
    EXPECT_THAT(context->credentials(), IsNull());
    auto configured = auth->ConfigureContext(context).result();
    EXPECT_THAT(configured.status(), tensorstore::IsOk());
    EXPECT_THAT(configured.value()->credentials(), NotNull());
  }
}

TEST(GrpcAccessTokenAuthentication, ServiceAccount) {
  // The private key in this JSON object was invalidated (deleted) long time
  // ago. The service account and key it was created for were deleted too, and
  // do not match the attributes listed here.
  constexpr char kJsonObject[] = R"""({
      "type": "service_account",
      "project_id": "foo-project",
      "private_key_id": "a1a111aa1111a11a11a11aa111a111a1a1111111",
      "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCltiF2oP3KJJ+S\ntTc1McylY+TuAi3AdohX7mmqIjd8a3eBYDHs7FlnUrFC4CRijCr0rUqYfg2pmk4a\n6TaKbQRAhWDJ7XD931g7EBvCtd8+JQBNWVKnP9ByJUaO0hWVniM50KTsWtyX3up/\nfS0W2R8Cyx4yvasE8QHH8gnNGtr94iiORDC7De2BwHi/iU8FxMVJAIyDLNfyk0hN\neheYKfIDBgJV2v6VaCOGWaZyEuD0FJ6wFeLybFBwibrLIBE5Y/StCrZoVZ5LocFP\nT4o8kT7bU6yonudSCyNMedYmqHj/iF8B2UN1WrYx8zvoDqZk0nxIglmEYKn/6U7U\ngyETGcW9AgMBAAECggEAC231vmkpwA7JG9UYbviVmSW79UecsLzsOAZnbtbn1VLT\nPg7sup7tprD/LXHoyIxK7S/jqINvPU65iuUhgCg3Rhz8+UiBhd0pCH/arlIdiPuD\n2xHpX8RIxAq6pGCsoPJ0kwkHSw8UTnxPV8ZCPSRyHV71oQHQgSl/WjNhRi6PQroB\nSqc/pS1m09cTwyKQIopBBVayRzmI2BtBxyhQp9I8t5b7PYkEZDQlbdq0j5Xipoov\n9EW0+Zvkh1FGNig8IJ9Wp+SZi3rd7KLpkyKPY7BK/g0nXBkDxn019cET0SdJOHQG\nDiHiv4yTRsDCHZhtEbAMKZEpku4WxtQ+JjR31l8ueQKBgQDkO2oC8gi6vQDcx/CX\nZ23x2ZUyar6i0BQ8eJFAEN+IiUapEeCVazuxJSt4RjYfwSa/p117jdZGEWD0GxMC\n+iAXlc5LlrrWs4MWUc0AHTgXna28/vii3ltcsI0AjWMqaybhBTTNbMFa2/fV2OX2\nUimuFyBWbzVc3Zb9KAG4Y7OmJQKBgQC5324IjXPq5oH8UWZTdJPuO2cgRsvKmR/r\n9zl4loRjkS7FiOMfzAgUiXfH9XCnvwXMqJpuMw2PEUjUT+OyWjJONEK4qGFJkbN5\n3ykc7p5V7iPPc7Zxj4mFvJ1xjkcj+i5LY8Me+gL5mGIrJ2j8hbuv7f+PWIauyjnp\nNx/0GVFRuQKBgGNT4D1L7LSokPmFIpYh811wHliE0Fa3TDdNGZnSPhaD9/aYyy78\nLkxYKuT7WY7UVvLN+gdNoVV5NsLGDa4cAV+CWPfYr5PFKGXMT/Wewcy1WOmJ5des\nAgMC6zq0TdYmMBN6WpKUpEnQtbmh3eMnuvADLJWxbH3wCkg+4xDGg2bpAoGAYRNk\nMGtQQzqoYNNSkfus1xuHPMA8508Z8O9pwKU795R3zQs1NAInpjI1sOVrNPD7Ymwc\nW7mmNzZbxycCUL/yzg1VW4P1a6sBBYGbw1SMtWxun4ZbnuvMc2CTCh+43/1l+FHe\nMmt46kq/2rH2jwx5feTbOE6P6PINVNRJh/9BDWECgYEAsCWcH9D3cI/QDeLG1ao7\nrE2NcknP8N783edM07Z/zxWsIsXhBPY3gjHVz2LDl+QHgPWhGML62M0ja/6SsJW3\nYvLLIc82V7eqcVJTZtaFkuht68qu/Jn1ezbzJMJ4YXDYo1+KFi+2CAGR06QILb+I\nlUtj+/nH3HDQjM4ltYfTPUg=\n-----END PRIVATE KEY-----\n",
      "client_email": "foo-email@foo-project.iam.gserviceaccount.com",
      "client_id": "100000000000000000001",
      "auth_uri": "https://accounts.google.com/o/oauth2/auth",
      "token_uri": "https://oauth2.googleapis.com/token",
      "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
      "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/foo-email%40foo-project.iam.gserviceaccount.com"
})""";

  auto auth = CreateServiceAccountAuthenticationStrategy(kJsonObject, {});

  grpc::ChannelArguments args;
  auto creds = auth->GetChannelCredentials("localhost:1", args);
  EXPECT_THAT(creds.get(), NotNull());

  for (auto _ : {1, 2, 3}) {
    auto context = std::make_shared<grpc::ClientContext>();
    EXPECT_THAT(context->credentials(), IsNull());
    auto configured = auth->ConfigureContext(context).result();
    EXPECT_THAT(configured.status(), tensorstore::IsOk());
    EXPECT_THAT(configured.value()->credentials(), NotNull());
  }
}

}  // namespace
