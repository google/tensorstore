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

#include "tensorstore/kvstore/s3/credentials/test_utils.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_response.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

absl::flat_hash_map<std::string, internal_http::HttpResponse>
DefaultEC2MetadataFlow(const std::string& endpoint,
                       const std::string& api_token,
                       const std::string& access_key,
                       const std::string& secret_key,
                       const std::string& session_token,
                       const absl::Time& expires_at) {
  return absl::flat_hash_map<std::string, internal_http::HttpResponse>{
      {absl::StrFormat("POST %s/latest/api/token", endpoint),
       internal_http::HttpResponse{200, absl::Cord{api_token}}},
      {absl::StrFormat("GET %s/latest/meta-data/iam/", endpoint),
       internal_http::HttpResponse{
           200, absl::Cord{"info"}, {{"x-aws-ec2-metadata-token", api_token}}}},
      {absl::StrFormat("GET %s/latest/meta-data/iam/security-credentials/",
                       endpoint),
       internal_http::HttpResponse{200,
                                   absl::Cord{"mock-iam-role"},
                                   {{"x-aws-ec2-metadata-token", api_token}}}},
      {absl::StrFormat(
           "GET %s/latest/meta-data/iam/security-credentials/mock-iam-role",
           endpoint),
       internal_http::HttpResponse{
           200,
           absl::Cord(absl::StrFormat(
               R"({
                      "Code": "Success",
                      "AccessKeyId": "%s",
                      "SecretAccessKey": "%s",
                      "Token": "%s",
                      "Expiration": "%s"
                  })",
               access_key, secret_key, session_token,
               absl::FormatTime("%Y-%m-%d%ET%H:%M:%E*S%Ez", expires_at,
                                absl::UTCTimeZone()))),
           {{"x-aws-ec2-metadata-token", api_token}}}}};
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
