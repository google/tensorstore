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

#include "tensorstore/kvstore/s3/s3_metadata.h"

#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorstore/internal/http/http_header.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_builder.h"
#include "tinyxml2.h"

using ::tensorstore::internal_http::HeaderMap;
using ::tensorstore::internal_http::HttpResponse;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {
static constexpr char kEtag[] = "etag";

bool IsRetryableAwsStatusCode(int32_t status_code) {
  switch (status_code) {
    case 408:  // REQUEST_TIMEOUT:
    case 419:  // AUTHENTICATION_TIMEOUT:
    case 429:  // TOO_MANY_REQUESTS:
    case 440:  // LOGIN_TIMEOUT:
    case 500:  // INTERNAL_SERVER_ERROR:
    case 502:  // BAD_GATEWAY:
    case 503:  // SERVICE_UNAVAILABLE:
    case 504:  // GATEWAY_TIMEOUT:
    case 509:  // BANDWIDTH_LIMIT_EXCEEDED:
    case 598:  // NETWORK_READ_TIMEOUT:
    case 599:  // NETWORK_CONNECT_TIMEOUT:
      return true;
    default:
      return false;
  }
}

bool IsRetryableAwsMessageCode(std::string_view code) {
  static const absl::NoDestructor<absl::flat_hash_set<std::string_view>>
      kRetryableMessages(absl::flat_hash_set<std::string_view>({
          "InternalFailureException",
          "InternalFailure",
          "InternalServerError",
          "InternalError",
          "RequestExpiredException",
          "RequestExpired",
          "ServiceUnavailableException",
          "ServiceUnavailableError",
          "ServiceUnavailable",
          "RequestThrottledException",
          "RequestThrottled",
          "ThrottlingException",
          "ThrottledException",
          "Throttling",
          "SlowDownException",
          "SlowDown",
          "RequestTimeTooSkewedException",
          "RequestTimeTooSkewed",
          "RequestTimeoutException",
          "RequestTimeout",
      }));
  return kRetryableMessages->contains(code);
}

// Extracts the disambiguated error type from an AWS error header/code value.
// Follows the smithy.io syntax rules, allowing for either ':' or '#' as a
// separator.
// https://smithy.io/2.0/aws/protocols/aws-json-1_1-protocol.html#operation-error-serialization
std::string_view ExtractAwsErrorType(std::string_view error_type) {
  if (auto pos = error_type.find(':'); pos != std::string_view::npos) {
    error_type = error_type.substr(0, pos);
  }
  if (auto pos = error_type.find('#'); pos != std::string_view::npos) {
    error_type = error_type.substr(pos + 1);
  }
  return error_type;
}

}  // namespace

std::optional<int64_t> GetNodeInt(tinyxml2::XMLNode* node) {
  if (!node) {
    return std::nullopt;
  }

  int64_t result;
  if (absl::SimpleAtoi(GetNodeText(node), &result)) {
    return result;
  }
  return std::nullopt;
}

std::optional<absl::Time> GetNodeTimestamp(tinyxml2::XMLNode* node) {
  if (!node) {
    return std::nullopt;
  }

  absl::Time result;
  if (absl::ParseTime(absl::RFC3339_full, GetNodeText(node),
                      absl::UTCTimeZone(), &result, nullptr)) {
    return result;
  }
  return std::nullopt;
}

std::string GetNodeText(tinyxml2::XMLNode* node) {
  if (!node) {
    return "";
  }

  std::string result;
  for (auto* child = node->FirstChild(); child != nullptr;
       child = child->NextSibling()) {
    if (auto* text = child->ToText()) {
      result.append(text->Value());
    }
  }
  return result;
}

Result<StorageGeneration> StorageGenerationFromHeaders(
    const HeaderMap& headers) {
  if (auto it = headers.find(kEtag); it != headers.end()) {
    return StorageGeneration::FromString(it->second);
  }
  return absl::NotFoundError("etag not found in response headers");
}

absl::Status AwsHttpResponseToStatus(const HttpResponse& response,
                                     bool& retryable, SourceLocation loc) {
  auto absl_status_code = internal_http::HttpResponseCodeToStatusCode(response);
  if (absl_status_code == absl::StatusCode::kOk) {
    return absl::OkStatus();
  }

  std::string error_type;
  if (auto error_header = response.headers.find("x-amzn-errortype");
      error_header != response.headers.end()) {
    error_type = ExtractAwsErrorType(error_header->second);
  }

  absl::Cord request_id;
  if (auto request_id_header = response.headers.find("x-amzn-requestid");
      request_id_header != response.headers.end()) {
    request_id = request_id_header->second;
  }

  std::string message;

  // Parse the XML response to get the error message.
  // https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingRESTError.html
  // https://docs.aws.amazon.com/AmazonS3/latest/API/ErrorResponses.html
  auto payload = response.payload;
  auto payload_str = payload.Flatten();
  [&]() {
    if (payload.empty()) return;

    tinyxml2::XMLDocument xmlDocument;
    if (int xmlcode = xmlDocument.Parse(payload_str.data(), payload_str.size());
        xmlcode != tinyxml2::XML_SUCCESS) {
      return;
    }
    auto* root_node = xmlDocument.FirstChildElement("Error");
    if (root_node == nullptr) return;

    if (error_type.empty()) {
      error_type = ExtractAwsErrorType(
          GetNodeText(root_node->FirstChildElement("Code")));
    }
    if (request_id.empty()) {
      request_id = GetNodeText(root_node->FirstChildElement("RequestId"));
    }
    message = GetNodeText(root_node->FirstChildElement("Message"));
  }();

  retryable = IsRetryableAwsStatusCode(response.status_code) ||
              (!error_type.empty() && IsRetryableAwsMessageCode(error_type));

  if (error_type.empty()) {
    error_type = "Unknown";
  }

  StatusBuilder builder(absl_status_code, loc);
  if (!payload_str.empty()) {
    builder.SetPayload(
        "http_response_body",
        payload.Subcord(0,
                        payload_str.size() < 256 ? payload_str.size() : 256));
  }
  if (!request_id.empty()) {
    builder.SetPayload("x-amzn-requestid", request_id);
  }
  return builder
      .SetPayload("http_response_code",
                  absl::StrFormat("%d", response.status_code))
      .Format("%s%s%s", error_type, (message.empty() ? "" : ": "), message);
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
