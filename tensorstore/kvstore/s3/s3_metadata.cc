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

#include <cassert>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "re2/re2.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tinyxml2.h"

using ::tensorstore::internal_http::HttpResponse;

namespace tensorstore {
namespace internal_kvstore_s3 {

namespace {
static constexpr char kEtag[] = "etag";
/// The 5 XML Special Character sequences
/// https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references#Standard_public_entity_sets_for_characters
static constexpr char kLt[] = "&lt;";
static constexpr char kGt[] = "&gt;";
static constexpr char kQuot[] = "&quot;";
static constexpr char kApos[] = "&apos;";
static constexpr char kAmp[] = "&amp;";

/// Unescape the 5 special XML character sequences
/// https://en.wikipedia.org/wiki/List_of_XML_and_HTML_character_entity_references#Standard_public_entity_sets_for_characters
std::string UnescapeXml(std::string_view data) {
  static LazyRE2 kSpecialXmlSymbols = {"(&gt;|&lt;|&quot;|&apos;|&amp;)"};

  std::string_view search = data;
  std::string_view symbol;
  size_t result_len = data.length();

  // Scan for xml sequences that need converting
  while (RE2::FindAndConsume(&search, *kSpecialXmlSymbols, &symbol)) {
    result_len -= symbol.length() - 1;
  }

  if (result_len == data.length()) {
    return std::string(data);
  }

  search = data;
  size_t pos = 0;
  size_t res_pos = 0;
  auto result = std::string(result_len, '0');

  while (RE2::FindAndConsume(&search, *kSpecialXmlSymbols, &symbol)) {
    size_t next = data.length() - search.length();
    // Copy any characters prior to sequence start
    for (size_t i = pos; i < next - symbol.length(); ++i, ++res_pos) {
      result[res_pos] = data[i];
    }

    // Substitute characters for sequences
    if (symbol == kGt) {
      result[res_pos++] = '>';
    } else if (symbol == kLt) {
      result[res_pos++] = '<';
    } else if (symbol == kQuot) {
      result[res_pos++] = '"';
    } else if (symbol == kApos) {
      result[res_pos++] = '`';
    } else if (symbol == kAmp) {
      result[res_pos++] = '&';
    } else {
      assert(false);
    }

    pos = next;
  }

  // Copy any remaining chars
  for (size_t i = pos; i < data.length(); ++i, ++res_pos) {
    result[res_pos] = data[i];
  }

  return result;
}

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

}  // namespace

std::optional<int64_t> GetNodeInt(tinyxml2::XMLNode* node) {
  if (!node) {
    return std::nullopt;
  }

  tinyxml2::XMLPrinter printer;
  for (auto* child = node->FirstChild(); child != nullptr;
       child = child->NextSibling()) {
    child->Accept(&printer);
  }

  int64_t result;
  if (absl::SimpleAtoi(printer.CStr(), &result)) {
    return result;
  }
  return std::nullopt;
}

std::optional<absl::Time> GetNodeTimestamp(tinyxml2::XMLNode* node) {
  if (!node) {
    return std::nullopt;
  }

  tinyxml2::XMLPrinter printer;
  for (auto* child = node->FirstChild(); child != nullptr;
       child = child->NextSibling()) {
    child->Accept(&printer);
  }
  absl::Time result;
  if (absl::ParseTime(absl::RFC3339_full, printer.CStr(), absl::UTCTimeZone(),
                      &result, nullptr)) {
    return result;
  }
  return std::nullopt;
}

std::string GetNodeText(tinyxml2::XMLNode* node) {
  if (!node) {
    return "";
  }

  tinyxml2::XMLPrinter printer;
  for (auto* child = node->FirstChild(); child != nullptr;
       child = child->NextSibling()) {
    child->Accept(&printer);
  }
  return UnescapeXml(printer.CStr());
}

Result<StorageGeneration> StorageGenerationFromHeaders(
    const absl::btree_multimap<std::string, std::string>& headers) {
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
    error_type = error_header->second;
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
      error_type = GetNodeText(root_node->FirstChildElement("Code"));
    }
    if (request_id.empty()) {
      request_id = GetNodeText(root_node->FirstChildElement("RequestId"));
    }
    message = GetNodeText(root_node->FirstChildElement("Message"));
  }();

  retryable = error_type.empty()
                  ? IsRetryableAwsStatusCode(response.status_code)
                  : IsRetryableAwsMessageCode(error_type);

  if (error_type.empty()) {
    error_type = "Unknown";
  }

  absl::Status status(absl_status_code,
                      absl::StrFormat("%s%s%s", error_type,
                                      message.empty() ? "" : ": ", message));

  status.SetPayload("http_response_code",
                    absl::Cord(absl::StrFormat("%d", response.status_code)));
  if (!payload_str.empty()) {
    status.SetPayload(
        "http_response_body",
        payload.Subcord(0,
                        payload_str.size() < 256 ? payload_str.size() : 256));
  }
  if (!request_id.empty()) {
    status.SetPayload("x-amzn-requestid", request_id);
  }

  MaybeAddSourceLocation(status, loc);
  return status;
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
