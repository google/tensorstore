// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs/gcs_mock.h"

#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"

using tensorstore::Status;
using tensorstore::internal_http::CurlUnescapeString;
using tensorstore::internal_http::HttpRequest;
using tensorstore::internal_http::HttpResponse;

namespace tensorstore {
namespace {

const char kInvalidLongBody[] =
    R"({"error": {"code": 400,  "message": "Invalid long value: '$0'." }})";

// QueryParameters are common between various GCS calls.
// https://cloud.google.com/storage/docs/json_api/v1/objects
struct QueryParameters {
  absl::optional<std::int64_t> ifGenerationMatch;
  absl::optional<std::int64_t> ifGenerationNotMatch;
};

// Parse QueryParameters or return an error HttpResponse
absl::optional<internal_http::HttpResponse> ParseQueryParameters(
    const GCSMockStorageBucket::ParamMap& params,
    QueryParameters* query_params) {
  // The generation must be numeric.
  for (auto it = params.find("ifGenerationMatch"); it != params.end();) {
    std::int64_t v = 0;
    if (!absl::SimpleAtoi(it->second, &v)) {
      return HttpResponse{
          400, absl::Cord(absl::Substitute(kInvalidLongBody, it->second))};
    }
    query_params->ifGenerationMatch = v;
    break;
  }
  for (auto it = params.find("ifGenerationNotMatch"); it != params.end();) {
    std::int64_t v = 0;
    if (!absl::SimpleAtoi(it->second, &v)) {
      return HttpResponse{
          400, absl::Cord(absl::Substitute(kInvalidLongBody, it->second))};
    }
    query_params->ifGenerationNotMatch = v;
    break;
  }
  return absl::nullopt;
}

}  // namespace

GCSMockStorageBucket::~GCSMockStorageBucket() = default;

GCSMockStorageBucket::GCSMockStorageBucket(
    absl::string_view bucket,
    std::optional<std::string> requestor_pays_project_id)
    : bucket_(bucket),
      bucket_path_(absl::StrCat("/storage/v1/b/", bucket)),
      upload_path_(absl::StrCat("/upload/storage/v1/b/", bucket)),
      requestor_pays_project_id_(std::move(requestor_pays_project_id)) {}

// Responds to a "www.google.apis/storage/v1/b/bucket" request.
Future<HttpResponse> GCSMockStorageBucket::IssueRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  // When using a mock context, we assume that the mock is
  // thread safe and not uninstalled when it might introduce
  // race conditions.
  auto match_result = Match(request, payload);
  if (absl::holds_alternative<Status>(match_result)) {
    return std::move(absl::get<Status>(match_result));
  } else if (absl::holds_alternative<HttpResponse>(match_result)) {
    return std::move(absl::get<HttpResponse>(match_result));
  }
  return absl::UnimplementedError("Mock cannot satisfy the request.");
}

absl::variant<absl::monostate, HttpResponse, Status>
GCSMockStorageBucket::Match(const HttpRequest& request, absl::Cord payload) {
  absl::string_view scheme, host, path;
  tensorstore::internal::ParseURI(request.url(), &scheme, &host, &path);

  if (host != "www.googleapis.com") {
    return {};
  }
  bool is_upload = false;
  if (absl::StartsWith(path, bucket_path_)) {
    // Bucket path.
    path.remove_prefix(bucket_path_.size());
  } else if (absl::StartsWith(path, upload_path_)) {
    // Upload path.
    path.remove_prefix(upload_path_.size());
    is_upload = true;
  } else {
    // Neither download nor upload path.
    return {};
  }

  absl::MutexLock l(&mutex_);

  // GCS can "randomly" return an HTTP 429.
  // In actuality, a 429 is based on the request rate for a resource, etc.
  bool trigger_error = false;
  if (next_error_count_ > 0) {
    trigger_error = true;
    --next_error_count_;
  }
  if (request_count_++ % 5 == 0) {
    trigger_error = true;
  }
  if (trigger_error) {
    return HttpResponse{429, absl::Cord()};
  }

  // Remove the query parameter substring.
  absl::string_view query;
  for (auto idx = path.find('?'); idx != absl::string_view::npos;) {
    query = path.substr(idx + 1);
    path.remove_suffix(1 + query.size());
    break;
  }

  // Parse the query params.
  std::map<absl::string_view, std::string> params;
  if (!query.empty()) {
    for (absl::string_view kv : absl::StrSplit(query, absl::ByChar('&'))) {
      std::pair<absl::string_view, absl::string_view> split =
          absl::StrSplit(kv, absl::MaxSplits('=', 1));
      params[split.first] = CurlUnescapeString(split.second);
    }
  }

  std::optional<std::string> user_project;
  if (auto it = params.find("userProject"); it != params.end()) {
    user_project = it->second;
  }

  if (requestor_pays_project_id_ &&
      (!user_project || *user_project != *requestor_pays_project_id_)) {
    // https://cloud.google.com/storage/docs/requester-pays
    return HttpResponse{400, absl::Cord("UserProjectMissing")};
  }

  // Dispatch based on path, method, etc.
  if (path == "/o" && request.method().empty() && payload.empty()) {
    // GET request for the bucket.
    return HandleListRequest(path, params);
  } else if (path == "/o" && request.method().empty() && !payload.empty()) {
    // POST
    if (!is_upload) {
      return HttpResponse{
          400,
          absl::Cord(
              R"({ "error": { "code": 400, "message": "Uploads must be sent to the upload URL." } })")};
    }
    return HandleInsertRequest(path, params, payload);
  } else if (absl::StartsWith(path, "/o/") && request.method().empty()) {
    // GET request on an object.
    return HandleGetRequest(path, params);
  } else if (absl::StartsWith(path, "/o/") && request.method() == "DELETE") {
    // DELETE request on an object.
    return HandleDeleteRequest(path, params);
  }

  // NOT HANDLED
  // update (PUT request)
  // .../compose
  // .../watch
  // .../rewrite/...
  // patch (PATCH request)
  // .../copyTo/...

  return HttpResponse{404, absl::Cord()};
}

absl::variant<absl::monostate, HttpResponse, Status>
GCSMockStorageBucket::HandleListRequest(absl::string_view path,
                                        const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/list
  const char kPrefix[] = R"(
{
 "kind": "storage#objects",
 "items": [)";

  const char kSuffix[] = R"(
  ],
  "nextPageToken": "$0"
}
)";

  const char kShortSuffix[] = R"(
  ]
})";

  // TODO: handle Delimiter
  std::int64_t maxResults = std::numeric_limits<std::int64_t>::max();
  for (auto it = params.find("maxResults"); it != params.end();) {
    if (!absl::SimpleAtoi(it->second, &maxResults) || maxResults < 1) {
      return HttpResponse{
          400, absl::Cord(absl::Substitute(kInvalidLongBody, it->second))};
    }
    break;
  }

  absl::string_view start_offset;
  Map::const_iterator object_it;
  if (auto it = params.find("pageToken"); it != params.end()) {
    start_offset = it->second;
    object_it = data_.upper_bound(it->second);
  } else if (auto it = params.find("startOffset"); it != params.end()) {
    start_offset = it->second;
    object_it = data_.lower_bound(start_offset);
  } else {
    object_it = data_.begin();
  }

  Map::const_iterator object_end_it;
  if (auto it = params.find("endOffset"); it != params.end()) {
    absl::string_view end_offset = it->second;
    if (end_offset <= start_offset) {
      object_end_it = object_it;
    } else {
      object_end_it = data_.lower_bound(end_offset);
    }
  } else {
    object_end_it = data_.end();
  }

  // NOTE: Use ::nlohmann::json to construct json objects & dump the response.
  std::string result(kPrefix);
  bool add_comma = false;
  for (; object_it != object_end_it; ++object_it) {
    if (add_comma) {
      absl::StrAppend(&result, ",\n");
    }
    absl::StrAppend(&result, ObjectMetadataString(object_it->second));
    add_comma = true;
    if (maxResults-- <= 0) break;
  }
  if (object_it == object_end_it) {
    absl::StrAppend(&result, kShortSuffix);
  } else {
    absl::StrAppend(&result, absl::Substitute(kSuffix, object_it->first));
  }

  return HttpResponse{200, absl::Cord(std::move(result))};
}

absl::variant<absl::monostate, HttpResponse, Status>
GCSMockStorageBucket::HandleInsertRequest(absl::string_view path,
                                          const ParamMap& params,
                                          absl::Cord payload) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/insert
  QueryParameters parsed_parameters;
  {
    auto parse_result = ParseQueryParameters(params, &parsed_parameters);
    if (parse_result.has_value()) {
      return std::move(parse_result.value());
    }
  }

  do {
    /// TODO: What does GCS return if these values are bad?
    auto uploadType = params.find("uploadType");
    if (uploadType == params.end() || uploadType->second != "media") break;

    auto name_it = params.find("name");
    if (name_it == params.end() || name_it->second.empty()) break;
    std::string name(name_it->second.data(), name_it->second.length());

    auto it = data_.find(name);
    if (parsed_parameters.ifGenerationMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationMatch.value();
      if (v == 0) {
        if (it != data_.end()) {
          // Live version => failure
          return HttpResponse{412, absl::Cord()};
        }
        // No live versions => success;
      } else if (it == data_.end() || v != it->second.generation) {
        // generation does not match.
        return HttpResponse{412, absl::Cord()};
      }
    }

    if (parsed_parameters.ifGenerationNotMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationNotMatch.value();
      if (it != data_.end() && v == it->second.generation) {
        // generation matches.
        return HttpResponse{412, absl::Cord()};
      }
    }

    auto& obj = data_[name];
    if (obj.name.empty()) {
      obj.name = std::move(name);
    }
    obj.generation = ++next_generation_;
    obj.data = payload;

    TENSORSTORE_LOG("Uploaded: ", obj.name, " ", obj.generation);

    return ObjectMetadataResponse(obj);
  } while (false);

  return HttpResponse{404, absl::Cord()};
}

absl::variant<absl::monostate, HttpResponse, Status>
GCSMockStorageBucket::HandleGetRequest(absl::string_view path,
                                       const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/get
  path.remove_prefix(3);  // remove /o/
  std::string name(path.data(), path.length());

  QueryParameters parsed_parameters;
  {
    auto parse_result = ParseQueryParameters(params, &parsed_parameters);
    if (parse_result.has_value()) {
      return std::move(parse_result.value());
    }
  }

  do {
    auto it = data_.find(name);

    if (parsed_parameters.ifGenerationMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationMatch.value();
      if (v == 0) {
        if (it != data_.end()) {
          // Live version => failure
          return HttpResponse{412, absl::Cord()};
        }
        // No live versions => success;
        return HttpResponse{204, absl::Cord()};
      } else if (it == data_.end() || v != it->second.generation) {
        // generation does not match.
        return HttpResponse{412, absl::Cord()};
      }
    }
    if (it == data_.end()) break;

    if (parsed_parameters.ifGenerationNotMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationNotMatch.value();
      if (v == it->second.generation) {
        // generation matches.
        return HttpResponse{304, absl::Cord()};
      }
    }

    /// Not a media request.
    auto alt = params.find("alt");
    if (params.empty() || alt == params.end() || alt->second != "media") {
      return ObjectMetadataResponse(it->second);
    }
    return ObjectMediaResponse(it->second);
  } while (false);

  return HttpResponse{404, absl::Cord()};
}

absl::variant<absl::monostate, HttpResponse, Status>
GCSMockStorageBucket::HandleDeleteRequest(absl::string_view path,
                                          const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/delete
  path.remove_prefix(3);  // remove /o/
  std::string name(path.data(), path.length());

  QueryParameters parsed_parameters;
  {
    auto parse_result = ParseQueryParameters(params, &parsed_parameters);
    if (parse_result.has_value()) {
      return std::move(parse_result.value());
    }
  }

  do {
    auto it = data_.find(name);
    if (it == data_.end()) {
      // No live versions => 404.
      break;
    }

    if (parsed_parameters.ifGenerationMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationMatch.value();
      if (v == 0 || v != it->second.generation) {
        // Live version, but generation does not match.
        return HttpResponse{412, absl::Cord()};
      }
    }

    TENSORSTORE_LOG("Deleted: ", path, " ", it->second.generation);

    data_.erase(it);
    return HttpResponse{204, absl::Cord()};
  } while (false);

  return HttpResponse{404, absl::Cord()};
}

HttpResponse GCSMockStorageBucket::ObjectMetadataResponse(
    const Object& object) {
  std::string data = ObjectMetadataString(object);
  HttpResponse response{200, absl::Cord(std::move(data))};
  response.headers.insert(
      {"content-length", absl::StrCat(response.payload.size())});
  response.headers.insert({"content-type", "application/json"});
  return response;
}

std::string GCSMockStorageBucket::ObjectMetadataString(const Object& object) {
  // NOTE:  Use ::nlohmann::json to construct json objects & dump the response.
  return absl::Substitute(
      R"({
  "kind": "storage#object",
  "id": "$0/$1/$2",
  "selfLink": "https://www.googleapis.com/storage/v1/b/$0/o/$1",
  "name": "$1",
  "bucket": "$0",
  "generation": "$2",
  "metageneration": "1",
  "contentType": "application/octet-stream",
  "timeCreated": "2018-10-24T00:41:38.264Z",
  "updated": "2018-10-24T00:41:38.264Z",
  "storageClass": "MULTI_REGIONAL",
  "timeStorageClassUpdated": "2018-10-24T00:41:38.264Z",
  "size": "$3",
  "mediaLink": "https://www.googleapis.com/download/storage/v1/b/$0/o/$1?generation=$2&alt=media"
 })",
      bucket_, object.name, object.generation, object.data.size());
}

HttpResponse GCSMockStorageBucket::ObjectMediaResponse(const Object& object) {
  HttpResponse response{200, object.data};
  response.headers.insert(
      {"content-length", absl::StrCat(response.payload.size())});
  response.headers.insert({"content-type", "application/octet-stream"});
  response.headers.insert(
      {"x-goog-generation", absl::StrCat(object.generation)});
  response.headers.insert({"x-goog-metageneration", "1"});
  response.headers.insert({"x-goog-storage-class", "MULTI_REGIONAL"});
  // todo: x-goog-hash
  return response;
}

}  // namespace tensorstore
