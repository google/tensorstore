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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/substitute.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/http/curl_handle.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/test_util.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace {

using ::tensorstore::internal_http::HttpRequest;
using ::tensorstore::internal_http::HttpResponse;

const char kInvalidLongBody[] =
    R"({"error": {"code": 400,  "message": "Invalid long value: '$0'." }})";

// QueryParameters are common between various GCS calls.
// https://cloud.google.com/storage/docs/json_api/v1/objects
struct QueryParameters {
  std::optional<std::int64_t> ifGenerationMatch;
  std::optional<std::int64_t> ifGenerationNotMatch;
};

// Parse QueryParameters or return an error HttpResponse
std::optional<internal_http::HttpResponse> ParseQueryParameters(
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
  return std::nullopt;
}

}  // namespace

GCSMockStorageBucket::~GCSMockStorageBucket() = default;

GCSMockStorageBucket::GCSMockStorageBucket(
    std::string_view bucket,
    std::optional<std::string> requestor_pays_project_id)
    : bucket_(bucket),
      bucket_prefix_(
          tensorstore::StrCat("storage.googleapis.com/storage/v1/b/", bucket)),
      upload_prefix_(tensorstore::StrCat(
          "storage.googleapis.com/upload/storage/v1/b/", bucket)),
      requestor_pays_project_id_(std::move(requestor_pays_project_id)) {}

// Responds to a "www.google.apis/storage/v1/b/bucket" request.
Future<HttpResponse> GCSMockStorageBucket::IssueRequest(
    const HttpRequest& request, absl::Cord payload,
    absl::Duration request_timeout, absl::Duration connect_timeout) {
  // When using a mock context, we assume that the mock is
  // thread safe and not uninstalled when it might introduce
  // race conditions.
  auto match_result = Match(request, payload);
  if (std::holds_alternative<absl::Status>(match_result)) {
    return std::move(std::get<absl::Status>(match_result));
  } else if (std::holds_alternative<HttpResponse>(match_result)) {
    return std::move(std::get<HttpResponse>(match_result));
  }
  return absl::UnimplementedError(
      tensorstore::StrCat("Mock cannot satisfy the request: ", request.url()));
}

std::variant<std::monostate, HttpResponse, absl::Status>
GCSMockStorageBucket::Match(const HttpRequest& request, absl::Cord payload) {
  bool is_upload = false;
  auto parsed = internal::ParseGenericUri(request.url());
  if (parsed.scheme != "https") {
    return {};
  }
  std::string_view path = parsed.authority_and_path;
  if (absl::StartsWith(path, bucket_prefix_)) {
    // Bucket path.
    path.remove_prefix(bucket_prefix_.size());
  } else if (absl::StartsWith(path, upload_prefix_)) {
    // Upload path.
    path.remove_prefix(upload_prefix_.size());
    is_upload = true;
  } else {
    // Neither download nor upload path.
    return {};
  }

  absl::MutexLock l(&mutex_);

  // https://cloud.google.com/storage/docs/request-rate
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

  // Parse the query params.
  std::map<std::string_view, std::string> params;
  if (!parsed.query.empty()) {
    for (std::string_view kv :
         absl::StrSplit(parsed.query, absl::ByChar('&'))) {
      std::pair<std::string_view, std::string_view> split =
          absl::StrSplit(kv, absl::MaxSplits('=', 1));
      params[split.first] = internal::PercentDecode(split.second);
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
  if (path == "/o" && request.method() == "GET") {
    // GET request for the bucket.
    return HandleListRequest(path, params);
  } else if (path == "/o" && request.method() == "POST") {
    if (!is_upload) {
      return HttpResponse{
          400,
          absl::Cord(
              R"({ "error": { "code": 400, "message": "Uploads must be sent to the upload URL." } })")};
    }
    return HandleInsertRequest(path, params, payload);
  } else if (absl::StartsWith(path, "/o/") && request.method() == "GET") {
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

std::variant<std::monostate, HttpResponse, absl::Status>
GCSMockStorageBucket::HandleListRequest(std::string_view path,
                                        const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/list
  // TODO: handle Delimiter
  std::int64_t maxResults = std::numeric_limits<std::int64_t>::max();
  for (auto it = params.find("maxResults"); it != params.end();) {
    if (!absl::SimpleAtoi(it->second, &maxResults) || maxResults < 1) {
      return HttpResponse{
          400, absl::Cord(absl::Substitute(kInvalidLongBody, it->second))};
    }
    break;
  }

  std::string_view start_offset;
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
    std::string_view end_offset = it->second;
    if (end_offset <= start_offset) {
      object_end_it = object_it;
    } else {
      object_end_it = data_.lower_bound(end_offset);
    }
  } else {
    object_end_it = data_.end();
  }

  // NOTE: Use ::nlohmann::json to construct json objects & dump the response.
  ::nlohmann::json result{{"kind", "storage#objects"}};
  ::nlohmann::json::array_t items;
  for (; object_it != object_end_it; ++object_it) {
    items.push_back(ObjectMetadata(object_it->second));
    if (maxResults-- <= 0) break;
  }
  result["items"] = std::move(items);
  if (object_it != object_end_it) {
    result["nextPageToken"] = object_it->first;
  }
  return HttpResponse{200, absl::Cord(result.dump())};
}

std::variant<std::monostate, HttpResponse, absl::Status>
GCSMockStorageBucket::HandleInsertRequest(std::string_view path,
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

    ABSL_LOG(INFO) << "Uploaded: " << obj.name << " " << obj.generation;

    return ObjectMetadataResponse(obj);
  } while (false);

  return HttpResponse{404, absl::Cord()};
}

std::variant<std::monostate, HttpResponse, absl::Status>
GCSMockStorageBucket::HandleGetRequest(std::string_view path,
                                       const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/get
  path.remove_prefix(3);  // remove /o/
  std::string name = internal::PercentDecode(path);

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
          return HttpResponse{412};
        }
        // No live versions => success;
        return HttpResponse{204};
      } else if (it == data_.end() || v != it->second.generation) {
        // generation does not match.
        return HttpResponse{412};
      }
    }
    if (it == data_.end()) break;

    if (parsed_parameters.ifGenerationNotMatch.has_value()) {
      const std::int64_t v = parsed_parameters.ifGenerationNotMatch.value();
      if (v == it->second.generation) {
        // generation matches.
        return HttpResponse{304};
      }
    }

    /// Not a media request.
    auto alt = params.find("alt");
    if (params.empty() || alt == params.end() || alt->second != "media") {
      return ObjectMetadataResponse(it->second);
    }
    return ObjectMediaResponse(it->second);
  } while (false);

  return HttpResponse{404};
}

std::variant<std::monostate, HttpResponse, absl::Status>
GCSMockStorageBucket::HandleDeleteRequest(std::string_view path,
                                          const ParamMap& params) {
  // https://cloud.google.com/storage/docs/json_api/v1/objects/delete
  path.remove_prefix(3);  // remove /o/
  std::string name = internal::PercentDecode(path);

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

    ABSL_LOG(INFO) << "Deleted: " << name << " " << it->second.generation;

    data_.erase(it);
    return HttpResponse{204, absl::Cord()};
  } while (false);

  return HttpResponse{404, absl::Cord()};
}

HttpResponse GCSMockStorageBucket::ObjectMetadataResponse(
    const Object& object) {
  std::string data = ObjectMetadata(object).dump();
  HttpResponse response{200, absl::Cord(std::move(data))};
  response.headers.insert(
      {"content-length", tensorstore::StrCat(response.payload.size())});
  response.headers.insert({"content-type", "application/json"});
  return response;
}

::nlohmann::json GCSMockStorageBucket::ObjectMetadata(const Object& object) {
  return {
      {"kind", "storage#object"},
      {"id",
       tensorstore::StrCat(bucket_, "/", object.name, "/", object.generation)},
      {"selfLink",
       tensorstore::StrCat("https://www.googleapis.com/storage/v1/b/", bucket_,
                           "/o/",
                           internal::PercentEncodeUriComponent(object.name))},
      {"name", object.name},
      {"bucket", bucket_},
      {"generation", tensorstore::StrCat(object.generation)},
      {"metageneration", "1"},
      {"contentType", "application/octet-stream"},
      {"timeCreated", "2018-10-24T00:41:38.264Z"},
      {"updated", "2018-10-24T00:41:38.264Z"},
      {"storageClass", "MULTI_REGIONAL"},
      {"timeStorageClassUpdated", "2018-10-24T00:41:38.264Z"},
      {"size", tensorstore::StrCat(object.data.size())},
      {"mediaLink",
       tensorstore::StrCat("https://www.googleapis.com/download/storage/v1/b/",
                           bucket_, "/o/",
                           internal::PercentEncodeUriComponent(object.name),
                           "?generation=", object.generation, "&alt=media")},
  };
}

HttpResponse GCSMockStorageBucket::ObjectMediaResponse(const Object& object) {
  HttpResponse response{200, object.data};
  response.headers.insert(
      {"content-length", tensorstore::StrCat(response.payload.size())});
  response.headers.insert({"content-type", "application/octet-stream"});
  response.headers.insert(
      {"x-goog-generation", tensorstore::StrCat(object.generation)});
  response.headers.insert({"x-goog-metageneration", "1"});
  response.headers.insert({"x-goog-storage-class", "MULTI_REGIONAL"});
  // todo: x-goog-hash
  return response;
}

}  // namespace tensorstore
