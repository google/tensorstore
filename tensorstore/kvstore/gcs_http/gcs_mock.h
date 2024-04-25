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

#ifndef TENSORSTORE_KVSTORE_GCS_HTTP_GCS_MOCK_H_
#define TENSORSTORE_KVSTORE_GCS_HTTP_GCS_MOCK_H_

#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <variant>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/util/result.h"

namespace tensorstore {

/// GCSMockStorageBucket provides a minimal mocking environment for
/// GCS requests.
class GCSMockStorageBucket {
 public:
  // An individual data object with a name, value, and generation.
  struct Object {
    std::string name;
    absl::Cord data;
    int64_t generation;
  };

  using ParamMap = std::map<std::string_view, std::string>;

  virtual ~GCSMockStorageBucket();

  /// Constructs a HttpRequestMockContext for a GCS bucket.
  ///
  /// \param bucket The bucket name.
  /// \param requestor_pays_project_id If not `std::nullopt`, this bucket
  ///     behaves as requestor pays and furthermore validates that the
  ///     requestor_pays project id is equal to the specified value.  The check
  ///     for an exact project id is a mock version of the actual check done by
  ///     GCS that the specified project ID has billing enabled.
  GCSMockStorageBucket(
      std::string_view bucket,
      std::optional<std::string> requestor_pays_project_id = std::nullopt);

  // Responds to a "www.google.apis/storage/v1/b/bucket" request.
  Result<internal_http::HttpResponse> IssueRequest(
      const internal_http::HttpRequest& request, absl::Cord payload);

  // Main entry-point for matching requests.
  std::variant<std::monostate, internal_http::HttpResponse, absl::Status> Match(
      const internal_http::HttpRequest& request, absl::Cord payload);

  // List objects in the bucket.
  std::variant<std::monostate, internal_http::HttpResponse, absl::Status>
  HandleListRequest(std::string_view path, const ParamMap& params);

  // Insert an object into the bucket.
  std::variant<std::monostate, internal_http::HttpResponse, absl::Status>
  HandleInsertRequest(std::string_view path, const ParamMap& params,
                      absl::Cord payload);

  // Get an object, which might be the data or the metadata.
  std::variant<std::monostate, internal_http::HttpResponse, absl::Status>
  HandleGetRequest(const internal_http::HttpRequest& request,
                   std::string_view path, const ParamMap& params);

  // Delete an object.
  std::variant<std::monostate, internal_http::HttpResponse, absl::Status>
  HandleDeleteRequest(std::string_view path, const ParamMap& params);

  // Construct an object metadata response.
  internal_http::HttpResponse ObjectMetadataResponse(const Object& object);

  // Construct an object media response.
  internal_http::HttpResponse ObjectMediaResponse(
      const Object& object, std::optional<OptionalByteRangeRequest> byte_range);

  ::nlohmann::json ObjectMetadata(const Object& object);

  // Triggers a guaranteed error for the next `count` requests.
  void TriggerErrors(int64_t count) {
    assert(count >= 0);
    absl::MutexLock l(&mutex_);
    next_error_count_ += count;
    p_error_ = 0;
  }

  // Sets the error rate on the mock interface.
  void SetErrorRate(double p_error) {
    assert(p_error >= 0 && p_error <= 1);
    absl::MutexLock l(&mutex_);
    p_error_ = p_error;
  }

 private:
  const std::string bucket_;
  const std::string bucket_prefix_;
  const std::string upload_prefix_;
  const std::optional<std::string> requestor_pays_project_id_;
  absl::Mutex mutex_;
  int64_t next_generation_ = 123;

  int64_t next_error_count_ ABSL_GUARDED_BY(mutex_) = 0;
  double p_error_ ABSL_GUARDED_BY(mutex_) = 0.05;
  std::minstd_rand urbg_ ABSL_GUARDED_BY(mutex_);

  using Map = std::map<std::string, Object, std::less<>>;
  Map data_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_HTTP_GCS_MOCK_H_
