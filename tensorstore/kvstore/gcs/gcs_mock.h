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

#ifndef TENSORSTORE_KVSTORE_GCS_GCS_MOCK_H_
#define TENSORSTORE_KVSTORE_GCS_GCS_MOCK_H_

#include <stdint.h>

#include <map>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/variant.h"
#include "tensorstore/internal/http/http_request.h"
#include "tensorstore/internal/http/http_response.h"
#include "tensorstore/internal/http/http_transport.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/key_value_store_testutil.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/status.h"

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

  using ParamMap = std::map<absl::string_view, std::string>;

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
      absl::string_view bucket,
      std::optional<std::string> requestor_pays_project_id = std::nullopt);

  // Implement the HttpTransport::IssueRequest interface.
  // Responds to a "www.google.apis/storage/v1/b/bucket" request.
  Future<internal_http::HttpResponse> IssueRequest(
      const internal_http::HttpRequest& request, absl::Cord payload,
      absl::Duration request_timeout, absl::Duration connect_timeout);

  // Main entry-point for matching requests.
  absl::variant<absl::monostate, internal_http::HttpResponse, Status> Match(
      const internal_http::HttpRequest& request, absl::Cord payload);

  // List objects in the bucket.
  absl::variant<absl::monostate, internal_http::HttpResponse, Status>
  HandleListRequest(absl::string_view path, const ParamMap& params);

  // Insert an object into the bucket.
  absl::variant<absl::monostate, internal_http::HttpResponse, Status>
  HandleInsertRequest(absl::string_view path, const ParamMap& params,
                      absl::Cord payload);

  // Get an object, which might be the data or the metadata.
  absl::variant<absl::monostate, internal_http::HttpResponse, Status>
  HandleGetRequest(absl::string_view path, const ParamMap& params);

  // Delete an object.
  absl::variant<absl::monostate, internal_http::HttpResponse, Status>
  HandleDeleteRequest(absl::string_view path, const ParamMap& params);

  // Construct an ojbect metadata response.
  internal_http::HttpResponse ObjectMetadataResponse(const Object& object);

  // Construct an object media response.
  internal_http::HttpResponse ObjectMediaResponse(const Object& object);

  std::string ObjectMetadataString(const Object& object);

  // Triggers a guaranteed error for the next `count` requests.
  void TriggerErrors(int64_t count) {
    assert(count >= 0);
    next_error_count_ += count;
  }

 private:
  const std::string bucket_;
  const std::string bucket_path_;
  const std::string upload_path_;
  const std::optional<std::string> requestor_pays_project_id_;
  absl::Mutex mutex_;
  int64_t next_generation_ = 123;
  int64_t request_count_ = 0;
  int64_t next_error_count_ = 0;

  using Map = std::map<std::string, Object, std::less<>>;
  Map data_;
};

}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_GCS_MOCK_H_
