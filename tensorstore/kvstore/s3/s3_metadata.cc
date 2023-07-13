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

#include <map>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore_s3 {

Result<StorageGeneration> StorageGenerationFromHeaders(
    const std::multimap<std::string, std::string>& headers) {

  if(auto it = headers.find("etag"); it != headers.end()) {
    return StorageGeneration::FromString(it->second);
  } else {
    return absl::NotFoundError("etag not found in response headers");
  }
}

Result<std::size_t> FindTag(std::string_view data, std::string_view tag,
                            std::size_t pos, bool start) {
  if(pos=data.find(tag, pos); pos != std::string_view::npos) {
    return start ? pos :  pos + tag.length();
  }
  return absl::NotFoundError(
    absl::StrCat("Malformed List Response XML: can't find ", tag, " in ", data));
}

Result<std::string_view> GetTag(std::string_view data,
                                std::string_view open_tag,
                                std::string_view close_tag,
                                std::size_t * pos) {

  TENSORSTORE_ASSIGN_OR_RETURN(auto tagstart, FindTag(data, open_tag, *pos, false));
  TENSORSTORE_ASSIGN_OR_RETURN(auto tagend, FindTag(data, close_tag, tagstart, true));
  *pos = tagend + close_tag.size();
  return data.substr(tagstart, tagend - tagstart);
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
