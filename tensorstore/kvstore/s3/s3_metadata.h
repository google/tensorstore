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

#ifndef TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_
#define TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_

/// \file
/// Key-value store where each key corresponds to a S3 object and the value is
/// stored as the file content.

#include <cstdint>

#include <map>
#include <string>
#include <string_view>

#include "absl/time/time.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_storage_s3 {

/// @brief Find the starting position of the tag or the position immediately after the tag
/// with the supplied XML payload
/// @param data XML payload
/// @param tag XML tag
/// @param pos Initial searching position
/// @param start true if starting position desired otherwise false
/// @return The tag starting position, otherwise the position immediately after the tag
Result<std::size_t> FindTag(std::string_view data, std::string_view tag,
                            std::size_t pos=0, bool start=true);

/// @brief Get tag contents within the supplied XML payload
///
/// This should primarily be used for obtaining the contents of inner tags.
/// Use FindTag to set the initial search position by traversing XML outer tags.
///
/// @param data XML payload
/// @param open_tag Opening tag
/// @param close_tag Closing tag
/// @param pos Initial searching position. Updated with position immediately
///            after the closing tag.
/// @return Tag contents
Result<std::string_view> GetTag(std::string_view data,
                                std::string_view open_tag,
                                std::string_view close_tag,
                                std::size_t * pos);

Result<StorageGeneration> ComputeGenerationFromHeaders(
    const std::multimap<std::string, std::string>& headers);

}  // namespace internal_storage_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_
