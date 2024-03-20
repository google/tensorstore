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

#include <stddef.h>

#include <string>

#include "absl/container/btree_map.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"
#include "tinyxml2.h"
namespace tensorstore {
namespace internal_kvstore_s3 {

/// Returns the text within an html node.
/// For example, if the node is <Key> as in:
///   <Key>Foo/bar</Key>
/// The result is 'Foo/bar'
std::string GetNodeText(tinyxml2::XMLNode* node);

/// Creates a storage generation from the etag header in the
/// HTTP response `headers`.
///
/// An etag is a hash of the S3 object contained in double-quotes.
/// For example "abcdef1234567890".
/// https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
/// This may or may not be an MD5 digest of the data
/// https://docs.aws.amazon.com/AmazonS3/latest/API/API_Object.html
Result<StorageGeneration> StorageGenerationFromHeaders(
    const absl::btree_multimap<std::string, std::string>& headers);

}  // namespace internal_kvstore_s3
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_S3_OBJECT_METADATA_H_
