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

#include <cassert>
#include <map>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"

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

}  // namespace

Result<StorageGeneration> StorageGenerationFromHeaders(
    const std::multimap<std::string, std::string>& headers) {
  if (auto it = headers.find(kEtag); it != headers.end()) {
    return StorageGeneration::FromString(it->second);
  }
  return absl::NotFoundError("etag not found in response headers");
}

Result<size_t> FindTag(std::string_view data, std::string_view tag, size_t pos,
                       bool start) {
  if (pos = data.find(tag, pos); pos != std::string_view::npos) {
    return start ? pos : pos + tag.length();
  }
  return absl::NotFoundError(absl::StrCat(
      "Malformed List Response XML: can't find ", tag, " in ", data));
}

Result<TagAndPosition> GetTag(std::string_view data, std::string_view open_tag,
                              std::string_view close_tag, size_t pos) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto tagstart,
                               FindTag(data, open_tag, pos, false));
  TENSORSTORE_ASSIGN_OR_RETURN(auto tagend,
                               FindTag(data, close_tag, tagstart, true));
  return TagAndPosition{UnescapeXml(data.substr(tagstart, tagend - tagstart)),
                        tagend + close_tag.size() + 1};
}

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
