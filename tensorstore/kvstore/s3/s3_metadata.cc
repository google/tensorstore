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
#include <string>
#include <string_view>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "re2/re2.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/result.h"
#include "tinyxml2.h"

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

}  // namespace internal_kvstore_s3
}  // namespace tensorstore
