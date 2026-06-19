// Copyright 2026 The TensorStore Authors
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

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "re2/re2.h"
#include "tensorstore/internal/metrics/domain_impl.h"

ABSL_FLAG(std::string, type, "auto",
          "Type of keys: 'auto', 'string', 'int8_t', 'uint8_t', 'int16_t', "
          "'uint16_t', 'int32_t', 'uint32_t', 'int64_t', 'uint64_t'");

ABSL_FLAG(bool, case_sensitive, false, "Case sensitive for string keys");

ABSL_FLAG(int64_t, table_size, 0,
          "Table size for perfect hash. If 0, searches upward "
          "from N to find the smallest table that admits a perfect hash. "
          "If set, searches all uint32 seeds for that exact table size.");

ABSL_FLAG(uint32_t, max_seed, 10000000u,
          "Maximum seed to search per table size when iterating "
          "(ignored when --table_size is set).");

ABSL_FLAG(std::string, file, "", "File to modify inline");

// DomainField hashing was designed to support compile-time, constexpr
// construction of the seed, but it was removed since it was potentially
// expensive and added a little complexity to the class.  The find_seed
// tool can be used to find seed values at declaration time instead,
// guaranteeing that the parameterization is identical.
template <typename T, bool CaseSensitive = false>
std::optional<uint32_t> FindSeed(const T* keys, size_t n, uint8_t* used_buf,
                                 size_t table_size, uint32_t max_seed) {
  // Sequential search works well for seed finding since the FNV hash has
  // good avalanche properties, and collisions are detected early.
  for (uint64_t seed = 0; seed < static_cast<uint64_t>(max_seed); ++seed) {
    std::memset(used_buf, 0, table_size * sizeof(*used_buf));
    if (tensorstore::internal_metrics::IsPerfect<T, CaseSensitive>(
            keys, n, static_cast<uint32_t>(seed), used_buf, table_size)) {
      return static_cast<uint32_t>(seed);
    }
  }
  return std::nullopt;
}

template <typename T, bool CaseSensitive = false>
std::optional<std::pair<uint32_t, size_t>> FindOptimalSeedAndSize(
    const T* keys, size_t n, size_t min_t, uint32_t max_seed) {
  for (size_t t = min_t; t < std::numeric_limits<size_t>::max(); ++t) {
    auto used = std::make_unique<uint8_t[]>(t);
    auto seed = FindSeed<T, CaseSensitive>(keys, n, used.get(), t, max_seed);
    if (seed.has_value()) {
      return std::make_pair(seed.value(), t);
    }
  }
  return std::nullopt;
}

std::vector<std::string> ParseTokens(const std::string& input) {
  std::vector<std::string> tokens;
  std::string current;
  bool in_quotes = false;
  for (size_t i = 0; i < input.size(); ++i) {
    char c = input[i];
    if (c == '"') {
      in_quotes = !in_quotes;
    } else if (absl::ascii_isspace(c) && !in_quotes) {
      if (!current.empty()) {
        // Handle line-continuation markers in multi-line comments: trailing
        // backslashes are stripped, and tokens that are exactly "\\" are
        // dropped.
        if (current != "\\") {
          if (current.back() == '\\') current.pop_back();
          tokens.push_back(current);
        }
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  // Handle line-continuation markers at the end of the input string.
  if (!current.empty() && current != "\\") {
    if (current.back() == '\\') current.pop_back();
    tokens.push_back(current);
  }
  return tokens;
}

struct FindSeedOptions {
  std::string type = "auto";
  bool case_sensitive = false;
  size_t table_size = 0;
  uint32_t max_seed = 10000000u;
};

struct BlockArgs {
  std::vector<std::string> keys;
  FindSeedOptions options;
};

FindSeedOptions OptionsFromFlags() {
  FindSeedOptions opts;
  opts.type = absl::GetFlag(FLAGS_type);
  opts.case_sensitive = absl::GetFlag(FLAGS_case_sensitive);
  int64_t ts = absl::GetFlag(FLAGS_table_size);
  opts.table_size = ts < 0 ? 0 : static_cast<size_t>(ts);
  opts.max_seed = absl::GetFlag(FLAGS_max_seed);
  return opts;
}

BlockArgs ParseBlockArgs(const std::vector<std::string>& tokens) {
  absl::SetFlag(&FLAGS_type, "auto");
  absl::SetFlag(&FLAGS_case_sensitive, false);
  absl::SetFlag(&FLAGS_table_size, 0);
  absl::SetFlag(&FLAGS_max_seed, 10000000);

  std::vector<std::string> args = {"find_seed"};
  args.insert(args.end(), tokens.begin(), tokens.end());

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& s : args) {
    argv.push_back(s.data());
  }
  argv.push_back(nullptr);

  int argc = argv.size() - 1;
  std::vector<char*> positional = absl::ParseCommandLine(argc, argv.data());

  BlockArgs result;
  result.options = OptionsFromFlags();

  for (size_t i = 1; i < positional.size(); ++i) {
    std::string token = positional[i];
    if (!token.empty() && token.back() == '\\') {
      token.pop_back();
    }
    if (!token.empty()) {
      result.keys.push_back(token);
    }
  }
  return result;
}

template <typename T>
std::optional<std::pair<uint32_t, size_t>> HandleIntegerArgs(
    const std::vector<std::string>& keys_args, size_t min_table_size, size_t n,
    uint32_t max_seed, bool fixed_size) {
  std::vector<T> keys;
  keys.reserve(n);
  for (const auto& k : keys_args) {
    using U = std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
    U val;
    if (!absl::SimpleAtoi(k, &val)) {
      LOG(ERROR) << "Invalid integer key: " << k;
      return std::nullopt;
    }
    if (val < static_cast<U>(std::numeric_limits<T>::min()) ||
        val > static_cast<U>(std::numeric_limits<T>::max())) {
      LOG(ERROR) << "Integer key out of range: " << k;
      return std::nullopt;
    }
    keys.push_back(static_cast<T>(val));
  }

  // Check duplicates
  absl::flat_hash_set<T> seen;
  for (const auto& key : keys) {
    if (!seen.insert(key).second) {
      using PrintType =
          std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>;
      LOG(ERROR) << "Duplicate key: " << static_cast<PrintType>(key);
      return std::nullopt;
    }
  }

  if (fixed_size) {
    // n > 0 is guaranteed by the earlier check in FindSeedForKeys.
    auto used = std::make_unique<uint8_t[]>(min_table_size);
    auto seed =
        FindSeed<T>(keys.data(), n, used.get(), min_table_size, max_seed);
    if (seed.has_value()) {
      return std::make_pair(seed.value(), min_table_size);
    }
    return std::nullopt;
  }
  return FindOptimalSeedAndSize<T>(keys.data(), n, min_table_size, max_seed);
}

std::string AutoDetectType(const std::vector<std::string>& keys) {
  if (keys.empty()) return "string";

  bool needs_int64 = false;
  bool needs_uint64 = false;

  for (const auto& k : keys) {
    int64_t val_i64;
    if (absl::SimpleAtoi(k, &val_i64)) {
      if (val_i64 < std::numeric_limits<int32_t>::min() ||
          val_i64 > std::numeric_limits<int32_t>::max()) {
        needs_int64 = true;
      }
      continue;
    }

    uint64_t val_u64;
    if (absl::SimpleAtoi(k, &val_u64)) {
      needs_uint64 = true;
      continue;
    }

    return "string";
  }

  if (needs_uint64) return "uint64_t";
  if (needs_int64) return "int64_t";
  return "int32_t";
}

std::optional<std::pair<uint32_t, size_t>> FindSeedForKeys(
    const std::vector<std::string>& keys_args, const FindSeedOptions& opts) {
  size_t n = keys_args.size();
  if (n == 0) return std::nullopt;

  std::string type = opts.type;
  if (type == "auto") {
    if (opts.case_sensitive) {
      type = "string";
    } else {
      type = AutoDetectType(keys_args);
    }
  }
  bool case_sensitive = opts.case_sensitive;
  size_t table_size = opts.table_size;
  uint32_t max_seed = opts.max_seed;

  // When table_size is specified, fix the size and search all uint32 seeds.
  // When not specified, iterate table sizes upward from n, bounded by
  // max_seed per size.
  bool fixed_size = table_size > 0;
  size_t min_table_size = n;
  if (fixed_size) {
    if (table_size < n) {
      LOG(ERROR) << "Table size (" << table_size
                 << ") cannot be less than number of keys (" << n << ").";
      return std::nullopt;
    }
    min_table_size = table_size;
    max_seed = UINT32_MAX;
  }

  if (type == "string") {
    absl::flat_hash_set<std::string> seen;
    std::vector<std::string_view> keys;
    keys.reserve(n);
    for (const auto& k : keys_args) {
      std::string normalized = k;
      if (!case_sensitive) {
        normalized = absl::AsciiStrToLower(k);
      }
      if (!seen.insert(normalized).second) {
        LOG(ERROR) << "Duplicate key: " << k;
        return std::nullopt;
      }
      // Note: We duplicate-check using lowercase (`normalized`), but pass the
      // original string `k` to the perfect hash construct. This is correct
      // because `IsPerfect` with `CaseSensitive=false` internally performs
      // case-insensitive comparisons/hashing (using `ToLower`), but we must
      // ensure we don't store a `string_view` pointing to `normalized` which
      // goes out of scope here.
      keys.push_back(k);
    }

    if (fixed_size) {
      // n > 0 is guaranteed by the check at the start of the function.
      auto used = std::make_unique<uint8_t[]>(min_table_size);
      auto seed =
          case_sensitive
              ? FindSeed<std::string_view, true>(keys.data(), n, used.get(),
                                                 min_table_size, max_seed)
              : FindSeed<std::string_view, false>(keys.data(), n, used.get(),
                                                  min_table_size, max_seed);
      if (seed.has_value()) {
        return std::make_pair(seed.value(), min_table_size);
      }
      return std::nullopt;
    }
    return case_sensitive ? FindOptimalSeedAndSize<std::string_view, true>(
                                keys.data(), n, min_table_size, max_seed)
                          : FindOptimalSeedAndSize<std::string_view, false>(
                                keys.data(), n, min_table_size, max_seed);
  }

  using IntegerHandler = std::optional<std::pair<uint32_t, size_t>> (*)(
      const std::vector<std::string>&, size_t, size_t, uint32_t, bool);
  struct TypeDispatcher {
    std::string_view name;
    IntegerHandler handler;
  };
  static constexpr TypeDispatcher kDispatchers[] = {
      {"int8_t", &HandleIntegerArgs<int8_t>},
      {"uint8_t", &HandleIntegerArgs<uint8_t>},
      {"int16_t", &HandleIntegerArgs<int16_t>},
      {"uint16_t", &HandleIntegerArgs<uint16_t>},
      {"int32_t", &HandleIntegerArgs<int32_t>},
      {"uint32_t", &HandleIntegerArgs<uint32_t>},
      {"int64_t", &HandleIntegerArgs<int64_t>},
      {"uint64_t", &HandleIntegerArgs<uint64_t>},
  };

  for (const auto& dispatcher : kDispatchers) {
    if (dispatcher.name == type) {
      return dispatcher.handler(keys_args, min_table_size, n, max_seed,
                                fixed_size);
    }
  }

  LOG(ERROR) << "Unknown type: " << type;
  return std::nullopt;
}

struct FindSeedBlock {
  size_t start_line;
  std::vector<std::string> comment_contents;
  std::optional<size_t> kseed_line;
  std::optional<size_t> ktable_line;
};

struct DeclarationLocations {
  std::optional<size_t> kseed_line;
  std::optional<size_t> ktable_line;
};

static constexpr size_t kMaxDeclarationDistance = 10;

// Returns the marker position (the index of the character immediately after the
// matched marker string) if the line starts a FIND_SEED block, otherwise
// std::nullopt.
std::optional<size_t> FindMarkerPosition(std::string_view line) {
  size_t slashes = line.find("//");
  if (slashes == std::string::npos) return std::nullopt;

  size_t comment_start = line.find_first_not_of(" \t", slashes + 2);
  if (comment_start == std::string::npos) return std::nullopt;

  for (std::string_view marker :
       {"FIND_SEED",
        "run //tensorstore/internal/metrics:find_seed --"}) {
    if (line.compare(comment_start, marker.size(), marker) == 0) {
      size_t next_pos = comment_start + marker.size();
      if (next_pos >= line.size() ||
          (!absl::ascii_isalnum(line[next_pos]) && line[next_pos] != '_')) {
        return next_pos;
      }
    }
  }
  return std::nullopt;
}

// Starting at `start` line index, collects contiguous comment lines.
// Returns the collected text and the index of the first non-comment line.
struct CommentBlock {
  std::vector<std::string> contents;
  size_t end_line;
};
CommentBlock CollectContinuationComments(const std::vector<std::string>& lines,
                                         size_t start) {
  CommentBlock block;
  size_t j = start + 1;
  while (j < lines.size()) {
    size_t slashes = lines[j].find("//");
    bool is_comment = false;
    if (slashes != std::string::npos) {
      is_comment = true;
      for (size_t k = 0; k < slashes; ++k) {
        if (!absl::ascii_isspace(lines[j][k])) {
          is_comment = false;
          break;
        }
      }
    }
    if (!is_comment) {
      break;
    }
    block.contents.push_back(lines[j].substr(slashes + 2));
    j++;
  }
  block.end_line = j;
  return block;
}

bool IsDeclarationLine(std::string_view line, std::string_view var_name) {
  size_t comment_pos = line.find("//");
  std::string_view code_part =
      (comment_pos == std::string::npos) ? line : line.substr(0, comment_pos);
  std::string pattern =
      absl::StrCat("\\b", RE2::QuoteMeta(var_name), "\\b\\s*=");
  return RE2::PartialMatch(code_part, pattern);
}

// Scans up to kMaxDeclarationDistance lines forward from `start` looking
// for "kSeed = ..." and "kTableSize = ..." declarations.
DeclarationLocations FindDeclarations(const std::vector<std::string>& lines,
                                      size_t start) {
  DeclarationLocations locs;
  size_t limit = std::min(lines.size(), start + kMaxDeclarationDistance);
  for (size_t next_i = start; next_i < limit; ++next_i) {
    if (!locs.kseed_line && IsDeclarationLine(lines[next_i], "kSeed")) {
      locs.kseed_line = next_i;
    }
    if (!locs.ktable_line && IsDeclarationLine(lines[next_i], "kTableSize")) {
      locs.ktable_line = next_i;
    }
  }
  return locs;
}

std::vector<std::string> ReadFileLines(const std::string& filepath) {
  std::vector<std::string> lines;
  std::ifstream f(filepath);
  if (!f.is_open()) return lines;
  std::string line;
  while (std::getline(f, line)) {
    lines.push_back(line);
  }
  return lines;
}

bool WriteFileLines(const std::string& filepath,
                    const std::vector<std::string>& lines) {
  std::ofstream f(filepath);
  if (!f.is_open()) return false;
  for (const auto& line : lines) {
    f << line << "\n";
  }
  f.flush();
  return f.good();
}

std::optional<std::string> UpdateDeclarationLine(const std::string& line,
                                                 const std::string& var_name,
                                                 const std::string& new_val) {
  // Only look before line comments.
  size_t comment_pos = line.find("//");
  std::string code_part =
      (comment_pos == std::string::npos) ? line : line.substr(0, comment_pos);
  std::string comment_part =
      (comment_pos == std::string::npos) ? "" : line.substr(comment_pos);

  std::string prefix, old_val, suffix;
  std::string pattern = absl::StrCat("(.*?\\b", RE2::QuoteMeta(var_name),
                                     "\\b\\s*=\\s*)([^ \\t;]+)(.*)");
  if (RE2::FullMatch(code_part, pattern, &prefix, &old_val, &suffix)) {
    return prefix + new_val + suffix + comment_part;
  }
  return std::nullopt;
}

bool ProcessFile(const std::string& filepath) {
  std::vector<std::string> lines = ReadFileLines(filepath);
  if (lines.empty()) {
    LOG(ERROR) << "File is empty or could not be opened: " << filepath;
    return false;
  }

  std::vector<FindSeedBlock> blocks;
  for (size_t i = 0; i < lines.size(); ++i) {
    std::optional<size_t> marker_pos = FindMarkerPosition(lines[i]);
    if (marker_pos.has_value()) {
      CommentBlock comment_block = CollectContinuationComments(lines, i);
      FindSeedBlock block;
      block.start_line = i;
      block.comment_contents.push_back(lines[i].substr(*marker_pos));
      block.comment_contents.insert(block.comment_contents.end(),
                                    comment_block.contents.begin(),
                                    comment_block.contents.end());

      DeclarationLocations locs =
          FindDeclarations(lines, comment_block.end_line);
      block.kseed_line = locs.kseed_line;
      block.ktable_line = locs.ktable_line;

      if (!block.kseed_line || !block.ktable_line) {
        LOG(WARNING) << "Marker block starting at line " << (i + 1)
                     << " is missing matching variable declarations "
                        "(kSeed or kTableSize).";
      } else {
        blocks.push_back(block);
      }
      i = comment_block.end_line - 1;
    }
  }

  if (blocks.empty()) {
    LOG(ERROR) << "No FIND_SEED blocks found in " << filepath;
    return false;
  }

  for (const auto& block : blocks) {
    std::string arg_string;
    for (const auto& c : block.comment_contents) {
      arg_string += " " + c;
    }
    std::vector<std::string> tokens = ParseTokens(arg_string);
    BlockArgs parsed = ParseBlockArgs(tokens);
    auto res = FindSeedForKeys(parsed.keys, parsed.options);
    if (!res.has_value()) {
      LOG(ERROR) << "Failed to find seed for block starting at line "
                 << block.start_line + 1;
      return false;
    }

    auto new_seed_line = UpdateDeclarationLine(
        lines[*block.kseed_line], "kSeed", std::to_string(res->first));
    if (!new_seed_line.has_value()) {
      LOG(ERROR) << "Failed to update kSeed declaration on line "
                 << *block.kseed_line + 1;
      return false;
    }
    lines[*block.kseed_line] = *new_seed_line;

    auto new_table_line = UpdateDeclarationLine(
        lines[*block.ktable_line], "kTableSize", std::to_string(res->second));
    if (!new_table_line.has_value()) {
      LOG(ERROR) << "Failed to update kTableSize declaration on line "
                 << *block.ktable_line + 1;
      return false;
    }
    lines[*block.ktable_line] = *new_table_line;
  }

  return WriteFileLines(filepath, lines);
}

int main(int argc, char* argv[]) {
  absl::SetProgramUsageMessage(
      "Usage: find_seed --type=[type] [--case_sensitive] "
      "[--table_size=T] key1 key2 ...\n"
      "       find_seed --file=[filepath]");
  std::vector<char*> positional_args = absl::ParseCommandLine(argc, argv);

  std::string file = absl::GetFlag(FLAGS_file);
  if (!file.empty()) {
    if (!ProcessFile(file)) {
      return 1;
    }
    return 0;
  }

  if (positional_args.size() <= 1) {
    std::cerr << absl::ProgramUsageMessage() << "\n";
    return 1;
  }

  std::vector<std::string> keys;
  for (size_t i = 1; i < positional_args.size(); ++i) {
    keys.push_back(positional_args[i]);
  }

  FindSeedOptions opts = OptionsFromFlags();

  auto result = FindSeedForKeys(keys, opts);
  if (!result.has_value()) {
    LOG(ERROR) << "Failed to find seed.";
    return 1;
  }
  std::cout << "Found seed: " << result->first
            << " for table_size: " << result->second << "\n";

  return 0;
}
