// Copyright 2024 The TensorStore Authors
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

#include "tensorstore/internal/metrics/histogram.h"

#include <stddef.h>

#include <cassert>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/strings/str_cat.h"

namespace tensorstore {
namespace internal_metrics {
namespace {

static const absl::NoDestructor<std::vector<std::string>> kDefaultBucketLabels(
    []() {
      std::vector<std::string> labels;
      labels.push_back("0");
      for (size_t i = 0; i < 19; i++) {
        labels.push_back(absl::StrCat(1ull << i));
      }
      // Values of 1024*1024 or greater use si suffixes.
      for (auto suffix : {"M", "G", "T", "P", "E"}) {
        for (size_t i = 0; i < 9; i++) {
          labels.push_back(absl::StrCat(1ull << i, suffix));
          if (labels.size() == DefaultBucketer::OverflowBucket) break;
        }
        if (labels.size() == DefaultBucketer::OverflowBucket) break;
      }
      return labels;
    }());

}  // namespace

std::string_view DefaultBucketer::LabelForBucket(size_t b) {
  assert(b < DefaultBucketer::Max);
  if (b < kDefaultBucketLabels->size()) return (*kDefaultBucketLabels)[b];
  return "Inf";
}

void DefaultBucketer::SetHistogramLabels(
    std::vector<std::string_view>& labels) {
  labels = std::vector<std::string_view>(kDefaultBucketLabels->begin(),
                                         kDefaultBucketLabels->end());
}

}  // namespace internal_metrics
}  // namespace tensorstore
