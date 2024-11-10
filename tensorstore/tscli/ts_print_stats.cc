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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/tscli/args.h"
#include "tensorstore/tscli/cli.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/json_absl_flag.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace cli {
namespace {

// Greedily merges adjacent boxes in the array.
// The basic greedy algorithm is to sort the boxes by each position, and then
// attempt to merge any boxes which are adjacent along a single dimension,
// repeating until no more merges are possible.
std::vector<Box<>> MergeAdjacentBoxes(std::vector<Box<>> boxes) {
  if (boxes.empty()) return boxes;
  bool merged = false;
  auto rank = boxes[0].rank();
  for (auto& box : boxes) {
    rank = std::max(rank, box.rank());
  }
  if (rank == 0) return boxes;

  auto merge_if_adjacent = [&](const Box<>& left,
                               const Box<>& right) -> std::optional<Box<>> {
    if (left.rank() != right.rank()) return std::nullopt;
    std::optional<size_t> dim_to_merge;
    std::vector<int64_t> shape(left.rank());
    for (size_t i = 0; i < left.rank(); ++i) {
      shape[i] = left.shape()[i];
      if (left.origin()[i] == right.origin()[i]) continue;
      if (left.origin()[i] + left.shape()[i] == right.origin()[i]) {
        if (!dim_to_merge.has_value()) {
          dim_to_merge = i;
          shape[i] = left.shape()[i] + right.shape()[i];
          continue;
        }
      }
      return std::nullopt;
    }
    return Box<>(left.origin(), shape);
  };

  std::vector<Box<>> result = std::move(boxes);

  do {
    merged = false;
    // Sort boxes by each rank.
    for (size_t i = 0; i < rank; ++i) {
      boxes = std::move(result);
      std::sort(boxes.begin(), boxes.end(),
                [i](const Box<>& a, const Box<>& b) {
                  if (a.rank() != b.rank()) {
                    return a.rank() < b.rank();
                  }
                  if (std::equal(a.origin().begin(), a.origin().end(),
                                 b.origin().begin())) {
                    return std::lexicographical_compare(
                        a.shape().begin(), a.shape().end(), b.shape().begin(),
                        b.shape().end());
                  }
                  if (i < a.rank() && a.origin()[i] != b.origin()[i]) {
                    return a.origin()[i] < b.origin()[i];
                  }
                  return std::lexicographical_compare(
                      a.origin().begin(), a.origin().end(), b.origin().begin(),
                      b.origin().end());
                });
      // Merge adjacent boxes.
      result = {};
      auto it = boxes.begin();
      result.push_back(std::move(*it));
      for (++it; it != boxes.end(); ++it) {
        auto box_maybe = merge_if_adjacent(result.back(), *it);
        if (box_maybe.has_value()) {
          result.back() = *std::move(box_maybe);
          merged = true;
        } else {
          result.push_back(std::move(*it));
        }
      }
    }
  } while (merged);

  // Sort the final set of merged boxes to ensure consistent ordering.
  std::sort(result.begin(), result.end(), [](const Box<>& a, const Box<>& b) {
    if (a.rank() != b.rank()) return a.rank() < b.rank();
    if (std::equal(a.origin().begin(), a.origin().end(), b.origin().begin())) {
      return std::lexicographical_compare(a.shape().begin(), a.shape().end(),
                                          b.shape().begin(), b.shape().end());
    } else {
      return std::lexicographical_compare(a.origin().begin(), a.origin().end(),
                                          b.origin().begin(), b.origin().end());
    }
  });
  return result;
}

// Queries the storage statistics for each possible stored chunk within the
// specified tensorstore.
// NOTE: On a large dataset this may be very expensive.
std::vector<Box<>> GetStoredChunks(tensorstore::TensorStore<> ts) {
  size_t rank = ts.rank();

  std::vector<int64_t> chunk_shape(rank);
  std::vector<int64_t> grid_shape(rank);
  std::vector<int64_t> grid_pos(rank);

  for (size_t i = 0; i < rank; ++i) {
    chunk_shape[i] = ts.chunk_layout()->read_chunk_shape()[i];
    grid_shape[i] =
        tensorstore::CeilOfRatio(ts.domain().shape()[i], chunk_shape[i]);
  }

  auto [stat_promise, stat_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());

  absl::Mutex mutex;
  std::vector<Box<>> array_boxes;
  std::vector<int64_t> slice_start(rank), slice_shape(rank);
  size_t count = 0;
  do {
    for (size_t i = 0; i < rank; ++i) {
      slice_start[i] = ts.domain().origin()[i] + (chunk_shape[i] * grid_pos[i]);
      slice_shape[i] =
          std::min(chunk_shape[i], ts.domain().shape()[i] - slice_start[i]);
    }
    if (count++ % 500 == 499) {
      std::cerr << "Issuing query " << count << std::endl;
    }
    const auto box = Box(slice_start, slice_shape);
    LinkValue(
        [b = box, &array_boxes, &mutex](auto promise,
                                        ReadyFuture<ArrayStorageStatistics> f) {
          absl::MutexLock lock(&mutex);
          if (!f.value().not_stored) {
            array_boxes.push_back(std::move(b));
          }
        },
        stat_promise,
        GetStorageStatistics(
            ts | tensorstore::AllDims().BoxSlice(box),
            tensorstore::ArrayStorageStatistics::query_not_stored));
  } while (tensorstore::internal::AdvanceIndices(rank, grid_pos.data(),
                                                 grid_shape.data()));
  if (count > 1000) {
    std::cerr << "Waiting for " << count << " queries." << std::endl;
  }
  stat_promise = {};
  stat_future.Wait();
  return MergeAdjacentBoxes(std::move(array_boxes));
}

void Output(tensorstore::Spec spec, bool brief,
            tensorstore::span<Box<>> not_stored,
            tensorstore::span<Box<>> partial,
            tensorstore::span<Box<>> fully_stored) {
  auto dump = [&](const Box<>& b) {
    if (brief) {
      return internal_json_binding::ToJson(b)->dump();
    }
    auto s = spec | tensorstore::AllDims().BoxSlice(b);
    return s->ToJson()->dump();
  };

  for (const auto& b : not_stored) {
    std::cout << "missing: " << dump(b) << std::endl;
  }
  for (const auto& b : partial) {
    std::cout << "partial: " << dump(b) << std::endl;
  }
  for (const auto& b : fully_stored) {
    std::cout << "present: " << dump(b) << std::endl;
  }
}

}  // namespace

absl::Status TsPrintStoredChunks(Context context, tensorstore::Spec spec,
                                 bool brief) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());

  auto array_boxes = GetStoredChunks(ts);
  Output(ts.spec().value(), brief, {}, {}, array_boxes);
  return absl::OkStatus();
}

absl::Status TsPrintStorageStatistics(Context context, tensorstore::Spec spec,
                                      tensorstore::span<Box<>> boxes,
                                      bool brief) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto ts,
      tensorstore::Open(spec, context, tensorstore::ReadWriteMode::read,
                        tensorstore::OpenMode::open)
          .result());

  absl::Mutex mutex;
  std::vector<Box<>> fully_stored;
  std::vector<Box<>> not_stored;
  std::vector<Box<>> partial;

  auto [stat_promise, stat_future] =
      PromiseFuturePair<void>::Make(absl::OkStatus());
  for (const auto& box : boxes) {
    LinkValue(
        [&, b = box](auto promise, ReadyFuture<ArrayStorageStatistics> f) {
          absl::MutexLock lock(&mutex);
          if (f.value().not_stored) {
            not_stored.push_back(std::move(b));
          } else if (f.value().fully_stored) {
            fully_stored.push_back(std::move(b));
          } else {
            partial.push_back(std::move(b));
          }
        },
        stat_promise,
        GetStorageStatistics(
            ts | tensorstore::AllDims().BoxSlice(box),
            tensorstore::ArrayStorageStatistics::query_not_stored,
            tensorstore::ArrayStorageStatistics::query_fully_stored));
  }
  stat_promise = {};
  stat_future.Wait();
  Output(ts.spec().value(), brief, not_stored, partial, fully_stored);
  return absl::OkStatus();
}

absl::Status RunTsPrintStorageStatistics(Context::Spec context_spec,
                                         CommandFlags flags) {
  tensorstore::JsonAbslFlag<std::optional<tensorstore::Spec>> spec;
  bool brief = true;
  std::vector<LongOption> long_options({
      LongOption{"--spec",
                 [&](std::string_view value) {
                   std::string error;
                   if (!AbslParseFlag(value, &spec, &error)) {
                     return absl::InvalidArgumentError(error);
                   }
                   return absl::OkStatus();
                 }},
  });
  std::vector<BoolOption> bool_options({
      BoolOption{"--full", [&]() { brief = false; }},
      BoolOption{"-f", [&]() { brief = false; }},
      BoolOption{"--brief", [&]() { brief = true; }},
      BoolOption{"-b", [&]() { brief = true; }},
  });

  TENSORSTORE_RETURN_IF_ERROR(
      TryParseOptions(flags, long_options, bool_options));

  tensorstore::Context context(context_spec);

  if (spec.value) {
    if (flags.positional_args.empty()) {
      return TsPrintStoredChunks(context, *spec.value, brief);
    }

    std::vector<Box<>> boxes;
    for (const std::string_view arg : flags.positional_args) {
      tensorstore::JsonAbslFlag<Box<>> box_flag;
      std::string error;
      if (!AbslParseFlag(arg, &box_flag, &error)) {
        std::cerr << "Invalid box: " << arg << ": " << error << std::endl;
        continue;
      }
      boxes.push_back(std::move(box_flag).value);
    }
    return TsPrintStorageStatistics(context, *spec.value, boxes, brief);
  }

  if (flags.positional_args.empty()) {
    return absl::InvalidArgumentError(
        "print_stats: Must include --spec or a sequence of specs");
  }

  absl::Status status;
  for (const std::string_view spec : flags.positional_args) {
    tensorstore::JsonAbslFlag<tensorstore::Spec> arg_spec;
    std::string error;
    if (AbslParseFlag(spec, &arg_spec, &error)) {
      status.Update(TsPrintStoredChunks(context, arg_spec.value, brief));
      continue;
    }
    std::cerr << "Invalid spec: " << spec << ": " << error << std::endl;
  }
  return status;
}

}  // namespace cli
}  // namespace tensorstore
