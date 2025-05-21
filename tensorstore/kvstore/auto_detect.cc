// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/auto_detect.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/btree_set.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_kvstore {
namespace {

struct AutoDetectRegistry {
  absl::Mutex mutex;
  size_t prefix_length ABSL_GUARDED_BY(mutex) = 0;
  size_t suffix_length ABSL_GUARDED_BY(mutex) = 0;
  absl::btree_set<std::string> filenames ABSL_GUARDED_BY(mutex);
  std::vector<AutoDetectFileMatcher> file_matchers ABSL_GUARDED_BY(mutex);
  std::vector<AutoDetectDirectoryMatcher> directory_matchers
      ABSL_GUARDED_BY(mutex);
};

AutoDetectRegistry& GetAutoDetectRegistry() {
  static absl::NoDestructor<AutoDetectRegistry> registry;
  return *registry;
}

std::pair<size_t, size_t> GetFilePrefixAndSuffixLength() {
  size_t prefix_length, suffix_length;
  auto& registry = GetAutoDetectRegistry();
  absl::ReaderMutexLock lock(&registry.mutex);
  prefix_length = registry.prefix_length;
  suffix_length = registry.suffix_length;
  return {prefix_length, suffix_length};
}

template <auto MatchersMember, typename Options>
std::vector<AutoDetectMatch> EvaluateFileOrDirectoryMatches(
    const Options& options) {
  std::vector<AutoDetectMatch> matches;
  auto& registry = GetAutoDetectRegistry();
  absl::ReaderMutexLock lock(&registry.mutex);
  for (const auto& matcher : registry.*MatchersMember) {
    auto cur_matches = matcher(options);
    if (matches.empty()) {
      matches = std::move(cur_matches);
    } else {
      matches.insert(matches.end(), cur_matches.begin(), cur_matches.end());
    }
  }
  return matches;
}

std::vector<AutoDetectMatch> EvaluateMatches(
    const AutoDetectFileOptions& options) {
  return EvaluateFileOrDirectoryMatches<&AutoDetectRegistry::file_matchers>(
      options);
}

std::vector<AutoDetectMatch> EvaluateMatches(
    const AutoDetectDirectoryOptions& options) {
  return EvaluateFileOrDirectoryMatches<
      &AutoDetectRegistry::directory_matchers>(options);
}

struct AutoDetectOperationState {
  explicit AutoDetectOperationState(KvStore&& base) : base(std::move(base)) {}
  using Ptr = std::unique_ptr<AutoDetectOperationState>;
  Executor executor;
  KvStore base;
  absl::Time time = absl::Now();

  absl::Status error;

  using Value = std::vector<AutoDetectMatch>;

  static Future<Value> Start(Executor&& executor, KvStore&& base) {
    auto [promise, future] = PromiseFuturePair<Value>::Make();
    auto state = std::make_unique<AutoDetectOperationState>(std::move(base));
    state->executor = std::move(executor);
    if (state->base.path.empty() || state->base.path.back() == '/') {
      MaybeDetectDirectoryFormat(std::move(state), std::move(promise));
    } else {
      MaybeDetectFileFormat(std::move(state), std::move(promise));
    }
    return std::move(future);
  }

  void SetError(const absl::Status& error, std::string_view path) {
    if (!this->error.ok() || error.ok()) return;
    this->error = base.driver->AnnotateError(
        tensorstore::StrCat(base.path, path), "reading", error);
  }

  static void MaybeDetectFileFormat(Ptr self, Promise<Value> promise) {
    auto [prefix_length, suffix_length] = GetFilePrefixAndSuffixLength();

    if (prefix_length == 0 && suffix_length == 0) {
      // No file formats registered, skip to directory format detection.
      MaybeDetectDirectoryFormat(std::move(self), std::move(promise));
      return;
    }

    Future<kvstore::ReadResult> prefix_future;
    Future<kvstore::ReadResult> suffix_future;

    {
      auto batch = Batch::New();
      if (prefix_length != 0) {
        kvstore::ReadOptions options;
        options.byte_range = OptionalByteRangeRequest(0, prefix_length);
        options.staleness_bound = self->time;
        options.batch = batch;
        prefix_future = kvstore::Read(self->base, "", std::move(options));
      } else {
        prefix_future = kvstore::ReadResult{};
      }

      if (suffix_length != 0) {
        kvstore::ReadOptions options;
        options.byte_range =
            OptionalByteRangeRequest::SuffixLength(suffix_length);
        options.staleness_bound = self->time;
        options.batch = batch;
        suffix_future = kvstore::Read(self->base, "", std::move(options));
      } else {
        suffix_future = kvstore::ReadResult{};
      }
    }

    auto& self_ref = *self;
    Link(WithExecutor(
             self_ref.executor,
             [self = std::move(self)](
                 Promise<Value> promise,
                 ReadyFuture<kvstore::ReadResult> prefix_read_future,
                 ReadyFuture<kvstore::ReadResult> suffix_read_future) mutable {
               HandlePrefixSuffixReadResults(
                   std::move(self), std::move(promise),
                   prefix_read_future.result(), suffix_read_future.result(),
                   /*retry_on_out_of_range=*/true);
             }),
         std::move(promise), std::move(prefix_future),
         std::move(suffix_future));
  }

  static void HandlePrefixSuffixReadResults(
      Ptr self, Promise<Value> promise,
      const Result<kvstore::ReadResult>& prefix_read_result,
      const Result<kvstore::ReadResult>& suffix_read_result,
      bool retry_on_out_of_range) {
    // Byte range may have been too large.  Retry, requesting the
    // entire file this time.
    if (retry_on_out_of_range &&
        (absl::IsOutOfRange(prefix_read_result.status()) ||
         absl::IsOutOfRange(suffix_read_result.status()))) {
      kvstore::ReadOptions options;
      options.staleness_bound = self->time;
      auto future = kvstore::Read(self->base, "", std::move(options));
      auto& self_ref = *self;
      Link(WithExecutor(
               self_ref.executor,
               [self = std::move(self)](
                   Promise<Value> promise,
                   ReadyFuture<kvstore::ReadResult> read_future) mutable {
                 HandlePrefixSuffixReadResults(
                     std::move(self), std::move(promise), read_future.result(),
                     read_future.result(),
                     /*retry_on_out_of_range=*/false);
               }),
           std::move(promise), std::move(future));
      return;
    }

    AutoDetectFileOptions file_options;

    if (prefix_read_result.ok() && prefix_read_result->has_value()) {
      file_options.prefix = prefix_read_result->value;
    } else {
      // Save the error, if present, for reporting later in
      // case auto-detection fails.
      //
      // Some kvstores may report "not found" as an error.
      self->SetError(prefix_read_result.status(), "");
    }

    if (suffix_read_result.ok() && suffix_read_result->has_value()) {
      file_options.suffix = suffix_read_result->value;
      // suffix_read_result might be "not found" even though
      // "prefix_read_result" returned data.  This
      // inconsistent response could indicate a race
      // condition, e.g. file was created or deleted
      // concurrently with auto-detection.  Just ignore it,
      // though.
    } else {
      // Some kvstores may not support suffix reads; in that
      // case, proceed but save the error for reporting
      // later in case auto-detection fails.
      self->SetError(suffix_read_result.status(), "");
    }

    if (file_options.prefix.empty() && file_options.suffix.empty()) {
      // Skip to directory format detection.
      MaybeDetectDirectoryFormat(std::move(self), std::move(promise));
      return;
    }

    self->SetMatches(std::move(promise), EvaluateMatches(file_options));
  }

  static void MaybeDetectDirectoryFormat(Ptr self, Promise<Value> promise) {
    absl::btree_set<std::string> filenames;
    {
      auto& registry = GetAutoDetectRegistry();
      absl::ReaderMutexLock lock(&registry.mutex);
      filenames = registry.filenames;
    }
    if (filenames.empty()) {
      self->SetMatches(std::move(promise), {});
      return;
    }
    internal::EnsureDirectoryPath(self->base.path);
    std::vector<Future<kvstore::ReadResult>> read_futures;
    read_futures.reserve(filenames.size());
    auto [all_promise, all_future] =
        PromiseFuturePair<void>::Make(absl::OkStatus());
    {
      auto batch = Batch::New();
      for (const auto& filename : filenames) {
        kvstore::ReadOptions options;
        options.staleness_bound = self->time;
        options.byte_range = OptionalByteRangeRequest::Stat();
        options.batch = batch;
        read_futures.push_back(
            kvstore::Read(self->base, filename, std::move(options)));
        // Create a link to prevent `promise` from becoming ready
        // until all read futures become ready.
        Link([](Promise<void> promise,
                ReadyFuture<kvstore::ReadResult> future) {},
             all_promise, read_futures.back());
      }
    }
    auto& self_ref = *self;
    Link(WithExecutor(
             self_ref.executor,
             [self = std::move(self), filenames = std::move(filenames),
              read_futures = std::move(read_futures)](
                 Promise<Value> promise, ReadyFuture<void> future) mutable {
               if (auto status = future.status(); !status.ok()) {
                 promise.SetResult(std::move(status));
                 return;
               }
               auto filename_it = filenames.begin();
               for (const auto& future : read_futures) {
                 auto& result = future.result();
                 if (result && result->has_value()) {
                   ++filename_it;
                 } else {
                   self->SetError(result.status(), *filename_it);
                   filename_it = filenames.erase(filename_it);
                 }
               }

               AutoDetectDirectoryOptions directory_options;
               directory_options.filenames = std::move(filenames);
               self->SetMatches(std::move(promise),
                                EvaluateMatches(directory_options));
             }),
         std::move(promise), std::move(all_future));
  }

  void SetMatches(Promise<Value> promise, Value&& matches) {
    if (matches.empty() && !error.ok()) {
      promise.SetResult(tensorstore::MaybeAnnotateStatus(
          error, "Format auto-detection failed"));
      return;
    }

    promise.SetResult(std::move(matches));
  }
};

}  // namespace

AutoDetectDirectorySpec AutoDetectDirectorySpec::SingleFile(
    std::string_view scheme, std::string_view filename) {
  AutoDetectDirectorySpec spec;
  spec.filenames.insert(std::string(filename));
  spec.match = [filename = std::string(filename), scheme = std::string(scheme)](
                   const AutoDetectDirectoryOptions& options) {
    std::vector<AutoDetectMatch> matches;
    if (options.filenames.count(filename)) {
      matches.push_back(AutoDetectMatch{scheme});
    }
    return matches;
  };
  return spec;
}

AutoDetectFileSpec AutoDetectFileSpec::PrefixSignature(
    std::string_view scheme, std::string_view signature) {
  return PrefixSignature(
      scheme, signature.size(),
      [signature = std::string(signature)](std::string_view prefix) {
        return prefix == signature;
      });
}

AutoDetectFileSpec AutoDetectFileSpec::PrefixSignature(
    std::string_view scheme, size_t signature_length,
    std::function<bool(std::string_view signature)> predicate) {
  AutoDetectFileSpec spec;
  spec.prefix_length = signature_length;
  spec.match = [signature_length, predicate = std::move(predicate),
                scheme =
                    std::string(scheme)](const AutoDetectFileOptions& options) {
    std::vector<AutoDetectMatch> matches;
    if (options.prefix.size() >= signature_length) {
      auto prefix = options.prefix.Subcord(0, signature_length);
      if (predicate(prefix.Flatten())) {
        matches.push_back(AutoDetectMatch{scheme});
      }
    }
    return matches;
  };
  return spec;
}

AutoDetectFileSpec AutoDetectFileSpec::SuffixSignature(
    std::string_view scheme, std::string_view signature) {
  return SuffixSignature(
      scheme, signature.size(),
      [signature = std::string(signature)](std::string_view suffix) {
        return suffix == signature;
      });
}

AutoDetectFileSpec AutoDetectFileSpec::SuffixSignature(
    std::string_view scheme, size_t signature_length,
    std::function<bool(std::string_view signature)> predicate) {
  AutoDetectFileSpec spec;
  spec.suffix_length = signature_length;
  spec.match = [signature_length, predicate = std::move(predicate),
                scheme =
                    std::string(scheme)](const AutoDetectFileOptions& options) {
    std::vector<AutoDetectMatch> matches;
    if (options.suffix.size() >= signature_length) {
      auto suffix = options.suffix.Subcord(
          options.suffix.size() - signature_length, signature_length);
      if (predicate(suffix.Flatten())) {
        matches.push_back(AutoDetectMatch{scheme});
      }
    }
    return matches;
  };
  return spec;
}

AutoDetectRegistration::AutoDetectRegistration(AutoDetectFileSpec&& file_spec) {
  auto& registry = GetAutoDetectRegistry();
  absl::MutexLock lock(&registry.mutex);
  registry.prefix_length =
      std::max(registry.prefix_length, file_spec.prefix_length);
  registry.suffix_length =
      std::max(registry.suffix_length, file_spec.suffix_length);
  registry.file_matchers.push_back(std::move(file_spec.match));
}

AutoDetectRegistration::AutoDetectRegistration(
    AutoDetectDirectorySpec&& directory_spec) {
  auto& registry = GetAutoDetectRegistry();
  absl::MutexLock lock(&registry.mutex);
  for (auto& filename : directory_spec.filenames) {
    registry.filenames.insert(std::move(filename));
  }
  registry.directory_matchers.push_back(std::move(directory_spec.match));
}

void AutoDetectRegistration::ClearRegistrations() {
  auto& registry = GetAutoDetectRegistry();
  absl::MutexLock lock(&registry.mutex);
  registry.filenames.clear();
  registry.directory_matchers.clear();
  registry.file_matchers.clear();
  registry.prefix_length = 0;
  registry.suffix_length = 0;
}

Future<std::vector<AutoDetectMatch>> AutoDetectFormat(Executor executor,
                                                      KvStore base) {
  return AutoDetectOperationState::Start(std::move(executor), std::move(base));
}

}  // namespace internal_kvstore
}  // namespace tensorstore
