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

#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"

#include <memory>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"
#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"


namespace tensorstore {
namespace internal_tiff_kvstore {

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff");

struct ReadDirectoryOp : public internal::AtomicReferenceCount<ReadDirectoryOp> {
  TiffDirectoryCache::Entry* entry_;
  std::shared_ptr<const TiffDirectoryParseResult> existing_read_data_;
  kvstore::ReadOptions options_;
  bool is_full_read_;

  void StartRead() {
    auto& cache = internal::GetOwningCache(*entry_);
    ABSL_LOG_IF(INFO, tiff_logging)
        << "StartRead " << entry_->key();
    
    // 1.  Default to the "slice‑first" strategy -----------------------------
    is_full_read_ = false;

    // Honour any *caller‑supplied* range that is smaller than the slice.
    if (!options_.byte_range.IsFull() &&
        options_.byte_range.size() <= kInitialReadBytes) {
      // Caller already requested an explicit (small) range → keep it.
    } else {
      // Otherwise issue our standard 0‑1023 probe.
      options_.byte_range =
          OptionalByteRangeRequest::Range(0, kInitialReadBytes);
    }

    auto future = cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
            ReadyFuture<kvstore::ReadResult> ready) {
          self->OnReadComplete(std::move(ready));
        });
  }

  void OnReadComplete(ReadyFuture<kvstore::ReadResult> ready) {
    auto& r = ready.result();
    if (!r.ok()) {
      // If the ranged request overshot the file, retry with a full read.
      if (!is_full_read_ && absl::IsOutOfRange(r.status())) {
      is_full_read_ = true;
      options_.byte_range = {};  // Full read.
      auto retry_future =
          internal::GetOwningCache(*entry_).kvstore_driver_->Read(
              std::string(entry_->key()), options_);
      retry_future.Force();
      retry_future.ExecuteWhenReady(
          [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
              ReadyFuture<kvstore::ReadResult> f) {
              self->OnReadComplete(std::move(f));
          });
      return;
      }
      entry_->ReadError(internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    auto& read_result = *r;
    if (read_result.not_found()) {
      entry_->ReadError(absl::NotFoundError(""));
      return;
    }

    if (read_result.aborted()) {
      // Return existing data if we have it
      if (existing_read_data_) {
        entry_->ReadSuccess(TiffDirectoryCache::ReadState{
            existing_read_data_,
            std::move(read_result.stamp)
        });
        return;
      }
      entry_->ReadError(absl::AbortedError("Read aborted"));
      return;
    }

    auto parse_result = std::make_shared<TiffDirectoryParseResult>();
    parse_result->raw_data = std::move(read_result.value);
    // If we asked for a slice but got fewer than requested bytes,
    // we effectively have the whole file.
    if (!is_full_read_ &&
          parse_result->raw_data.size() < internal_tiff_kvstore::kInitialReadBytes) {
        parse_result->full_read = true;
      } else {
        parse_result->full_read = is_full_read_;
      }

    // Create a riegeli reader from the cord
    riegeli::CordReader cord_reader(&parse_result->raw_data);
    
    // Parse TIFF header
    Endian endian;
    uint64_t first_ifd_offset;
    auto status = ParseTiffHeader(cord_reader, endian, first_ifd_offset);
    if (!status.ok()) {
      entry_->ReadError(status);
      return;
    }
    
    // Store the endian in the parse result for use when reading external arrays
    parse_result->endian = endian;
    
    // Parse TIFF directory at the given offset
    TiffDirectory directory;
    status = ParseTiffDirectory(
        cord_reader, endian, first_ifd_offset, 
        parse_result->raw_data.size() - first_ifd_offset, directory);
    if (!status.ok()) {
      entry_->ReadError(status);
      return;
    }
    
    // Store the IFD entries
    parse_result->ifd_entries = std::move(directory.entries);
    
    // Parse the ImageDirectory from the IFD entries
    status = ParseImageDirectory(parse_result->ifd_entries, parse_result->image_directory);
    if (!status.ok()) {
      entry_->ReadError(status);
      return;
    }
    
    // Check if we need to load external arrays
    bool has_external_arrays = false;
    for (const auto& entry : parse_result->ifd_entries) {
      if (entry.is_external_array) {
        has_external_arrays = true;
        break;
      }
    }
    
    if (has_external_arrays) {
      // Load external arrays before completing the cache read
      auto future = entry_->LoadExternalArrays(parse_result, read_result.stamp);
      future.Force();
      
      // Once the external arrays are loaded, complete the cache read
      future.ExecuteWhenReady(
          [self = internal::IntrusivePtr<ReadDirectoryOp>(this), 
           parse_result, 
           stamp = std::move(read_result.stamp)](ReadyFuture<void> future) mutable {
            auto& r = future.result();
            if (!r.ok()) {
              // If external arrays couldn't be loaded, propagate the error
              self->entry_->ReadError(r.status());
              return;
            }
            
            // External arrays loaded successfully
            self->entry_->ReadSuccess(TiffDirectoryCache::ReadState{
                std::move(parse_result),
                std::move(stamp)
            });
          });
    } else {
      // No external arrays to load
      entry_->ReadSuccess(TiffDirectoryCache::ReadState{
          std::move(parse_result),
          std::move(read_result.stamp)
      });
    }
  }
};

}  // namespace

Future<void> TiffDirectoryCache::Entry::LoadExternalArrays(
    std::shared_ptr<TiffDirectoryParseResult> parse_result,
    tensorstore::TimestampedStorageGeneration stamp) {
  
  // Get references to the arrays that might need loading
  auto& entries = parse_result->ifd_entries;
  auto& img_dir = parse_result->image_directory;
  
  // Collect all external arrays that need to be loaded
  struct ExternalArrayInfo {
    Tag tag;
    TiffDataType type;
    uint64_t offset;
    uint64_t count;
    std::vector<uint64_t>* output_array;
  };
  
  std::vector<ExternalArrayInfo> external_arrays;
  
  // Check for strip and tile arrays that need to be loaded
  for (const auto& entry : entries) {
    if (!entry.is_external_array) continue;
    
    switch (entry.tag) {
      case Tag::kStripOffsets:
        external_arrays.push_back({entry.tag, entry.type, entry.value_or_offset, 
                                 entry.count, &img_dir.strip_offsets});
        break;
      case Tag::kStripByteCounts:
        external_arrays.push_back({entry.tag, entry.type, entry.value_or_offset, 
                                 entry.count, &img_dir.strip_bytecounts});
        break;
      case Tag::kTileOffsets:
        external_arrays.push_back({entry.tag, entry.type, entry.value_or_offset, 
                                 entry.count, &img_dir.tile_offsets});
        break;
      case Tag::kTileByteCounts:
        external_arrays.push_back({entry.tag, entry.type, entry.value_or_offset, 
                                 entry.count, &img_dir.tile_bytecounts});
        break;
      default:
        // Other external arrays aren't needed for the image directory
        break;
    }
  }
  
  // If no external arrays to load, return immediately
  if (external_arrays.empty()) {
    return MakeReadyFuture<void>();
  }
  
  ABSL_LOG_IF(INFO, tiff_logging)
    << "Loading " << external_arrays.size() << " external arrays";
  
  // Create a Promise/Future pair to track completion of all array loads
  auto [promise, future] = PromiseFuturePair<void>::Make();
  auto& cache = internal::GetOwningCache(*this);
  
  // Track the number of array loads that remain to be processed
  struct LoadState : public internal::AtomicReferenceCount<LoadState> {
    size_t remaining_count;
    absl::Status status;
    Promise<void> promise;
    
    explicit LoadState(size_t count, Promise<void> promise) 
        : remaining_count(count), promise(std::move(promise)) {}
    
    void CompleteOne(absl::Status s) {
      if (!s.ok() && status.ok()) {
        status = s;  // Store the first error encountered
      }
      
      if (--remaining_count == 0) {
        // All operations complete, resolve the promise
        if (status.ok()) {
          promise.SetResult(absl::OkStatus());
        } else {
          promise.SetResult(status);
        }
      }
    }
  };
  
  auto load_state = internal::MakeIntrusivePtr<LoadState>(
      external_arrays.size(), std::move(promise));
  
  // Load each external array
  for (const auto& array_info : external_arrays) {
    // Calculate the byte range needed for this array
    size_t element_size = GetTiffDataTypeSize(array_info.type);
    size_t byte_count = array_info.count * element_size;
    
    // Set up the read options
    kvstore::ReadOptions options;
    options.generation_conditions.if_equal = stamp.generation;
    options.byte_range = OptionalByteRangeRequest::Range(
        array_info.offset, array_info.offset + byte_count);
    
    ABSL_LOG_IF(INFO, tiff_logging)
      << "Reading external array for tag " << static_cast<int>(array_info.tag)
      << " at offset " << array_info.offset << " size " << byte_count;
    
    // Issue the read request and track the future
    auto read_future = cache.kvstore_driver_->Read(std::string(this->key()), options);
    read_future.Force();
    
    // Process the read result when ready
    read_future.ExecuteWhenReady(
        [state = load_state, array_info, endian = parse_result->endian](
            ReadyFuture<kvstore::ReadResult> ready) {
          auto& r = ready.result();
          if (!r.ok()) {
            state->CompleteOne(internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
            return;
          }
          
          auto& read_result = *r;
          if (read_result.not_found() || read_result.aborted()) {
            state->CompleteOne(absl::DataLossError(
                "External array not found or read aborted"));
            return;
          }
          
          // Create a reader for the data
          riegeli::CordReader cord_reader(&read_result.value);
          ABSL_LOG_IF(INFO, tiff_logging)
              << "Parsing external array for tag " << static_cast<int>(array_info.tag)
              << " at offset " << array_info.offset << " size " << read_result.value.size();
              
          // Parse the external array
          auto status = ParseExternalArray(
              cord_reader, endian, 0, array_info.count, 
              array_info.type, *array_info.output_array);
              
          // Complete this array load operation
          state->CompleteOne(status);
        });
  }
  
  // Return the future that completes when all array loads are finished
  return future;
}

size_t TiffDirectoryCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return static_cast<const ReadData*>(read_data)->raw_data.size();
}

void TiffDirectoryCache::Entry::DoRead(AsyncCacheReadRequest request) {
  auto state = internal::MakeIntrusivePtr<ReadDirectoryOp>();
  state->entry_ = this;
  state->options_.staleness_bound = request.staleness_bound;
  {
    ReadLock<ReadData> lock(*this);
    state->existing_read_data_ = lock.shared_data();
    state->options_.generation_conditions.if_not_equal = 
        lock.read_state().stamp.generation;
  }

  state->StartRead();
}

TiffDirectoryCache::Entry* TiffDirectoryCache::DoAllocateEntry() {
  return new Entry;
}

size_t TiffDirectoryCache::DoGetSizeofEntry() {
  return sizeof(Entry);
}

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore