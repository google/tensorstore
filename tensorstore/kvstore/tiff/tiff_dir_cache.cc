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
#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/util/future.h"

#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_tiff_kvstore {

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff");

struct ReadDirectoryOp
    : public internal::AtomicReferenceCount<ReadDirectoryOp> {
  TiffDirectoryCache::Entry* entry_;
  std::shared_ptr<const TiffParseResult> existing_read_data_;
  kvstore::ReadOptions options_;

  // partial reads are needed.
  bool is_full_read_;

  // The resulting parse data we will build up. This includes raw file data, IFD
  // entries, etc.
  std::shared_ptr<TiffParseResult> parse_result_;

  // Buffer for storing raw file data during reading and parsing operations
  absl::Cord buffer;

  // The offset in the file that corresponds to buffer[0].
  uint64_t file_offset_;

  // The next IFD offset we expect to parse. If 0, we have no more IFDs in the
  // chain.
  uint64_t next_ifd_offset_;

  void StartTiffRead() {
    auto& cache = internal::GetOwningCache(*entry_);
    ABSL_LOG_IF(INFO, tiff_logging)
        << "StartTiffRead " << entry_->key()
        << " with byte range: " << options_.byte_range;

    is_full_read_ = false;
    file_offset_ = 0;  // We’re reading from the start.
    parse_result_ = std::make_shared<TiffParseResult>();

    // Honour any *caller‑supplied* range that is smaller than the slice.
    if (!options_.byte_range.IsFull() &&
        options_.byte_range.size() <= kInitialReadBytes) {
      // Caller already requested an explicit (small) range → keep it.
    } else {
      // Otherwise issue our standard 0‑1023 probe.
      options_.byte_range =
          OptionalByteRangeRequest::Range(0, kInitialReadBytes);
    }

    auto future =
        cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Issued initial read request for key: " << entry_->key()
        << " with byte range: " << options_.byte_range;
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
            ReadyFuture<kvstore::ReadResult> ready) {
          ABSL_LOG_IF(INFO, tiff_logging)
              << "Initial read completed for key: " << self->entry_->key();
          self->OnHeaderReadComplete(std::move(ready));
        });
  }

  // Called after the initial read completes (the read that tries to parse the
  // TIFF header).
  void OnHeaderReadComplete(ReadyFuture<kvstore::ReadResult> ready) {
    const auto& r = ready.result();
    ABSL_LOG_IF(INFO, tiff_logging)
        << "OnHeaderReadComplete called for key: " << entry_->key();

    if (!r.ok()) {
      ABSL_LOG_IF(WARNING, tiff_logging)
          << "Read failed with status: " << r.status();
      // Possibly partial read overshot the file
      if (!is_full_read_ && absl::IsOutOfRange(r.status())) {
        is_full_read_ = true;
        // Switch to a full read
        ABSL_LOG_IF(INFO, tiff_logging)
            << "Overshot file. Issuing a full read for key: " << entry_->key();
        options_.byte_range = {};
        auto& cache = internal::GetOwningCache(*entry_);
        auto retry_future =
            cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
        retry_future.Force();
        retry_future.ExecuteWhenReady(
            [self = internal::IntrusivePtr<ReadDirectoryOp>(this)](
                ReadyFuture<kvstore::ReadResult> f) {
              self->OnHeaderReadComplete(std::move(f));
            });
        return;
      }
      // Some other error
      entry_->ReadError(
          internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    if (r->not_found()) {
      ABSL_LOG_IF(WARNING, tiff_logging)
          << "File not found for key: " << entry_->key();
      entry_->ReadError(absl::NotFoundError("File not found"));
      return;
    }
    if (r->aborted()) {
      if (existing_read_data_) {
        // Return existing data
        ABSL_LOG_IF(INFO, tiff_logging)
            << "Read aborted, returning existing data for key: "
            << entry_->key();
        entry_->ReadSuccess(TiffDirectoryCache::ReadState{existing_read_data_,
                                                          std::move(r->stamp)});
      } else {
        entry_->ReadError(absl::AbortedError("Read aborted."));
      }
      return;
    }

    // We now have partial data at offsets [0..someSize).
    buffer = std::move(r->value);
    uint64_t bytes_received = buffer.size();

    // If we got less data than requested, treat it as a full read.
    if (!is_full_read_ && bytes_received < kInitialReadBytes) {
      parse_result_->full_read = true;
    } else {
      parse_result_->full_read = is_full_read_;
    }

    // Parse the header
    riegeli::CordReader cord_reader(&buffer);
    Endian endian;
    absl::Status header_status =
        ParseTiffHeader(cord_reader, endian, next_ifd_offset_);
    if (!header_status.ok()) {
      ABSL_LOG_IF(WARNING, tiff_logging)
          << "Failed to parse TIFF header: " << header_status;
      entry_->ReadError(header_status);
      return;
    }
    ABSL_LOG_IF(INFO, tiff_logging)
        << "TIFF header parsed successfully."
        << ", Next IFD offset: " << next_ifd_offset_;
    parse_result_->endian = endian;

    StartParsingIFDs(std::move(r->stamp));
  }

  /// This function begins (or continues) parsing IFDs at next_ifd_offset_ until
  /// we reach offset=0 or an error.
  void StartParsingIFDs(tensorstore::TimestampedStorageGeneration stamp) {
    if (next_ifd_offset_ == 0) {
      // No IFDs, so finalize
      OnAllIFDsDone(std::move(stamp));
      return;
    }

    absl::Status s = ParseOneIFD();
    if (absl::IsOutOfRange(s)) {
      // Means we need more data
      RequestMoreData(std::move(stamp));
      return;
    }
    if (!s.ok()) {
      // Some other error
      entry_->ReadError(s);
      return;
    }

    // If parse succeeded, check if the IFD we parsed gave us a new offset for
    // the next IFD.
    if (next_ifd_offset_ == 0) {
      OnAllIFDsDone(std::move(stamp));
      return;
    }

    // Parse the next IFD in the chain.
    StartParsingIFDs(std::move(stamp));
  }

  // This attempts to parse one IFD at next_ifd_offset_ using our current
  // buffer. If that offset is beyond the buffer range, returns OutOfRangeError.
  // If success, updates parse_result_, next_ifd_offset_.
  absl::Status ParseOneIFD() {
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Parsing IFD at offset: " << next_ifd_offset_
        << " for key: " << entry_->key();

    if (next_ifd_offset_ < file_offset_) {
      return absl::DataLossError(
          "IFD offset is behind our current buffer offset, which is "
          "unexpected.");
    }

    uint64_t relative_pos = next_ifd_offset_ - file_offset_;
    uint64_t buffer_size = buffer.size();

    if (relative_pos > buffer_size) {
      ABSL_LOG_IF(WARNING, tiff_logging)
          << "Buffer underflow while parsing IFD. Needed next_ifd_offset: "
          << relative_pos
          << ", Max available offset: " << file_offset_ + buffer_size;
      // We’re missing data
      return absl::OutOfRangeError(
          "Next IFD is outside our current buffer range.");
    }

    // Slice off everything before relative_pos, because we no longer need it.
    buffer = buffer.Subcord(relative_pos, buffer_size - relative_pos);
    file_offset_ = next_ifd_offset_;

    // Now parse from the beginning of buffer as offset=0 in the local sense.
    riegeli::CordReader reader(&buffer);
    TiffDirectory dir;
    absl::Status s = ParseTiffDirectory(reader, parse_result_->endian,
                                        /*local_offset=*/0, buffer.size(), dir);
    if (!s.ok()) {
      ABSL_LOG_IF(WARNING, tiff_logging) << "Failed to parse IFD: " << s;
      return s;  // Could be OutOfRange, parse error, etc.
    }

    // Store the IFD’s entries in parse_result_->ifd_entries (or directories).
    parse_result_->directories.push_back(dir);

    // Update next_ifd_offset_ to the directory’s next offset
    next_ifd_offset_ = dir.next_ifd_offset;
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Parsed IFD successfully. Next IFD offset: " << dir.next_ifd_offset;
    return absl::OkStatus();
  }

  /// If we discover we need more data to parse the next IFD, we read newer
  /// bytes from the file. Suppose we read from [file_offset_ + buffer.size(),
  /// file_offset_ + buffer.size() + chunk).
  void RequestMoreData(tensorstore::TimestampedStorageGeneration stamp) {
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Requesting more data for key: " << entry_->key()
        << ". Current buffer size: " << buffer.size()
        << ", Full read: " << parse_result_->full_read;
    if (parse_result_->full_read) {
      // We’re already in full read mode and still are outOfRange => truncated
      // file or corrupted offset
      entry_->ReadError(
          absl::DataLossError("Insufficient data after full read."));
      return;
    }

    if (!is_full_read_) {
      uint64_t current_data_end = file_offset_ + buffer.size();
      // Start from the next IFD offset if it's beyond what we already have:
      uint64_t read_begin = std::max(current_data_end, next_ifd_offset_);
      uint64_t read_end = read_begin + kInitialReadBytes;

      // If that end is some large threshold, we might want to do a full read:
      if (read_end > (16 * 1024 * 1024)) { 
        is_full_read_ = true;
        options_.byte_range = OptionalByteRangeRequest(file_offset_);
      } else {
        options_.byte_range =
            OptionalByteRangeRequest::Range(read_begin, read_end);
      }
    } else {
      // We set parse_result_->full_read but apparently we didn’t get enough
      // data. That’s an error or truncated file.
      entry_->ReadError(absl::DataLossError(
          "Need more data after already in full‑read mode."));
      return;
    }

    auto& cache = internal::GetOwningCache(*entry_);
    auto fut =
        cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Issued additional read request for key: " << entry_->key()
        << " with byte range: " << options_.byte_range;
    fut.Force();
    fut.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this),
         s = std::move(stamp)](ReadyFuture<kvstore::ReadResult> ready) mutable {
          ABSL_LOG_IF(INFO, tiff_logging)
              << "Additional read completed for key: " << self->entry_->key();
          self->OnAdditionalDataRead(std::move(ready), std::move(s));
        });
  }

  /// Called once more data arrives. We append that data to
  /// buffer and attempt parsing the IFD again.
  void OnAdditionalDataRead(ReadyFuture<kvstore::ReadResult> ready,
                            tensorstore::TimestampedStorageGeneration stamp) {
    const auto& r = ready.result();
    if (!r.ok()) {
      // Possibly partial read overshoot again
      if (!is_full_read_ && absl::IsOutOfRange(r.status())) {
        is_full_read_ = true;
        options_.byte_range = OptionalByteRangeRequest(file_offset_);
        auto& cache = internal::GetOwningCache(*entry_);
        auto future =
            cache.kvstore_driver_->Read(std::string(entry_->key()), options_);
        future.Force();
        future.ExecuteWhenReady(
            [self = internal::IntrusivePtr<ReadDirectoryOp>(this),
             st =
                 std::move(stamp)](ReadyFuture<kvstore::ReadResult> f) mutable {
              self->OnAdditionalDataRead(std::move(f), std::move(st));
            });
        return;
      }
      entry_->ReadError(
          internal::ConvertInvalidArgumentToFailedPrecondition(r.status()));
      return;
    }

    auto& rr = *r;
    if (rr.not_found()) {
      entry_->ReadError(
          absl::NotFoundError("Not found during incremental read."));
      return;
    }
    if (rr.aborted()) {
      if (existing_read_data_) {
        entry_->ReadSuccess(TiffDirectoryCache::ReadState{existing_read_data_,
                                                          std::move(rr.stamp)});
        return;
      }
      entry_->ReadError(absl::AbortedError("Read aborted, no existing data."));
      return;
    }

    // If we're reading from next_ifd_offset directly (which is far away from
    // our buffer end), we should reset our buffer instead of appending.
    if (options_.byte_range.inclusive_min >= file_offset_ + buffer.size()) {
      // This is a non-contiguous read, so replace buffer instead of appending
      buffer = std::move(rr.value);
      file_offset_ = options_.byte_range.inclusive_min;
    } else {
      // Append new data to buffer (contiguous read)
      size_t old_size = buffer.size();
      buffer.Append(rr.value);
      size_t new_size = buffer.size();

      // If we got less data than requested, treat it as a full read
      if (!is_full_read_ &&
          (new_size - old_size) < (options_.byte_range.size() - old_size)) {
        parse_result_->full_read = true;
      }
    }

    parse_result_->full_read = parse_result_->full_read || is_full_read_;

    // We can now try parsing the same IFD offset again
    StartParsingIFDs(std::move(stamp));
  }

  /// Called when we exhaust next_ifd_offset_ (i.e., reached offset=0 in the
  /// chain). We parse the final directory or load external arrays, etc.
  void OnAllIFDsDone(tensorstore::TimestampedStorageGeneration stamp) {
    ABSL_LOG_IF(INFO, tiff_logging)
        << "All IFDs parsed successfully for key: " << entry_->key()
        << ". Total directories: " << parse_result_->directories.size();
    // We now have parse_result_->directories for all IFDs.
    // Reserve space for a matching list of ImageDirectory objects.
    parse_result_->image_directories.clear();
    parse_result_->image_directories.resize(parse_result_->directories.size());

    bool has_external_arrays = false;

    // Parse each TiffDirectory into a corresponding ImageDirectory.
    // Also check entries for external arrays.
    for (size_t i = 0; i < parse_result_->directories.size(); ++i) {
      // Parse the IFD into parse_result_->image_directories[i].
      ABSL_LOG_IF(INFO, tiff_logging) << "Parsing image metadata from IFD #"
                                      << i << " for key: " << entry_->key();
      absl::Status s =
          ParseImageDirectory(parse_result_->directories[i].entries,
                              parse_result_->image_directories[i]);
      if (!s.ok()) {
        entry_->ReadError(s);
        return;
      }

      // Check for external arrays in this directory’s entries
      for (const auto& e : parse_result_->directories[i].entries) {
        if (e.is_external_array) {
          has_external_arrays = true;
        }
      }
    }

    if (!has_external_arrays) {
      ABSL_LOG_IF(INFO, tiff_logging)
          << "No external arrays found for key: " << entry_->key();
      // We’re done
      entry_->ReadSuccess(TiffDirectoryCache::ReadState{
          std::move(parse_result_), std::move(stamp)});
      return;
    }

    // Otherwise, load external arrays
    auto future = entry_->LoadExternalArrays(parse_result_, stamp);
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadDirectoryOp>(this),
         stamp](ReadyFuture<void> load_done) {
          if (!load_done.result().ok()) {
            self->entry_->ReadError(load_done.result().status());
            return;
          }
          // Done
          self->entry_->ReadSuccess(TiffDirectoryCache::ReadState{
              std::move(self->parse_result_), std::move(stamp)});
        });
  }
};

}  // namespace

Future<void> TiffDirectoryCache::Entry::LoadExternalArrays(
    std::shared_ptr<TiffParseResult> parse_result,
    tensorstore::TimestampedStorageGeneration stamp) {
  ABSL_LOG_IF(INFO, tiff_logging)
      << "Loading external arrays for key: " << this->key();
  // Collect all external arrays that need to be loaded
  struct ExternalArrayInfo {
    Tag tag;
    TiffDataType type;
    uint64_t offset;
    uint64_t count;
    size_t image_index;
  };

  std::vector<ExternalArrayInfo> external_arrays;

  // Collect external arrays from each directory (and store them by index).
  for (size_t i = 0; i < parse_result->directories.size(); ++i) {
    const auto& tiff_dir = parse_result->directories[i];

    for (const auto& entry : tiff_dir.entries) {
      if (!entry.is_external_array) continue;

      ExternalArrayInfo info;
      info.tag = entry.tag;
      info.type = entry.type;
      info.offset = entry.value_or_offset;
      info.count = entry.count;
      info.image_index = i;
      external_arrays.push_back(info);
    }
  }

  // If no external arrays, we can return immediately.
  if (external_arrays.empty()) {
    return MakeReadyFuture<void>();
  }

  auto [promise, future] = PromiseFuturePair<void>::Make();
  auto& cache = internal::GetOwningCache(*this);

  // Track how many arrays remain. We build a small shared struct to handle
  // completion.
  struct LoadState : public internal::AtomicReferenceCount<LoadState> {
    size_t remaining_count;
    absl::Status first_error;
    Promise<void> done_promise;

    LoadState(size_t count, Promise<void> pr)
        : remaining_count(count), done_promise(std::move(pr)) {}

    void CompleteOne(absl::Status s) {
      if (!s.ok() && first_error.ok()) {
        first_error = s;  // Record the first error
      }
      if (--remaining_count == 0) {
        // If we encountered any error, set that; otherwise OK.
        if (first_error.ok()) {
          done_promise.SetResult(absl::OkStatus());
        } else {
          done_promise.SetResult(first_error);
        }
      }
    }
  };

  auto load_state = internal::MakeIntrusivePtr<LoadState>(
      external_arrays.size(), std::move(promise));

  // Issue read operations for each external array in parallel.
  for (const auto& array_info : external_arrays) {
    ABSL_LOG_IF(INFO, tiff_logging)
        << "Reading external array for tag: "
        << static_cast<int>(array_info.tag) << ", Offset: " << array_info.offset
        << ", Count: " << array_info.count;
    // Compute the byte range.
    size_t element_size = GetTiffDataTypeSize(array_info.type);
    uint64_t byte_count = array_info.count * element_size;

    kvstore::ReadOptions read_opts;
    read_opts.generation_conditions.if_equal = stamp.generation;
    read_opts.byte_range = OptionalByteRangeRequest::Range(
        array_info.offset, array_info.offset + byte_count);

    ABSL_LOG_IF(INFO, tiff_logging)
        << "Reading external array for tag " << static_cast<int>(array_info.tag)
        << " at offset " << array_info.offset << " size " << byte_count;

    auto read_future =
        cache.kvstore_driver_->Read(std::string(this->key()), read_opts);
    read_future.Force();

    read_future.ExecuteWhenReady(
        [ls = load_state, parse_result, array_info,
         stamp](ReadyFuture<kvstore::ReadResult> ready) mutable {
          auto& rr = ready.result();
          if (!rr.ok()) {
            ls->CompleteOne(
                internal::ConvertInvalidArgumentToFailedPrecondition(
                    rr.status()));
            return;
          }

          if (rr->not_found() || rr->aborted()) {
            ls->CompleteOne(
                absl::DataLossError("Missing or aborted external array read."));
            return;
          }

          // We'll parse the data into the image directory's appropriate field.
          // Grab the corresponding ImageDirectory.
          auto& img_dir =
              parse_result->image_directories[array_info.image_index];

          // Create a reader for the data
          riegeli::CordReader cord_reader(&rr->value);

          // Determine how to parse the array based on the tag and type
          absl::Status parse_status;

          // Handle uint16_t arrays differently than uint64_t arrays
          if (array_info.type == TiffDataType::kShort &&
              (array_info.tag == Tag::kBitsPerSample ||
               array_info.tag == Tag::kSampleFormat)) {
            // Parse uint16_t arrays
            std::vector<uint16_t>* uint16_array = nullptr;

            switch (array_info.tag) {
              case Tag::kBitsPerSample:
                uint16_array = &img_dir.bits_per_sample;
                break;
              case Tag::kSampleFormat:
                uint16_array = &img_dir.sample_format;
                break;
              default:
                break;
            }

            if (uint16_array) {
              parse_status = ParseUint16Array(cord_reader, parse_result->endian,
                                              /*offset=*/0, array_info.count,
                                              *uint16_array);
            } else {
              parse_status = absl::OkStatus();  // Skip unhandled uint16_t array
            }
          } else {
            // Handle uint64_t arrays
            std::vector<uint64_t>* output_array = nullptr;
            switch (array_info.tag) {
              case Tag::kStripOffsets:
                output_array = &img_dir.chunk_offsets;
                break;
              case Tag::kStripByteCounts:
                output_array = &img_dir.chunk_bytecounts;
                break;
              case Tag::kTileOffsets:
                output_array = &img_dir.chunk_offsets;
                break;
              case Tag::kTileByteCounts:
                output_array = &img_dir.chunk_bytecounts;
                break;
              default:
                break;  // Skip unhandled uint64_t array
            }

            if (output_array) {
              parse_status =
                  ParseExternalArray(cord_reader, parse_result->endian,
                                     /*offset=*/0, array_info.count,
                                     array_info.type, *output_array);
            } else {
              parse_status = absl::OkStatus();  // Skip unhandled tag
            }
          }

          ls->CompleteOne(parse_status);
        });
  }

  return future;
}

size_t TiffDirectoryCache::Entry::ComputeReadDataSizeInBytes(
    const void* read_data) {
  return internal::EstimateHeapUsage(*static_cast<const ReadData*>(read_data));
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

  state->StartTiffRead();
}

TiffDirectoryCache::Entry* TiffDirectoryCache::DoAllocateEntry() {
  return new Entry;
}

size_t TiffDirectoryCache::DoGetSizeofEntry() { return sizeof(Entry); }

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore