// Copyright 2020 The TensorStore Authors
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

/// \file
/// Key-value store where each key corresponds to file path and the value is
/// stored as the file content.
///
/// The files containing the values of existing keys are never modified;
/// instead, they are replaced via `rename`.  Reading is, therefore, atomic
/// without the need for any locking, and the storage generation for a file is
/// obtained by encoding the device/volume identifier, inode number, and last
/// modification time into a string.  The inclusion of the modification time
/// provides protection against the recycling of inode numbers.
///
/// If the process is not permitted to write the filesystem, `Read` operations
/// will succeed but `Write` and `Delete` operations will return an error.
///
/// Atomic conditional read-modify-writes and deletes are implemented as
/// follows:
///
/// For each key, a lock file path is obtained by adding a suffix of ".__lock"
/// to the file path that would normally contain the value.  The locking rule
/// used is that only the owner of a POSIX byte range lock on this lock file,
/// while it has the canonical lock file path, may: (1) remove the lock file, or
/// (2) rename the lock file to become the new data file.
///
/// 1. To begin the atomic read-modify-write, open the lock file path (and
///    create it if it doesn't exist).
///
/// 2. Block until an exclusive lock (Linux 3.15+ open file description
///    lock/flock on BSD/Windows byte range lock) is acquired on the entire lock
///    file.
///
/// 3. Even once the lock has been acquired, it is possible that it no longer
///    has the canonical lock file path because the previous owner unlinked it
///    or renamed it, in which case the lock isn't actually valid per the
///    locking rule described above.  To check for this possibility, hold the
///    file open from step 1 and 2 open, and additionally re-open the lock file
///    (creating it if it doesn't exist), exactly as in step 1, and check that
///    both open files have the same device/volume number and inode number
///    values.  If they aren't the same file, then try again to acquire a valid
///    lock: close the original file, and return to step 2 with the newly opened
///    file.  If they are the same file, close the newly opened file and proceed
///    to step 4.  At this point a valid lock has been acquired.
///
/// 4. If the write is conditioned on the existing generation, check the
///    existing generation.  With the lock acquired, this is guaranteed to be
///    race free.  If the condition isn't satisfied, delete the lock file and
///    abort to indicate the condition was not satisfied.
///
/// 5. If the write operation is a delete operation:
///
///    a. Remove the actual data path.
///
///    b. Remove the lock file.
///
/// 6. Otherwise (normal write):
///
///    a. Truncate the lock file (it may contain garbage data from a previous
///       write attempt that terminated unexpectedly) and write the new
///       contents.
///
///    b. `fsync` the lock file.
///
///    c. Rename the lock file to the actual data path.
///
/// 7. Close the lock file to release the lock.
///
/// 8. `fsync` the parent directory of the file (to ensure the `unlink` or
///    `rename` operations are durable).  This step is skipped on MS Windows,
///    where `fsync` is not supported for directories.

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>  // IWYU pragma: keep for std::get<>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/functional/function_ref.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/match.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/context_binding.h"
#include "tensorstore/internal/file_io_concurrency_resource.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/os/error_code.h"
#include "tensorstore/internal/os/unique_handle.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/file/util.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// Include these last to reduce impact of macros.
#include "tensorstore/internal/os/file_lister.h"
#include "tensorstore/internal/os/file_util.h"

/// On FreeBSD and Mac OS X, `flock` can safely be used instead of open file
/// descriptor locks.  `flock`/`fcntl`/`lockf` all use the same underlying lock
/// mechanism and are all compatible with each other, and with NFS.
///
/// On Linux, `lockf` is simply equivalent to traditional `fcntl` UNIX record
/// locking (which is compatible with open file descriptor locks), while `flock`
/// is a completely independent mechanism, with some bad NFS interactions: on
/// Linux <=2.6.11, `flock` on an NFS-mounted filesystem provides only local
/// locking; on Linux >=2.6.12, `flock` on an NFS-mounted filesystem is treated
/// as an `fnctl` UNIX record lock that does affect all NFS clients.

/// This implementation does not currently support cancellation.  On Linux, most
/// filesystem operations, like `open`, `read`, `write`, and `fsync` cannot be
/// interrupted (even if they are blocked due to a non-responsive network
/// filesystem).  However, a more limited form of cancellation could be
/// supported, such as checking if the operation was cancelled while the task is
/// queued by the executor, or while waiting to acquire a lock.  This would
/// require passing through the Promise to the task code, rather than using
/// `MapFuture`.

namespace tensorstore {
namespace internal_file_kvstore {
namespace {
namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal::OsErrorCode;
using ::tensorstore::internal_file_util::IsKeyValid;
using ::tensorstore::internal_file_util::LongestDirectoryPrefix;
using ::tensorstore::internal_os::FileDescriptor;
using ::tensorstore::internal_os::FileInfo;
using ::tensorstore::internal_os::kLockSuffix;
using ::tensorstore::internal_os::UniqueFileDescriptor;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::ReadResult;
using ::tensorstore::kvstore::SupportedFeatures;

auto& file_bytes_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/bytes_read",
    "Bytes read by the file kvstore driver");

auto& file_bytes_written = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/bytes_written",
    "Bytes written by the file kvstore driver");

auto& file_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/read", "file driver kvstore::Read calls");

auto& file_open_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/open_read",
    "Number of times a file is opened for reading");

auto& file_batch_read = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/batch_read", "file driver reads after batching");

auto& file_write = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/write", "file driver kvstore::Write calls");

auto& file_delete_range = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/delete_range",
    "file driver kvstore::DeleteRange calls");

auto& file_list = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/list", "file driver kvstore::List calls");

auto& file_lock_contention = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/kvstore/file/lock_contention",
    "file driver write lock contention");

ABSL_CONST_INIT internal_log::VerboseFlag file_logging("file");

struct FileIoSyncResource
    : public internal::ContextResourceTraits<FileIoSyncResource> {
  constexpr static bool config_only = true;
  static constexpr char id[] = "file_io_sync";
  using Spec = bool;
  using Resource = Spec;
  static Spec Default() { return true; }
  static constexpr auto JsonBinder() { return jb::DefaultBinder<>; }
  static Result<Resource> Create(
      Spec v, internal::ContextResourceCreationContext context) {
    return v;
  }
  static Spec GetSpec(Resource v, const internal::ContextSpecBuilder& builder) {
    return v;
  }
};

struct FileKeyValueStoreSpecData {
  Context::Resource<internal::FileIoConcurrencyResource> file_io_concurrency;
  Context::Resource<FileIoSyncResource> file_io_sync;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.file_io_concurrency, x.file_io_sync);
  };

  // TODO(jbms): Storing a UNIX path as a JSON string presents a challenge
  // because UNIX paths are byte strings, and while it is common to use
  // UTF-8 encoding it is not required that the path be a valid UTF-8
  // string.  On MS Windows, there is a related problem that path names
  // are stored as UCS-2 may contain invalid surrogate pairs.
  //
  // However, while supporting such paths is important for general purpose
  // software like a file backup tool, it is relatively unlikely that the
  // user will want to use such a path as the root of a file-backed
  // KeyValueStore.
  //
  // If we do want to support such paths, there are various options
  // including base64-encoding, or using NUL as an escape sequence (taking
  // advantage of the fact that valid paths on all operating systems
  // cannot contain NUL characters).
  constexpr static auto default_json_binder = jb::Object(
      jb::Member(
          internal::FileIoConcurrencyResource::id,
          jb::Projection<&FileKeyValueStoreSpecData::file_io_concurrency>()),
      jb::Member(FileIoSyncResource::id,
                 jb::Projection<&FileKeyValueStoreSpecData::file_io_sync>())
      //
  );
};

class FileKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<FileKeyValueStoreSpec,
                                                    FileKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "file";

  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<std::string> ToUrl(std::string_view path) const override {
    return tensorstore::StrCat(id, "://", internal::PercentEncodeUriPath(path));
  }
};

class FileKeyValueStore
    : public internal_kvstore::RegisteredDriver<FileKeyValueStore,
                                                FileKeyValueStoreSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  Future<const void> DeleteRange(KeyRange range) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  const Executor& executor() { return spec_.file_io_concurrency->executor; }

  std::string DescribeKey(std::string_view key) override {
    return tensorstore::StrCat("local file ", tensorstore::QuoteString(key));
  }

  absl::Status GetBoundSpecData(FileKeyValueStoreSpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return SupportedFeatures::kSingleKeyAtomicReadModifyWrite |
           SupportedFeatures::kAtomicWriteWithoutOverwrite;
  }

  bool sync() const { return *spec_.file_io_sync; }

  SpecData spec_;
};

absl::Status ValidateKey(std::string_view key) {
  if (!IsKeyValid(key, kLockSuffix)) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Invalid key: ", tensorstore::QuoteString(key)));
  }
  return absl::OkStatus();
}

absl::Status ValidateKeyRange(const KeyRange& range) {
  auto prefix = LongestDirectoryPrefix(range);
  if (prefix.empty()) return absl::OkStatus();
  return ValidateKey(prefix);
}

// Encode in the generation fields that uniquely identify the file.
StorageGeneration GetFileGeneration(const FileInfo& info) {
  return StorageGeneration::FromValues(
      internal_os::GetDeviceId(info), internal_os::GetFileId(info),
      absl::ToUnixNanos(internal_os::GetMTime(info)));
}

/// RAII lock on an open file.
struct FileLock {
 public:
  absl::Status Acquire(FileDescriptor fd) {
    auto result = internal_os::FileLockTraits::Acquire(fd);
    if (result.ok()) {
      lock_.reset(fd);
    }
    return result;
  }

 private:
  internal_os::UniqueHandle<FileDescriptor, internal_os::FileLockTraits> lock_;
};

/// Creates any directory ancestors of `path` that do not exist, and returns an
/// open file descriptor to the parent directory of `path`.
Result<UniqueFileDescriptor> OpenParentDirectory(std::string path) {
  size_t end_pos = path.size();
  Result<UniqueFileDescriptor> fd;
  // Remove path components until we find a directory that exists.
  while (true) {
    // Loop backward until we reach a directory we can open (or .).
    // Loop forward, making directories, until we are done.
    size_t separator_pos = end_pos;
    while (separator_pos != 0 &&
           !internal_os::IsDirSeparator(path[separator_pos - 1])) {
      --separator_pos;
    }
    --separator_pos;
    const char* dir_path;
    if (separator_pos == std::string::npos) {
      dir_path = ".";
    } else if (separator_pos == 0) {
      dir_path = "/";
    } else {
      // Temporarily modify path to make `path.c_str()` a NULL-terminated string
      // containing the current ancestor directory path.
      path[separator_pos] = '\0';
      dir_path = path.c_str();
      end_pos = separator_pos;
    }
    fd = internal_os::OpenDirectoryDescriptor(dir_path);
    if (!fd.ok()) {
      if (absl::IsNotFound(fd.status())) {
        assert(separator_pos != 0 && separator_pos != std::string::npos);
        end_pos = separator_pos - 1;
        continue;
      }
      return fd.status();
    }
    // Revert the change to `path`.
    if (dir_path == path.c_str()) path[separator_pos] = '/';
    break;
  }

  // Add path components and attempt to `mkdir` until we have reached the full
  // path.
  while (true) {
    size_t separator_pos = path.find('\0', end_pos);
    if (separator_pos == std::string::npos) {
      // No more ancestors remain.
      return fd;
    }
    TENSORSTORE_RETURN_IF_ERROR(internal_os::MakeDirectory(path));
    fd = internal_os::OpenDirectoryDescriptor(path);
    TENSORSTORE_RETURN_IF_ERROR(fd.status());
    path[separator_pos] = '/';
    end_pos = separator_pos + 1;
  }
}

absl::Status VerifyRegularFile(FileDescriptor fd, FileInfo* info,
                               const char* path) {
  TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd, info));
  if (!internal_os::IsRegularFile(*info)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("Not a regular file: ", path));
  }
  return absl::OkStatus();
}

Result<UniqueFileDescriptor> OpenValueFile(const char* path,
                                           StorageGeneration* generation,
                                           int64_t* size = nullptr) {
  auto fd = internal_os::OpenExistingFileForReading(path);
  if (!fd.ok()) {
    // Map not found to an empty file.
    if (absl::IsNotFound(fd.status())) {
      *generation = StorageGeneration::NoValue();
      return UniqueFileDescriptor{};
    }
    return fd;
  }
  FileInfo info;
  TENSORSTORE_RETURN_IF_ERROR(VerifyRegularFile(fd->get(), &info, path));
  if (size) *size = internal_os::GetSize(info);
  *generation = GetFileGeneration(info);
  return fd;
}

/// Helper class to acquire write lock for the specified path.
struct WriteLockHelper {
  std::string lock_path;  // Composed write lock file path.
  UniqueFileDescriptor lock_fd;
  FileInfo info;
  FileLock lock;

  /// Constructor.
  ///
  /// \param path Full path to the key.
  WriteLockHelper(const std::string& path)
      : lock_path(tensorstore::StrCat(path, kLockSuffix)) {}

  /// Opens or Creates the lock file.
  Result<UniqueFileDescriptor> OpenLockFile(FileInfo* info) {
    auto fd = internal_os::OpenFileForWriting(lock_path);
    if (fd.ok()) {
      TENSORSTORE_RETURN_IF_ERROR(
          VerifyRegularFile(fd->get(), info, lock_path.c_str()));
    }
    return fd;
  }

  /// Creates the lock file and acquires the lock.
  absl::Status CreateAndAcquire() {
    TENSORSTORE_ASSIGN_OR_RETURN(lock_fd, OpenLockFile(&info));
    // Loop until lock is acquired successfully.
    while (true) {
      // Acquire lock.
      if (auto status = lock.Acquire(lock_fd.get()); !status.ok()) {
        return MaybeAnnotateStatus(
            std::move(status),
            tensorstore::StrCat("Failed to acquire lock on file: ",
                                QuoteString(lock_path)));
      }
      // Check if the lock file has been renamed.
      FileInfo other_info;
      TENSORSTORE_ASSIGN_OR_RETURN(UniqueFileDescriptor other_fd,
                                   OpenLockFile(&other_info));
      if (internal_os::GetDeviceId(other_info) ==
              internal_os::GetDeviceId(info) &&
          internal_os::GetFileId(other_info) == internal_os::GetFileId(info)) {
        // Lock was acquired successfully.
        return absl::OkStatus();
      }
      info = other_info;
      // Release lock and try again.
      lock = FileLock{};
      lock_fd = std::move(other_fd);
      file_lock_contention.Increment();
    }
  }

  // Deletes the lock file.
  absl::Status Delete() {
    auto status = internal_os::DeleteOpenFile(lock_fd.get(), lock_path);
    if (status.ok() || absl::IsNotFound(status)) {
      return absl::OkStatus();
    }
    return MaybeAnnotateStatus(std::move(status), "Failed to clean lock file");
  }
};

Result<absl::Cord> ReadFromFileDescriptor(FileDescriptor fd,
                                          ByteRange byte_range) {
  file_batch_read.Increment();
  internal::FlatCordBuilder buffer(byte_range.size(), false);
  size_t offset = 0;
  while (offset < buffer.size()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto n, internal_os::ReadFromFile(fd, buffer.data() + offset,
                                          buffer.size() - offset,
                                          byte_range.inclusive_min + offset));
    if (n > 0) {
      file_bytes_read.IncrementBy(n);
      offset += n;
      buffer.set_inuse(offset);
      continue;
    }
    if (n == 0) {
      return absl::UnavailableError(
          tensorstore::StrCat("Length changed while reading"));
    }
  }
  return std::move(buffer).Build();
}

class BatchReadTask;
using BatchReadTaskBase = internal_kvstore_batch::BatchReadEntry<
    FileKeyValueStore,
    internal_kvstore_batch::ReadRequest<kvstore::ReadGenerationConditions>,
    // BatchEntryKey members:
    std::string /* file_path*/>;

class BatchReadTask final
    : public BatchReadTaskBase,
      public internal::AtomicReferenceCount<BatchReadTask> {
 private:
  // Working state.
  TimestampedStorageGeneration stamp_;
  UniqueFileDescriptor fd_;
  int64_t size_;

 public:
  BatchReadTask(BatchEntryKey&& batch_entry_key_)
      : BatchReadTaskBase(std::move(batch_entry_key_)),
        // Create initial reference count that will be transferred to `Submit`.
        internal::AtomicReferenceCount<BatchReadTask>(/*initial_ref_count=*/1) {
  }

  void Submit(Batch::View batch) final {
    if (request_batch.requests.empty()) return;
    driver().executor()(
        [self = internal::IntrusivePtr<BatchReadTask>(
             // Acquire initial reference count.
             this, internal::adopt_object_ref)] { self->ProcessBatch(); });
  }

  Result<kvstore::ReadResult> DoByteRangeRead(ByteRange byte_range) {
    absl::Cord value;
    TENSORSTORE_ASSIGN_OR_RETURN(
        value, ReadFromFileDescriptor(fd_.get(), byte_range),
        tensorstore::MaybeAnnotateStatus(_, "Error reading from open file"));
    return kvstore::ReadResult::Value(std::move(value), stamp_);
  }

  void ProcessBatch() {
    stamp_.time = absl::Now();
    file_open_read.Increment();
    auto& requests = request_batch.requests;
    TENSORSTORE_ASSIGN_OR_RETURN(
        fd_,
        OpenValueFile(std::get<std::string>(batch_entry_key).c_str(),
                      &stamp_.generation, &size_),
        internal_kvstore_batch::SetCommonResult(requests, std::move(_)));
    if (!fd_.valid()) {
      internal_kvstore_batch::SetCommonResult(
          requests, kvstore::ReadResult::Missing(stamp_.time));
      return;
    }

    internal_kvstore_batch::ValidateGenerationsAndByteRanges(requests, stamp_,
                                                             size_);

    if (requests.empty()) return;
    if (requests.size() == 1) {
      auto& byte_range_request =
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(requests[0]);
      // Perform single read immediately.
      byte_range_request.promise.SetResult(
          DoByteRangeRead(byte_range_request.byte_range.AsByteRange()));
      return;
    }

    const auto& executor = driver().executor();

    internal_kvstore_batch::CoalescingOptions coalescing_options;
    coalescing_options.max_extra_read_bytes = 255;
    internal_kvstore_batch::ForEachCoalescedRequest<Request>(
        requests, coalescing_options,
        [&](ByteRange coalesced_byte_range, span<Request> coalesced_requests) {
          auto self = internal::IntrusivePtr<BatchReadTask>(this);
          executor([self = std::move(self), coalesced_byte_range,
                    coalesced_requests] {
            self->ProcessCoalescedRead(coalesced_byte_range,
                                       coalesced_requests);
          });
        });
  }

  void ProcessCoalescedRead(ByteRange coalesced_byte_range,
                            span<Request> coalesced_requests) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 DoByteRangeRead(coalesced_byte_range),
                                 internal_kvstore_batch::SetCommonResult(
                                     coalesced_requests, std::move(_)));
    internal_kvstore_batch::ResolveCoalescedRequests(
        coalesced_byte_range, coalesced_requests, std::move(read_result));
  }
};

Future<ReadResult> FileKeyValueStore::Read(Key key, ReadOptions options) {
  file_read.Increment();
  TENSORSTORE_RETURN_IF_ERROR(ValidateKey(key));
  auto [promise, future] = PromiseFuturePair<kvstore::ReadResult>::Make();
  BatchReadTask::MakeRequest<BatchReadTask>(
      *this, {std::move(key)}, options.batch, options.staleness_bound,
      BatchReadTask::Request{{std::move(promise), options.byte_range},
                             std::move(options.generation_conditions)});
  return std::move(future);
}

/// Implements `FileKeyValueStore::Write`.
struct WriteTask {
  std::string full_path;
  absl::Cord value;
  kvstore::WriteOptions options;
  bool sync;

  Result<TimestampedStorageGeneration> operator()() const {
    TimestampedStorageGeneration r;
    r.time = absl::Now();

    WriteLockHelper lock_helper(full_path);
    TENSORSTORE_ASSIGN_OR_RETURN(auto dir_fd, OpenParentDirectory(full_path));
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.CreateAndAcquire());
    bool delete_lock_file = true;

    auto generation_result = [&]() -> Result<StorageGeneration> {
      FileDescriptor fd = lock_helper.lock_fd.get();
      const std::string& lock_path = lock_helper.lock_path;
      // Check condition.
      if (!StorageGeneration::IsUnknown(
              options.generation_conditions.if_equal)) {
        StorageGeneration generation;
        TENSORSTORE_ASSIGN_OR_RETURN(
            UniqueFileDescriptor value_fd,
            OpenValueFile(full_path.c_str(), &generation));
        if (generation != options.generation_conditions.if_equal) {
          return StorageGeneration::Unknown();
        }
      }
      if (internal_os::GetSize(lock_helper.info) > value.size()) {
        // Only truncate when the file is larger. In the common path, the lock
        // file is newly created, so truncate is useless.
        TENSORSTORE_RETURN_IF_ERROR(internal_os::TruncateFile(fd));
      }
      absl::Cord value_for_write = value;
      for (; !value_for_write.empty();) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto n, internal_os::WriteCordToFile(fd, value_for_write),
            MaybeAnnotateStatus(_,
                                tensorstore::StrCat("Failed writing: ",
                                                    QuoteString(lock_path))));
        file_bytes_written.IncrementBy(n);
        if (n == value_for_write.size()) break;
        value_for_write.RemovePrefix(n);
      }

      if (this->sync) {
        TENSORSTORE_RETURN_IF_ERROR(internal_os::FsyncFile(fd));
      }
      TENSORSTORE_RETURN_IF_ERROR(
          internal_os::RenameOpenFile(fd, lock_path, full_path));
      delete_lock_file = false;
      if (this->sync) {
        // fsync the parent directory to ensure the `rename` is durable.
        TENSORSTORE_RETURN_IF_ERROR(
            internal_os::FsyncDirectory(dir_fd.get()),
            MaybeAnnotateStatus(
                _, tensorstore::StrCat(
                       "Error calling fsync on parent directory of: ",
                       full_path)));
      }
      lock_helper.lock = FileLock{};

      // Retrieve `FileInfo` after the fsync and rename to ensure the
      // modification time doesn't change afterwards.
      FileInfo info;
      TENSORSTORE_RETURN_IF_ERROR(internal_os::GetFileInfo(fd, &info));
      return GetFileGeneration(info);
    }();

    if (delete_lock_file) {
      TENSORSTORE_RETURN_IF_ERROR(lock_helper.Delete());
    }
    if (!generation_result) {
      return std::move(generation_result).status();
    }
    r.generation = std::move(*generation_result);
    return r;
  }
};

/// Implements `FileKeyValueStore::Delete`.
struct DeleteTask {
  std::string full_path;
  kvstore::WriteOptions options;
  bool sync;

  Result<TimestampedStorageGeneration> operator()() const {
    TimestampedStorageGeneration r;
    r.time = absl::Now();

    WriteLockHelper lock_helper(full_path);
    TENSORSTORE_ASSIGN_OR_RETURN(auto dir_fd, OpenParentDirectory(full_path));
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.CreateAndAcquire());

    bool fsync_directory = false;
    auto generation_result = [&]() -> Result<StorageGeneration> {
      // Check condition.
      if (!StorageGeneration::IsUnknown(
              options.generation_conditions.if_equal)) {
        StorageGeneration generation;
        TENSORSTORE_ASSIGN_OR_RETURN(
            UniqueFileDescriptor value_fd,
            OpenValueFile(full_path.c_str(), &generation));
        if (generation != options.generation_conditions.if_equal) {
          return StorageGeneration::Unknown();
        }
      }
      auto status = internal_os::DeleteFile(full_path);
      if (!status.ok() && !absl::IsNotFound(status)) {
        return status;
      }
      fsync_directory = this->sync;
      return StorageGeneration::NoValue();
    }();

    // Delete the lock file.
    TENSORSTORE_RETURN_IF_ERROR(lock_helper.Delete());

    // fsync the parent directory to ensure the `rename` is durable.
    if (fsync_directory) {
      TENSORSTORE_RETURN_IF_ERROR(
          internal_os::FsyncDirectory(dir_fd.get()),
          MaybeAnnotateStatus(
              _,
              tensorstore::StrCat(
                  "Error calling fsync on parent directory of: ", full_path)));
    }
    if (!generation_result) {
      return std::move(generation_result).status();
    }
    r.generation = std::move(*generation_result);
    return r;
  }
};

Future<TimestampedStorageGeneration> FileKeyValueStore::Write(
    Key key, std::optional<Value> value, WriteOptions options) {
  file_write.Increment();
  TENSORSTORE_RETURN_IF_ERROR(ValidateKey(key));
  if (value) {
    return MapFuture(executor(), WriteTask{std::move(key), std::move(*value),
                                           std::move(options), this->sync()});
  } else {
    return MapFuture(executor(), DeleteTask{std::move(key), std::move(options),
                                            this->sync()});
  }
}

/// Implements `FileKeyValueStore::DeleteRange`.
struct DeleteRangeTask {
  KeyRange range;

  // TODO(jbms): Add fsync support

  void operator()(Promise<void> promise) {
    std::string prefix(internal_file_util::LongestDirectoryPrefix(range));
    absl::Status delete_status;
    auto status = internal_os::RecursiveFileList(
        prefix,
        [&](std::string_view path) {
          return tensorstore::IntersectsPrefix(range, path);
        },
        [&](auto entry) -> absl::Status {
          if (!promise.result_needed()) return absl::CancelledError("");
          bool do_delete = false;
          if (entry.IsDirectory()) {
            // Delete fully contained directories.
            do_delete = tensorstore::ContainsPrefix(range, entry.GetFullPath());
          } else {
            do_delete = tensorstore::Contains(range, entry.GetFullPath());
          }
          if (do_delete) {
            auto s = entry.Delete();
            if (!s.ok() && !absl::IsNotFound(s) &&  // Already deleted
                !absl::IsFailedPrecondition(s)) {   // No delete permissions
              ABSL_LOG_IF(INFO, file_logging) << s;
              delete_status.Update(s);
            }
          }
          // Even when failing to delete the current file, continue to the next
          // file.
          return absl::OkStatus();
        });
    if (!status.ok()) {
      promise.SetResult(MakeResult(std::move(status)));
    }
    promise.SetResult(MakeResult(std::move(delete_status)));
  }
};

Future<const void> FileKeyValueStore::DeleteRange(KeyRange range) {
  file_delete_range.Increment();
  if (range.empty()) return absl::OkStatus();  // Converted to a ReadyFuture.
  TENSORSTORE_RETURN_IF_ERROR(ValidateKeyRange(range));
  return PromiseFuturePair<void>::Link(
             WithExecutor(executor(), DeleteRangeTask{std::move(range)}))
      .future;
}

/// Implements `FileKeyValueStore:::List`.
struct ListTask {
  kvstore::ListOptions options;
  ListReceiver receiver;

  void operator()() {
    std::atomic<bool> cancelled = false;
    execution::set_starting(receiver, [&cancelled] {
      cancelled.store(true, std::memory_order_relaxed);
    });
    std::string prefix(
        internal_file_util::LongestDirectoryPrefix(options.range));
    auto status = internal_os::RecursiveFileList(
        prefix,
        [&](std::string_view path) {
          return tensorstore::IntersectsPrefix(options.range, path);
        },
        [&](auto entry) -> absl::Status {
          if (cancelled.load(std::memory_order_relaxed)) {
            return absl::CancelledError("");
          }
          if (entry.IsDirectory()) return absl::OkStatus();
          std::string_view path = entry.GetFullPath();
          if (tensorstore::Contains(options.range, path) &&
              !absl::EndsWith(path, kLockSuffix)) {
            // TODO: If the file was stat'd, include length.
            path.remove_prefix(options.strip_prefix_length);
            execution::set_value(receiver,
                                 ListEntry{std::string(path), entry.GetSize()});
          }
          return absl::OkStatus();
        });
    if (!status.ok() && !cancelled.load(std::memory_order_relaxed)) {
      execution::set_error(receiver, std::move(status));
      execution::set_stopping(receiver);
      return;
    }
    execution::set_done(receiver);
    execution::set_stopping(receiver);
  }
};

void FileKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  file_list.Increment();
  if (options.range.empty()) {
    execution::set_starting(receiver, [] {});
    execution::set_done(receiver);
    execution::set_stopping(receiver);
    return;
  }
  if (auto error = ValidateKeyRange(options.range); !error.ok()) {
    execution::set_starting(receiver, [] {});
    execution::set_error(receiver, std::move(error));
    execution::set_stopping(receiver);
    return;
  }
  executor()(ListTask{std::move(options), std::move(receiver)});
}

Future<kvstore::DriverPtr> FileKeyValueStoreSpec::DoOpen() const {
  auto driver_ptr = internal::MakeIntrusivePtr<FileKeyValueStore>();
  driver_ptr->spec_ = data_;
  return driver_ptr;
}

Result<kvstore::Spec> ParseFileUrl(std::string_view url) {
  auto driver_spec = internal::MakeIntrusivePtr<FileKeyValueStoreSpec>();
  driver_spec->data_.file_io_concurrency =
      Context::Resource<internal::FileIoConcurrencyResource>::DefaultSpec();
  driver_spec->data_.file_io_sync =
      Context::Resource<FileIoSyncResource>::DefaultSpec();
  auto parsed = internal::ParseGenericUri(url);
  assert(parsed.scheme == internal_file_kvstore::FileKeyValueStoreSpec::id);
  if (!parsed.query.empty()) {
    return absl::InvalidArgumentError("Query string not supported");
  }
  if (!parsed.fragment.empty()) {
    return absl::InvalidArgumentError("Fragment identifier not supported");
  }
  return {std::in_place, std::move(driver_spec),
          internal::PercentDecode(parsed.authority_and_path)};
}

}  // namespace
}  // namespace internal_file_kvstore
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_file_kvstore::FileKeyValueStore)

namespace {

const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_file_kvstore::FileKeyValueStoreSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{
        tensorstore::internal_file_kvstore::FileKeyValueStoreSpec::id,
        tensorstore::internal_file_kvstore::ParseFileUrl};

const tensorstore::internal::ContextResourceRegistration<
    tensorstore::internal_file_kvstore::FileIoSyncResource>
    file_io_sync_registration;

}  // namespace
