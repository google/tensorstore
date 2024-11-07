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

#include "tensorstore/internal/os/file_lister.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/internal/os/file_util.h"
#include "tensorstore/internal/testing/scoped_directory.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::internal_os::FsyncDirectory;
using ::tensorstore::internal_os::FsyncFile;
using ::tensorstore::internal_os::MakeDirectory;
using ::tensorstore::internal_os::OpenDirectoryDescriptor;
using ::tensorstore::internal_os::OpenExistingFileForReading;
using ::tensorstore::internal_os::OpenFileWrapper;
using ::tensorstore::internal_os::OpenFlags;
using ::tensorstore::internal_os::ReadFromFile;
using ::tensorstore::internal_os::RecursiveFileList;
using ::tensorstore::internal_os::WriteToFile;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

static std::optional<ScopedTemporaryDirectory> g_scoped_dir;

void AddFiles(std::string_view root) {
  ABSL_CHECK(!root.empty());

  // Setup files.
  TENSORSTORE_CHECK_OK(MakeDirectory(absl::StrCat(root, "/xyz")));
  TENSORSTORE_CHECK_OK(MakeDirectory(absl::StrCat(root, "/zzq")));
  std::string fname = "/a.txt";
  for (; fname[1] < 'd'; fname[1] += 1) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto f,
        OpenFileWrapper(absl::StrCat(root, fname), OpenFlags::DefaultWrite));
    TENSORSTORE_CHECK_OK(FsyncFile(f.get()));

    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto g, OpenFileWrapper(absl::StrCat(root, "/xyz", fname),
                                OpenFlags::DefaultWrite));
    TENSORSTORE_CHECK_OK(FsyncFile(g.get()));
  }

  for (const auto& suffix : {"/xyz", ""}) {
    TENSORSTORE_CHECK_OK_AND_ASSIGN(
        auto f, OpenDirectoryDescriptor(absl::StrCat(root, suffix)));
    EXPECT_THAT(FsyncDirectory(f.get()), IsOk());
  }
}

class RecursiveFileListTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    g_scoped_dir.emplace();
    AddFiles(g_scoped_dir->path());
  }

  static void TearDownTestSuite() { g_scoped_dir = std::nullopt; }

  RecursiveFileListTest() : cwd_(g_scoped_dir->path()) {}

 private:
  tensorstore::internal_testing::ScopedCurrentWorkingDirectory cwd_;
};

TEST_F(RecursiveFileListTest, MissingIsOk) {
  EXPECT_THAT(RecursiveFileList(
                  g_scoped_dir->path() + "/aax",
                  /*recurse_into=*/[](std::string_view path) { return true; },
                  /*on_item=*/[](auto entry) { return absl::OkStatus(); }),
              IsOk());
}

TEST_F(RecursiveFileListTest, EmptyIsOk) {
  // zzq only has . and .., so it will start as empty.
  EXPECT_THAT(RecursiveFileList(
                  g_scoped_dir->path() + "/zzq",
                  /*recurse_into=*/[](std::string_view path) { return true; },
                  /*on_item=*/[](auto entry) { return absl::OkStatus(); }),
              IsOk());
}

TEST_F(RecursiveFileListTest, FileIsFailure) {
  // zzq only has . and .., so it will start as empty.
  EXPECT_THAT(RecursiveFileList(
                  g_scoped_dir->path() + "/a.txt",
                  /*recurse_into=*/[](std::string_view path) { return true; },
                  /*on_item=*/[](auto entry) { return absl::OkStatus(); }),
              ::testing::Not(IsOk()));
}

TEST_F(RecursiveFileListTest, FullDirectory) {
  // List the directory (fullpath / relative path)
  for (const std::string& root :
       {g_scoped_dir->path(), std::string("."), std::string()}) {
    std::vector<std::string> files;
    EXPECT_THAT(
        RecursiveFileList(
            root, /*recurse_into=*/[](std::string_view path) { return true; },
            /*on_item=*/
            [&](auto entry) {
              files.push_back(absl::StrCat(entry.IsDirectory() ? "<dir>" : "",
                                           entry.GetPathComponent()));
              return absl::OkStatus();
            }),
        IsOk());
    EXPECT_THAT(files, ::testing::UnorderedElementsAre(
                           "c.txt", "b.txt", "a.txt", "<dir>zzq", "c.txt",
                           "b.txt", "a.txt", "<dir>xyz", "<dir>"));
  }
}

TEST_F(RecursiveFileListTest, SubDirectory) {
  // List the subdirectory (relative path)
  std::vector<std::string> files;
  EXPECT_THAT(
      RecursiveFileList(
          "xyz", /*recurse_into=*/[](std::string_view path) { return true; },
          /*on_item=*/
          [&](auto entry) {
            files.push_back(absl::StrCat(entry.IsDirectory() ? "<dir>" : "",
                                         entry.GetFullPath()));
            return absl::OkStatus();
          }),
      IsOk());
  EXPECT_THAT(files, ::testing::UnorderedElementsAre("xyz/a.txt", "xyz/b.txt",
                                                     "xyz/c.txt", "<dir>xyz"));
}

TEST_F(RecursiveFileListTest, NonRecursive) {
  // List the subdirectory (relative path)
  std::vector<std::string> files;
  EXPECT_THAT(
      RecursiveFileList(
          "",
          /*recurse_into=*/
          [](std::string_view path) {
            ABSL_LOG(INFO) << path;
            return path.empty();
          },
          /*on_item=*/
          [&](auto entry) {
            files.push_back(absl::StrCat(entry.IsDirectory() ? "<dir>" : "",
                                         entry.GetFullPath()));
            return absl::OkStatus();
          }),
      IsOk());
  EXPECT_THAT(files,
              ::testing::UnorderedElementsAre("c.txt", "b.txt", "a.txt",
                                              "<dir>zzq", "<dir>xyz", "<dir>"));
}

TEST(RecursiveFileListEntryTest, DeleteWithOpenFile) {
  // List the subdirectory (relative path)
  ScopedTemporaryDirectory tmpdir;
  AddFiles(tmpdir.path());

  {
    auto f = OpenFileWrapper(absl::StrCat(tmpdir.path(), "/read.txt"),
                             OpenFlags::DefaultWrite);
    EXPECT_THAT(f, IsOk());
    EXPECT_THAT(WriteToFile(f->get(), "bar", 3), IsOkAndHolds(3));
  }

  {
    // Open file; it should be deleted.
    auto f =
        OpenExistingFileForReading(absl::StrCat(tmpdir.path(), "/read.txt"));
    EXPECT_THAT(f, IsOk());

    std::vector<std::string> files;
    EXPECT_THAT(RecursiveFileList(
                    tmpdir.path(),
                    /*recurse_into=*/
                    [](std::string_view path) { return true; },
                    /*on_item=*/
                    [&](auto entry) {
                      if (entry.GetFullPath() == tmpdir.path()) {
                        return absl::OkStatus();
                      }
                      auto status = entry.Delete();
                      if (status.ok() || absl::IsNotFound(status))
                        return absl::OkStatus();
                      return status;
                    }),
                IsOk());

    char buf[16];
    EXPECT_THAT(ReadFromFile(f->get(), buf, 3, 0), IsOkAndHolds(3));
  }

  std::vector<std::string> files;
  EXPECT_THAT(
      RecursiveFileList(
          tmpdir.path(),
          /*recurse_into=*/[](std::string_view path) { return true; },
          /*on_item=*/
          [&](auto entry) {
            files.push_back(absl::StrCat(entry.IsDirectory() ? "<dir>" : "",
                                         entry.GetPathComponent()));
            return absl::OkStatus();
          }),
      IsOk());

  EXPECT_THAT(files, ::testing::UnorderedElementsAre("<dir>"));
}

}  // namespace
