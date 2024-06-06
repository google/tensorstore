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
using ::tensorstore::internal_os::FsyncDirectory;
using ::tensorstore::internal_os::FsyncFile;
using ::tensorstore::internal_os::MakeDirectory;
using ::tensorstore::internal_os::OpenDirectoryDescriptor;
using ::tensorstore::internal_os::OpenFileForWriting;
using ::tensorstore::internal_os::RecursiveFileList;
using ::tensorstore::internal_testing::ScopedTemporaryDirectory;

static std::optional<ScopedTemporaryDirectory> g_scoped_dir;

class RecursiveFileListTest : public testing::Test {
 public:
  static void SetUpTestSuite() {
    g_scoped_dir.emplace();

    // Setup files.
    TENSORSTORE_CHECK_OK(MakeDirectory(g_scoped_dir->path() + "/xyz"));
    TENSORSTORE_CHECK_OK(MakeDirectory(g_scoped_dir->path() + "/zzq"));
    std::string fname = "a.txt";
    for (; fname[0] < 'd'; fname[0] += 1) {
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto f,
          OpenFileForWriting(absl::StrCat(g_scoped_dir->path(), "/", fname)));
      TENSORSTORE_CHECK_OK(FsyncFile(f.get()));

      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto g, OpenFileForWriting(
                      absl::StrCat(g_scoped_dir->path(), "/xyz/", fname)));
      TENSORSTORE_CHECK_OK(FsyncFile(g.get()));
    }

    for (const auto& suffix : {"/xyz", ""}) {
      TENSORSTORE_CHECK_OK_AND_ASSIGN(
          auto f,
          OpenDirectoryDescriptor(absl::StrCat(g_scoped_dir->path(), suffix)));
      EXPECT_THAT(FsyncDirectory(f.get()), IsOk());
    }
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

}  // namespace
