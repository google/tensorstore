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

#ifndef TENSORSTORE_INTERNAL_HTTP_TEST_HTTPSERVER_H_
#define TENSORSTORE_INTERNAL_HTTP_TEST_HTTPSERVER_H_

#include <optional>
#include <string>

#include "tensorstore/internal/os/subprocess.h"
#include "tensorstore/internal/testing/scoped_directory.h"

namespace tensorstore {
namespace internal_http {

/// TestHttpServer runs the test_httpserver binary and returns a port.
class TestHttpServer {
 public:
  TestHttpServer();
  ~TestHttpServer();

  // Spawns the subprocess and sets the http address.
  void SpawnProcess();

  // Returns the http address of the server.
  std::string http_address() { return http_address_; }

  // Returns the path to the certificate file.
  std::string GetCertPath();

  // If available, dumps the stdout pipe of the child process to INFO.
  void MaybeLogStdoutPipe();

 private:
  void InitializeCertificates();

  std::string http_address_;
  std::string root_path_;
  std::optional<internal_testing::ScopedTemporaryDirectory> cert_dir_;
  std::optional<internal::Subprocess> child_;
};

}  // namespace internal_http
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_HTTP_TEST_HTTPSERVER_H_
