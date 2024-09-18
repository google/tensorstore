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

#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include <curl/curl.h>  // IWYU pragma: keep
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/http/curl_factory.h"
#include "tensorstore/internal/http/curl_wrappers.h"
#include "tensorstore/internal/os/file_util.h"

#ifndef _WIN32
// boringssl / openssl is not used in windows; in mingw it currently
// has a compilation error.
#include <openssl/x509.h>  // IWYU pragma: keep
#ifndef OPENSSL_IS_BORINGSSL
#define TENSORSTORE_LOOKUP_X509_PATHS
#endif
#endif

ABSL_FLAG(std::optional<bool>, tensorstore_use_fallback_ssl_certs, std::nullopt,
          "Search for certificate files/directories in fallback CA paths.");

#ifndef TENSORSTORE_USE_FALLBACK_SSL_CERTS
#if defined(OPENSSL_IS_BORINGSSL) && !defined(TENSORSTORE_SYSTEM_CURL)
// By default, look at fallback paths when both of the following are true:
// * using any boringssl; boringssl does not vendor the system certs.
// * using the bundled curl; the bundled curl does not detect system certs.
// Otherwise curl should either detect the system certs, or use the certs
// provided by the underlying ssl libraries.
#define TENSORSTORE_USE_FALLBACK_SSL_CERTS 1
#else
#define TENSORSTORE_USE_FALLBACK_SSL_CERTS 0
#endif
#endif

namespace tensorstore {
namespace internal_http {
namespace {

using ::tensorstore::internal_os::GetFileInfo;

// Attempt to find common SSL certificate files and directories.
// See also gRPC and golang:
// https://github.com/grpc/grpc/blob/23adb994cfbb91a66d0d5a52a4cc07a2a42c2d53/src/core/lib/security/security_connector/load_system_roots_supported.cc#L51
// https://go.dev/src/crypto/x509/root_linux.go
//
#if defined(__APPLE__)
const char* kCertFiles[] = {"/etc/ssl/cert.pem"};
const char* kCertDirectories[] = {};
#else  // __linux__
const char* kCertFiles[] = {
    "/etc/ssl/certs/ca-certificates.crt",  // Debian/Ubuntu/Gentoo etc.
    "/etc/pki/tls/certs/ca-bundle.crt",    // Fedora/RHEL 6
    "/etc/ssl/ca-bundle.pem",              // OpenSUSE
    "/etc/pki/tls/cacert.pem",             // OpenELEC
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",  // CentOS/RHEL 7
    "/etc/ssl/cert.pem",                                  // Alpine Linux
#if defined(__FreeBSD__)
    "/usr/local/etc/ssl/cert.pem",
    "/usr/local/share/certs/ca-root-nss.crt",
#endif
};
const char* kCertDirectories[] = {
    "/etc/ssl/certs",      // SLES10/SLES11
    "/etc/pki/tls/certs",  // Fedora/RHEL
#if defined(__FreeBSD__)
    "/usr/local/share/certs",
#endif
    "/etc/openssl/certs",
};
#endif

struct CertFile {
  std::string cert_file;
};
struct CertDirectory {
  std::string cert_directory;
};

}  // namespace

void CurlPtrHook(CurlPtr& handle) {
  static auto certs =
      []() -> std::variant<std::monostate, CertFile, CertDirectory> {
    // Return either "SSL_CERT_FILE" or "SSL_CERT_DIR" if set.
    if (auto default_cert_env = internal::GetEnv("SSL_CERT_FILE");
        default_cert_env.has_value()) {
      return CertFile{*std::move(default_cert_env)};
    }
    if (auto default_cert_env = internal::GetEnv("SSL_CERT_DIR");
        default_cert_env.has_value()) {
      return CertDirectory{*std::move(default_cert_env)};
    }

    // This check only happens on startup so that all the curl handles use
    // the same certificates.
    if (!internal::GetFlagOrEnvValue(FLAGS_tensorstore_use_fallback_ssl_certs,
                                     "TENSORSTORE_USE_FALLBACK_SSL_CERTS")
             .value_or(TENSORSTORE_USE_FALLBACK_SSL_CERTS)) {
      return std::monostate();
    }

    internal_os::FileInfo info;
    auto try_file = [&info](const std::string& filepath) {
      return !filepath.empty() && GetFileInfo(filepath, &info).ok() &&
             internal_os::IsRegularFile(info) && internal_os::GetSize(info) > 0;
    };
    auto try_dir = [&info](const std::string& dirpath) {
      return !dirpath.empty() && GetFileInfo(dirpath, &info).ok() &&
             internal_os::IsDirectory(info);
    };

#ifdef TENSORSTORE_LOOKUP_X509_PATHS
    // When not using boringssl, try the default certificate file/directory,
    // as they are likely vendored by the operating system.
    if (std::string default_cert_file(X509_get_default_cert_file());
        try_file(default_cert_file)) {
      return CertFile{std::move(default_cert_file)};
    }
    if (std::string default_cert_dir(X509_get_default_cert_dir());
        try_dir(default_cert_dir)) {
      return CertDirectory{std::move(default_cert_dir)};
    }
#endif
    // Otherwise try common locations for the certificate files/directories.
    for (const char* target : kCertFiles) {
      if (std::string cert_file(target); try_file(cert_file)) {
        return CertFile{std::move(cert_file)};
      }
    }
    for (const char* target : kCertDirectories) {
      if (std::string cert_dir(target); try_dir(cert_dir)) {
        return CertDirectory{std::move(cert_dir)};
      }
    }
    return std::monostate();
  }();

  // Only install the SSL certificates if they were found; otherwise, curl
  // should use the library defaults.
  if (const auto* cert_file = std::get_if<CertFile>(&certs)) {
    ABSL_CHECK_EQ(CURLE_OK, curl_easy_setopt(handle.get(), CURLOPT_CAINFO,
                                             cert_file->cert_file.c_str()));
  } else if (const auto* cert_directory = std::get_if<CertDirectory>(&certs)) {
    ABSL_CHECK_EQ(CURLE_OK,
                  curl_easy_setopt(handle.get(), CURLOPT_CAPATH,
                                   cert_directory->cert_directory.c_str()));
  }
}

}  // namespace internal_http
}  // namespace tensorstore
