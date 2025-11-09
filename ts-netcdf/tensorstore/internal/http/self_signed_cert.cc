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

#include "tensorstore/internal/http/self_signed_cert.h"

#include <stdint.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/x509.h>
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_http {
namespace {

struct EVP_PKEYFree {
  void operator()(EVP_PKEY* pkey) { EVP_PKEY_free(pkey); }
};

struct X509Free {
  void operator()(X509* x509) { X509_free(x509); }
};

struct X509NameFree {
  void operator()(X509_NAME* name) { X509_NAME_free(name); }
};

struct BIOFree {
  void operator()(BIO* bio) { BIO_free(bio); }
};

struct BIGNUMFree {
  void operator()(BIGNUM* bn) { BN_free(bn); }
};

struct RSAFree {
  void operator()(RSA* rsa) { RSA_free(rsa); }
};

std::unique_ptr<X509_NAME, X509NameFree> CreateName(
    const std::vector<std::pair<std::string, std::string>>& name_parts) {
  std::unique_ptr<X509_NAME, X509NameFree> name(X509_NAME_new());
  for (const auto& part : name_parts) {
    X509_NAME_add_entry_by_txt(name.get(), part.first.c_str(), MBSTRING_ASC,
                               (unsigned char*)part.second.c_str(), -1, -1, 0);
  }
  return name;
}

// Append the OpenSSL error strings to the status message.
absl::Status OpenSslError(std::string message,
                          SourceLocation loc = SourceLocation::current()) {
  for (int line = 0; line < 4; ++line) {
    absl::StrAppend(&message, "\n");
    const char* extra_data = nullptr;
    const char* file_name = nullptr;
    int line_number = 0;
    int flags = 0;
    // This extracts the top error code from the error codes stack.
    const uint32_t error_code =
        ERR_get_error_line_data(&file_name, &line_number, &extra_data, &flags);
    if (error_code == 0) {  // No more error codes.
      break;
    }
    if (file_name != nullptr) {
      absl::StrAppend(&message, file_name, ":", line_number, " ");
    }
    const char* reason_error_string = ERR_reason_error_string(error_code);
    if (reason_error_string != nullptr) {
      absl::StrAppend(&message, reason_error_string);
    } else {
      absl::StrAppend(&message, error_code);
    }
    if (extra_data != nullptr && (flags & ERR_TXT_STRING)) {
      absl::StrAppend(&message, " - ", extra_data);
    }
  }
  auto status = absl::InternalError(message);
  MaybeAddSourceLocation(status, loc);
  return status;
}

}  // namespace

Result<SelfSignedCertificate> GenerateSelfSignedCerts() {
  std::unique_ptr<EVP_PKEY, EVP_PKEYFree> pkey(EVP_PKEY_new());

  {
    std::unique_ptr<BIGNUM, BIGNUMFree> bn(BN_new());
    std::unique_ptr<RSA, RSAFree> rsa(RSA_new());

    ABSL_CHECK(BN_set_word(bn.get(), RSA_F4));
    ABSL_CHECK(RSA_generate_key_ex(rsa.get(), 2048, bn.get(), nullptr));
    ABSL_CHECK(EVP_PKEY_assign_RSA(pkey.get(), rsa.release()));
  }

  std::unique_ptr<X509, X509Free> x509(X509_new());

  // TODO: Random serial number?
  ABSL_CHECK(ASN1_INTEGER_set(X509_get_serialNumber(x509.get()), 1));
  ABSL_CHECK(X509_gmtime_adj(X509_get_notBefore(x509.get()), 0));
  ABSL_CHECK(X509_gmtime_adj(X509_get_notAfter(x509.get()), 3600L));  // 1 hour

  if (!X509_set_pubkey(x509.get(), pkey.get())) {
    return OpenSslError("Failed to set public key on certificate");
  }

  auto name = CreateName({
      {"C", "CA"},
      {"O", "Tensorstore Test"},
      {"CN", "localhost"},
  });
  if (!X509_set_issuer_name(x509.get(), name.get())) {
    return OpenSslError("Failed to set issuer name on certificate");
  }

  // Sign the certificate.
  if (!X509_sign(x509.get(), pkey.get(), EVP_sha256())) {
    return OpenSslError("Failed to sign x509 certificate");
  }

  // Output the certificate and private key to PEM.
  std::unique_ptr<BIO, BIOFree> bio_pem(BIO_new(BIO_s_mem()));
  if (!PEM_write_bio_PrivateKey(bio_pem.get(), pkey.get(), nullptr, nullptr, 0,
                                0, nullptr)) {
    return OpenSslError("Failed to generate certificate");
  }

  SelfSignedCertificate result;
  auto& key_pem = result.key_pem;

  key_pem.resize(BIO_pending(bio_pem.get()));
  if (BIO_read(bio_pem.get(), key_pem.data(), key_pem.size()) !=
      key_pem.size()) {
    return OpenSslError("Failed to generate certificate");
  }

  std::unique_ptr<BIO, BIOFree> bio_cert(BIO_new(BIO_s_mem()));
  if (!PEM_write_bio_X509(bio_cert.get(), x509.get())) {
    return OpenSslError("Failed to generate certificate");
  }

  auto& cert_pem = result.cert_pem;
  cert_pem.resize(BIO_pending(bio_cert.get()));
  if (BIO_read(bio_cert.get(), cert_pem.data(), cert_pem.size()) !=
      cert_pem.size()) {
    return OpenSslError("Failed to generate certificate");
  }
  return result;
}

}  // namespace internal_http
}  // namespace tensorstore
