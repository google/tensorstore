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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <string>
#include <string_view>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/rand.h>
#include "python/tensorstore/python_imports.h"
#include "python/tensorstore/status.h"

namespace tensorstore {
namespace internal_python {

namespace {

namespace py = ::pybind11;

/// Per-process keys that protect the integrity of pickled status payloads.
///
/// Since unpickling is not safe to invoke on untrusted input, two security
/// measures are employed to ensure we don't accidentally accept a payload
/// received, e.g. via some RPC call.
///
/// 1. The payload is stored under a random id.  This ensures that unpickling
///    cannot be triggered without access to the id.  However, there is a
///    possibility that the payload id is leaked to another process, e.g. if an
///    `absl::Status` that contains a pickled Python exception payload happens
///    to be serialized and sent over an RPC call.
///
/// 2. The payload is authenticated using HMAC-SHA256 with a random per-process
///    HMAC key.  This ensures that only exceptions previously pickled by this
///    process will be unpickled.  This does not protect against replay attacks,
///    but unpickling legitimate exception values is unlikely to have harmful
///    side effects.
struct StatusPayloadKeys {
  StatusPayloadKeys() { ABSL_CHECK_EQ(1, RAND_bytes(keys, kTotalKeyLength)); }

  /// Size of key used as the payload identifier.
  constexpr static size_t kPayloadIdSize = 32;

  /// Size of HMAC-SHA256 key used to prevent untrusted payloads from being
  /// unpickled.
  constexpr static size_t kHmacKeySize = 32;

  /// Size of HMAC (size of SHA256 digest).
  constexpr static size_t kHmacSize = 32;

  /// HMAC key followed by payload id.
  constexpr static size_t kTotalKeyLength = kHmacKeySize + kPayloadIdSize;

  unsigned char keys[kTotalKeyLength];

  std::string_view payload_id() const {
    return std::string_view(reinterpret_cast<const char*>(&keys[kHmacKeySize]),
                            kPayloadIdSize);
  }

  void ComputeHmac(std::string_view message,
                   unsigned char (&hmac)[kHmacSize]) const {
    unsigned int md_len = kHmacSize;
    // Computing HMAC should never fail.
    ABSL_CHECK(HMAC(EVP_sha256(), keys, kHmacKeySize,
                    reinterpret_cast<const unsigned char*>(message.data()),
                    message.size(), hmac, &md_len) &&
               md_len == kHmacSize);
  }

  /// Validates that `payload` begins with the expected MAC (message
  /// authentication code) for the remaining message.
  ///
  /// Sets `payload` to the message.
  ///
  /// \returns `true` if valid, `false` if invalid.
  bool ExtractValidPayload(std::string_view& payload) const {
    if (payload.size() < kHmacSize) return false;
    unsigned char expected_hmac[kHmacSize];
    auto* actual_hmac = reinterpret_cast<const unsigned char*>(payload.data());
    payload.remove_prefix(kHmacSize);
    ComputeHmac(payload, expected_hmac);
    return CRYPTO_memcmp(actual_hmac, expected_hmac, kHmacSize) == 0;
  }

  /// Returns `HMAC(payload) + payload`.
  absl::Cord AddMac(std::string_view payload) const {
    absl::Cord result;
    unsigned char hmac[kHmacSize];
    ComputeHmac(payload, hmac);
    result.Append(
        std::string_view(reinterpret_cast<const char*>(&hmac[0]), kHmacSize));
    result.Append(payload);
    return result;
  }
};

const StatusPayloadKeys& GetStatusPayloadKeys() {
  static const StatusPayloadKeys keys;
  return keys;
}

py::object GetExceptionFromStatus(const absl::Status& status) noexcept {
  assert(!status.ok());
  auto& keys = GetStatusPayloadKeys();
  auto payload = status.GetPayload(keys.payload_id());
  if (!payload) return {};
  auto flattened = payload->Flatten();
  if (!keys.ExtractValidPayload(flattened)) return {};
  // Note: Any `false` condition in the sequence below means the Python error
  // indicator has been set, and we must clear it.
  if (auto flattened_bytes = py::reinterpret_steal<py::object>(
          PyBytes_FromStringAndSize(flattened.data(), flattened.size()))) {
    if (auto exc =
            py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
                python_imports.pickle_loads_function.ptr(),
                flattened_bytes.ptr(), nullptr))) {
      return exc;
    }
  }
  PyErr_Clear();
  return {};
}

}  // namespace

pybind11::handle GetExceptionType(absl::StatusCode error_code,
                                  StatusExceptionPolicy policy) {
  switch (error_code) {
    case absl::StatusCode::kInvalidArgument:
    case absl::StatusCode::kOutOfRange:
      if (policy == StatusExceptionPolicy::kIndexError) {
        return PyExc_IndexError;
      } else {
        return PyExc_ValueError;
      }
    default:
      break;
  }
  return PyExc_ValueError;
}

class DynamicPythonException : public pybind11::builtin_exception {
 public:
  DynamicPythonException(pybind11::handle type, const std::string& what = "")
      : pybind11::builtin_exception(what), type_(type) {}
  void set_error() const override {
    PyErr_SetString(type_.ptr(), this->what());
  }

 private:
  pybind11::handle type_;
};

void ThrowStatusException(const absl::Status& status,
                          StatusExceptionPolicy policy) {
  if (status.ok()) return;
  if (auto exc = GetExceptionFromStatus(status); exc.ptr()) {
    PyErr_SetObject(reinterpret_cast<PyObject*>(exc.ptr()->ob_type), exc.ptr());
    throw py::error_already_set();
  }
  throw DynamicPythonException(GetExceptionType(status.code(), policy),
                               std::string{status.message()});
}

void SetErrorIndicatorFromStatus(const absl::Status& status,
                                 StatusExceptionPolicy policy) {
  assert(status.ok());
  if (auto exc = GetExceptionFromStatus(status); exc.ptr()) {
    PyErr_SetObject(reinterpret_cast<PyObject*>(exc.ptr()->ob_type), exc.ptr());
    return;
  }
  std::string_view message = status.message();
  if (py::object python_message = py::reinterpret_steal<py::object>(
          PyUnicode_FromStringAndSize(message.data(), message.size()))) {
    PyErr_SetObject(GetExceptionType(status.code(), policy).ptr(),
                    python_message.ptr());
  }
}

pybind11::object GetStatusPythonException(const absl::Status& status,
                                          StatusExceptionPolicy policy) {
  if (status.ok()) return pybind11::none();
  if (auto exc = GetExceptionFromStatus(status); exc.ptr()) {
    return exc;
  }
  return GetExceptionType(status.code(), policy)(status.ToString());
}

absl::Status GetStatusFromPythonException(pybind11::handle exc) noexcept {
  py::object exc_value;
  if (!exc.ptr()) {
    // Convert the current exception.
    py::object exc_type, exc_traceback;
    PyErr_Fetch(&exc_type.ptr(), &exc_value.ptr(), &exc_traceback.ptr());
    PyErr_NormalizeException(&exc_type.ptr(), &exc_value.ptr(),
                             &exc_traceback.ptr());
    assert(exc_value.ptr());
    exc = exc_value;
  }
  try {
    auto& keys = GetStatusPayloadKeys();
    py::bytes buf = python_imports.pickle_dumps_function(exc);
    absl::Status status = absl::InternalError("Python error");
    status.SetPayload(keys.payload_id(), keys.AddMac(std::string_view(
                                             PyBytes_AS_STRING(buf.ptr()),
                                             PyBytes_GET_SIZE(buf.ptr()))));
    return status;
  } catch (...) {
    return absl::UnknownError("Unpicklable Python error");
  }
}

}  // namespace internal_python
}  // namespace tensorstore
