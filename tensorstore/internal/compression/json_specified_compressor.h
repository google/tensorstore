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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_JSON_SPECIFIED_COMPRESSOR_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_JSON_SPECIFIED_COMPRESSOR_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_registry_fwd.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal {

/// Abstract base class for compressors that may be specified via a JSON
/// representation.
class JsonSpecifiedCompressor
    : public AtomicReferenceCount<JsonSpecifiedCompressor> {
 public:
  using Ptr = IntrusivePtr<const JsonSpecifiedCompressor>;

  virtual ~JsonSpecifiedCompressor();

  /// Returns a writer that encodes the compression format.
  virtual std::unique_ptr<riegeli::Writer> GetWriter(
      std::unique_ptr<riegeli::Writer> base_writer,
      size_t element_bytes) const = 0;

  /// Returns a reader that decodes the compression format.
  ///
  /// \param base_reader Reader from which encoded input should be read.
  /// \param element_bytes Specifies the element size as a hint to the
  ///     compressor, e.g. `4` if `input` is actually a sequence of `int32_t`
  ///     values.  Must be `> 0`.
  virtual std::unique_ptr<riegeli::Reader> GetReader(
      std::unique_ptr<riegeli::Reader> base_reader,
      size_t element_bytes) const = 0;

  /// Encodes `input`.
  ///
  /// \param input The input data.
  /// \param output[out] Output buffer to which encoded output will be appended.
  ///     The value is unspecified if an error occurs.
  /// \param element_bytes Specifies the element size as a hint to the
  ///     compressor, e.g. `4` if `input` is actually a sequence of `int32_t`
  ///     values.  Must be `> 0`.
  /// \returns `absl::Status()` on success, or an error if encoding fails.
  absl::Status Encode(const absl::Cord& input, absl::Cord* output,
                      size_t element_bytes) const;

  /// Decodes `input`.
  ///
  /// \param input The input data.
  /// \param output[out] Output buffer to which decoded output will be appended.
  ///     The value is unspecified if an error occurs.
  /// \param element_bytes Specifies the element size as a hint to the
  ///     compressor, e.g. `4` if `input` is actually a sequence of `int32_t`
  ///     values.  Must be `> 0`.
  /// \returns `absl::Status()` on success, or an error if decoding fails.
  /// \error `absl::StatusCode::kInvalidArgument` if `input` is invalid.
  absl::Status Decode(const absl::Cord& input, absl::Cord* output,
                      size_t element_bytes) const;

  using ToJsonOptions = JsonSerializationOptions;
  using FromJsonOptions = JsonSerializationOptions;

  using Registry = JsonRegistry<JsonSpecifiedCompressor, FromJsonOptions,
                                ToJsonOptions, Ptr>;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_JSON_SPECIFIED_COMPRESSOR_H_
