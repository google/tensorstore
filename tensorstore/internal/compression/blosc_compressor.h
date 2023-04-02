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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_COMPRESSOR_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_COMPRESSOR_H_

/// \file Define a blosc JsonSpecifiedCompressor.

#include <cstddef>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include <blosc.h>
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/internal/compression/json_specified_compressor.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

class BloscCompressor : public JsonSpecifiedCompressor {
 public:
  std::unique_ptr<riegeli::Writer> GetWriter(
      std::unique_ptr<riegeli::Writer> base_writer,
      size_t element_bytes) const override;

  std::unique_ptr<riegeli::Reader> GetReader(
      std::unique_ptr<riegeli::Reader> base_reader,
      size_t element_bytes) const override;

  static constexpr auto CodecBinder() {
    namespace jb = tensorstore::internal_json_binding;
    return jb::Validate([](const auto& options, std::string* cname) {
      if (cname->find('\0') != std::string::npos ||
          blosc_compname_to_compcode(cname->c_str()) == -1) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Expected one of ", blosc_list_compressors(),
                                " but received: ", QuoteString(*cname)));
      }
      return absl::OkStatus();
    });
  }

  std::string codec;
  int level;
  int shuffle;
  std::size_t blocksize;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_BLOSC_COMPRESSOR_H_
