// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/image/bmp_reader.h"

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/endian/endian_reading.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/image/image_view.h"

namespace tensorstore {
namespace internal_image {
namespace {

using riegeli::ReadLittleEndian16;
using riegeli::ReadLittleEndian32;
using riegeli::ReadLittleEndianSigned32;

// https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapfileheader
struct BitmapFileHeader {
  uint16_t type = 0;
  uint32_t size = 0;
  uint16_t reserved1 = 0;
  uint16_t reserved2 = 0;
  uint32_t offset = 0;
};

struct BitmapCIEXYZ {
  uint32_t x = 0;  // fixedpoint2.30
  uint32_t y = 0;  // fixedpoint2.30
  uint32_t z = 0;  // fixedpoint2.30
};

// See the following MS documentation
// https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapinfoheader
// https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapv4header
// https://learn.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapv5header
struct BitmapDIBHeader {
  uint32_t header_size = 0;
  int32_t width = 0;
  int32_t height = 0;
  uint16_t planes = 0;
  uint16_t bit_count = 0;
  uint32_t compression = 0;
  uint32_t raw_image_size = 0;
  int32_t xpels_per_meter = 0;
  int32_t ypels_per_meter = 0;
  uint32_t clr_used = 0;  // palette
  uint32_t clr_important = 0;
  // end BITMAPINFOHEADER (40 bytes)
  uint32_t red_mask = 0;
  uint32_t green_mask = 0;
  uint32_t blue_mask = 0;
  uint32_t alpha_mask = 0;
  uint32_t cstype = 0;
  BitmapCIEXYZ endpoints_red;
  BitmapCIEXYZ endpoints_green;
  BitmapCIEXYZ endpoints_blue;
  uint32_t gamma_red = 0;
  uint32_t gamma_green = 0;
  uint32_t gamma_blue = 0;
  // end BITMAPV4HEADER (108 bytes)
  uint32_t intent = 0;
  uint32_t profile_data = 0;
  uint32_t profile_size = 0;
  uint32_t reserved = 0;
  // end BITMAPV5HEADER (124 bytes)
};

static constexpr const int BI_RGB = 0;
static constexpr const int BI_RLE8 = 1;
static constexpr const int BI_RLE4 = 2;

static constexpr const int kMinDIBHeaderSize = 40;
static constexpr const int kMaxDIBHeaderSize = 124;

struct BitmapHeader {
  BitmapFileHeader file_header;
  BitmapDIBHeader dib_header;

  int32_t abs_height;
  size_t bytes_per_row;
};

absl::Status ReadBitmapHeaders(riegeli::Reader* reader, BitmapHeader& header) {
  if (!ReadLittleEndian16(*reader, header.file_header.type) ||
      !ReadLittleEndian32(*reader, header.file_header.size) ||
      !ReadLittleEndian16(*reader, header.file_header.reserved1) ||
      !ReadLittleEndian16(*reader, header.file_header.reserved2) ||
      !ReadLittleEndian32(*reader, header.file_header.offset)) {
    return reader->StatusOrAnnotate(
        absl::InvalidArgumentError("Failed to read BMP file header"));
  }
  if (header.file_header.type != 0x4D42) {
    return absl::InvalidArgumentError("Not a BMP file");
  }

  auto& dib = header.dib_header;
  if (!ReadLittleEndian32(*reader, dib.header_size)) {
    return reader->StatusOrAnnotate(
        absl::InvalidArgumentError("Failed to read BMP DIB header"));
  }
  if (dib.header_size < kMinDIBHeaderSize ||
      dib.header_size > kMaxDIBHeaderSize) {
    return absl::InvalidArgumentError("Failed to read BMP DIB header");
  }

  // Only read header_size bytes from the stream.
  riegeli::LimitingReader hdr_reader(
      reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                  dib.header_size - 4));

  // Read all fields until failure.
  ReadLittleEndianSigned32(hdr_reader, dib.width) &&
      ReadLittleEndianSigned32(hdr_reader, dib.height) &&
      ReadLittleEndian16(hdr_reader, dib.planes) &&
      ReadLittleEndian16(hdr_reader, dib.bit_count) &&
      ReadLittleEndian32(hdr_reader, dib.compression) &&
      ReadLittleEndian32(hdr_reader, dib.raw_image_size) &&
      ReadLittleEndianSigned32(hdr_reader, dib.xpels_per_meter) &&
      ReadLittleEndianSigned32(hdr_reader, dib.ypels_per_meter) &&
      ReadLittleEndian32(hdr_reader, dib.clr_used) &&
      ReadLittleEndian32(hdr_reader, dib.clr_important) &&
      ReadLittleEndian32(hdr_reader, dib.red_mask) &&
      ReadLittleEndian32(hdr_reader, dib.green_mask) &&
      ReadLittleEndian32(hdr_reader, dib.blue_mask) &&
      ReadLittleEndian32(hdr_reader, dib.alpha_mask) &&
      ReadLittleEndian32(hdr_reader, dib.cstype) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_red.x) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_red.y) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_red.z) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_green.x) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_green.y) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_green.z) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_blue.x) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_blue.y) &&
      ReadLittleEndian32(hdr_reader, dib.endpoints_blue.z) &&
      ReadLittleEndian32(hdr_reader, dib.gamma_red) &&
      ReadLittleEndian32(hdr_reader, dib.gamma_green) &&
      ReadLittleEndian32(hdr_reader, dib.gamma_blue) &&
      ReadLittleEndian32(hdr_reader, dib.intent) &&
      ReadLittleEndian32(hdr_reader, dib.profile_data) &&
      ReadLittleEndian32(hdr_reader, dib.profile_size) &&
      ReadLittleEndian32(hdr_reader, dib.reserved);

  if (!hdr_reader.VerifyEndAndClose()) {
    return hdr_reader.status();
  }
  if (!reader->ok()) {
    return reader->status();
  }
  return absl::OkStatus();
}

absl::Status CheckBitmapHeaders(BitmapHeader& header) {
  auto& dib = header.dib_header;
  if (dib.width <= 0) {
    return absl::InvalidArgumentError("Width must be positive");
  }
  if (dib.height == 0) {
    return absl::InvalidArgumentError("Height must be nonzero");
  }

  // `img_channels` is number of channels inherent in the image.
  int img_channels = dib.bit_count / 8;
  if (img_channels != 1 && img_channels != 3 && img_channels != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Number of channels inherent in the image must be 1, 3 or 4, was %d",
        img_channels));
  }
  if (dib.planes != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported BMP configuration; planes = %d", dib.planes));
  }
  if (dib.compression != BI_RGB && dib.compression != BI_RLE4 &&
      dib.compression != BI_RLE8) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported BMP configuration; compression = %d", dib.compression));
  }
  if (dib.raw_image_size == 0 &&
      (dib.compression == BI_RLE4 || dib.compression == BI_RLE8)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unsupported BMP configuration; compression = %d, missing image size",
        dib.compression));
  }
  if (dib.clr_used != 0 && dib.clr_used != 256) {
    return absl::InvalidArgumentError(
        "Unsupported BMP configuration (palette)");
  }

  int32_t abs_height = std::abs(dib.height);
  if (static_cast<int64_t>(dib.width) * static_cast<int64_t>(abs_height) >
      static_cast<int64_t>(std::numeric_limits<int32_t>::max() / 8)) {
    return absl::InvalidArgumentError("Total size is too large.");
  }

  const size_t bytes_per_row = (img_channels * dib.width + 3) / 4 * 4;

  header.bytes_per_row = bytes_per_row;
  header.abs_height = abs_height;
  return absl::OkStatus();
}

ImageInfo BmpGetImageInfo(const BitmapHeader& header) {
  ImageInfo info;
  info.height = header.abs_height;
  info.width = header.dib_header.width;
  info.num_components = header.dib_header.bit_count / 8;
  return info;
}

absl::Status BmpDefaultDecodeRAW(riegeli::Reader* reader, const ImageInfo& info,
                                 const BitmapHeader& header,
                                 ImageView& dest_view) {
  size_t pos = header.file_header.offset;

  // The rows are read in file-order. If height is negative, data layout is
  // top down otherwise, it's bottom up.
  bool top_down = (header.dib_header.height < 0);
  for (int y = 0; y < info.height; ++y) {
    if (!reader->Seek(pos)) {
      // pos should always be >= reader->pos()
      return reader->StatusOrAnnotate(
          absl::InvalidArgumentError("Cannot read BMP; Seek failed"));
    }
    pos += header.bytes_per_row;

    auto output = dest_view.data_row(top_down ? y : info.height - y - 1);
    if (!reader->Read(output.size(), reinterpret_cast<char*>(output.data()))) {
      return reader->StatusOrAnnotate(
          absl::InvalidArgumentError("Cannot read BMP; Read failed"));
    }
  }

  if (!reader->Seek(pos + header.dib_header.profile_size)) {
    return reader->StatusOrAnnotate(
        absl::InvalidArgumentError("Cannot read BMP; Seek failed"));
  }
  return absl::OkStatus();
}

// Decodes RLE4 and RLE8 encoded BMP images. For documentation of the format,
// see the following urls:
// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-wmf/73b57f24-6d78-4eeb-9c06-8f892d88f1ab
// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-wmf/b64d0c0b-bb80-4b53-8382-f38f264eb685
absl::Status BmpDefaultDecodeRLE(riegeli::Reader* reader, const ImageInfo& info,
                                 const BitmapHeader& header,
                                 ImageView& dest_view) {
  if (!reader->Seek(header.file_header.offset)) {
    return reader->StatusOrAnnotate(
        absl::InvalidArgumentError("Cannot read BMP; Seek failed"));
  }

  /// RLE allows skipping bytes, so pre-zero the buffer.
  memset(dest_view.data().data(), 0, dest_view.data().size());

  riegeli::LimitingReader bits_reader(
      reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                  header.dib_header.raw_image_size));

  auto read_byte = [&bits_reader] {
    uint8_t b = static_cast<uint8_t>(*bits_reader.cursor());
    bits_reader.move_cursor(1);
    return b;
  };

  const bool top_down = (header.dib_header.height < 0);
  size_t x = 0;
  size_t y = 0;
  tensorstore::span<unsigned char> data_row =
      dest_view.data_row(top_down ? y : info.height - y - 1);

  for (;;) {
    // Each RLE starts with 2 bytes, nominally a count and a value.
    // The count *may* exceed the remaining line length,
    if (!bits_reader.Pull(2)) break;
    size_t count = read_byte();
    if (count > 0) {
      /// We have a RLE value. RLE encoding doesn't cross line boundaries,
      /// but the count may be larger than the remaining line.
      if (count > data_row.size() - x) {
        count = data_row.size() - x;
      }
      uint8_t value = read_byte();
      if (header.dib_header.compression == BI_RLE8) {
        for (size_t i = 0; i < count; i++) {
          data_row[x++] = value;
        }
      } else if (header.dib_header.compression == BI_RLE4) {
        for (size_t i = 0; i < count; i++) {
          data_row[x++] = ((i & 0x1) ? (value >> 4) : value) & 0xf;
        }
      }
    } else {
      /// Otherwise we're in escape mode, controls the RLE stream
      uint8_t escaped_count = read_byte();
      if (escaped_count == 0) {
        // end of line.
        y++;
        x = 0;
        if (y < info.height) {
          data_row = dest_view.data_row(top_down ? y : info.height - y - 1);
        }
        continue;
      }
      if (escaped_count == 1) {
        // end of bitmap.
        break;
      }
      if (escaped_count == 2) {
        // delta mode advances the entire encoding by a jump;
        if (!bits_reader.Pull(2)) break;

        auto delta_x = read_byte();
        auto delta_y = read_byte();
        y += delta_y;
        x += delta_x;

        if (x >= info.width || y > info.height) {
          return absl::DataLossError("RLE encoding delta appears corrupt");
        }
        if (y < info.height) {
          data_row = dest_view.data_row(top_down ? y : info.height - y - 1);
        }
        continue;
      }

      // Otherwise the following bytes are absolute values.
      if (escaped_count > data_row.size()) {
        return absl::DataLossError(absl::StrFormat(
            "RLE encoding absolute mode appears corrupt %d > %d", escaped_count,
            data_row.size()));
      }
      if (!bits_reader.Pull(escaped_count)) break;
      bool align_to_word = false;

      if (header.dib_header.compression == BI_RLE8) {
        align_to_word = (escaped_count & 0x1);
        for (size_t i = 0; i < escaped_count; i++) {
          if (x < data_row.size()) {
            data_row[x++] = read_byte();
          } else {
            bits_reader.Skip(1);
          }
        }
      } else if (header.dib_header.compression == BI_RLE4) {
        align_to_word =
            ((escaped_count & 0x3) == 1 || (escaped_count & 0x3) == 2);
        unsigned char value;
        for (size_t i = 0; i < escaped_count; i++) {
          if ((i & 0x1) == 0) value = read_byte();
          if (x < data_row.size()) {
            data_row[x++] = ((i & 0x1) ? (value >> 4) : value) & 0xf;
          }
        }
      }
      // Absolute mode always ends on a WORD boundary.
      if (align_to_word) {
        if (!bits_reader.Skip(1)) {
          break;
        }
      }
    }
  }

  if (!bits_reader.VerifyEndAndClose()) {
    return bits_reader.status();
  }

  if (!reader->Skip(header.dib_header.profile_size)) {
    return reader->StatusOrAnnotate(
        absl::InvalidArgumentError("Cannot read BMP; Seek failed"));
  }
  return absl::OkStatus();
}

absl::Status BmpDefaultDecode(riegeli::Reader* reader,
                              const BitmapHeader& header,
                              tensorstore::span<unsigned char> dest) {
  auto info = BmpGetImageInfo(header);
  if (auto required = ImageRequiredBytes(info); required > dest.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot read BMP; required buffer size %d, got %d",
                        required, dest.size()));
  }
  ImageView dest_view(info, dest);

  if (header.dib_header.compression == BI_RGB) {
    TENSORSTORE_RETURN_IF_ERROR(
        BmpDefaultDecodeRAW(reader, info, header, dest_view));
  } else if (header.dib_header.compression == BI_RLE8 ||
             header.dib_header.compression == BI_RLE4) {
    TENSORSTORE_RETURN_IF_ERROR(
        BmpDefaultDecodeRLE(reader, info, header, dest_view));
  }
  if (!reader->ok()) return reader->status();

  for (int y = 0; y < info.height; ++y) {
    auto output = dest_view.data_row(y);
    if (info.num_components == 3) {
      // BGR -> RGB
      for (int x = 0; x < info.width; x++) {
        std::swap(output[x * 3], output[x * 3 + 2]);
      }
    } else if (info.num_components == 4) {
      // BGRA -> RGBA
      for (int x = 0; x < info.width; x++) {
        std::swap(output[x * 4], output[x * 4 + 2]);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

struct BmpReader::BmpHeader : public BitmapHeader {};

BmpReader::BmpReader() = default;
BmpReader::~BmpReader() = default;
BmpReader::BmpReader(BmpReader&& src) = default;
BmpReader& BmpReader::operator=(BmpReader&& src) = default;

absl::Status BmpReader::Initialize(riegeli::Reader* reader) {
  ABSL_CHECK(reader != nullptr);
  reader_ = reader;

  auto header = std::make_unique<BmpHeader>();
  TENSORSTORE_RETURN_IF_ERROR(ReadBitmapHeaders(reader_, *header));

  auto status = CheckBitmapHeaders(*header);
  if (!status.ok()) {
    return status;
  }
  header_ = std::move(header);
  return absl::OkStatus();
}

ImageInfo BmpReader::GetImageInfo() {
  if (!header_) return {};
  return BmpGetImageInfo(*header_);
}

absl::Status BmpReader::DecodeImpl(tensorstore::span<unsigned char> dest,
                                   const BmpReaderOptions& options) {
  if (!header_) {
    return absl::InternalError("No BMP file to decode");
  }

  std::unique_ptr<BmpHeader> header = std::move(header_);
  return BmpDefaultDecode(reader_, *header, dest);
}

}  // namespace internal_image
}  // namespace tensorstore
