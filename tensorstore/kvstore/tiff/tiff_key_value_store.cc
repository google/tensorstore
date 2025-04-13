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

// -----------------------------------------------------------------------------
// TIFF key‑value‑store adapter
//   * read‑only
//   * validates the 8‑byte header during DoOpen
//   * all other operations are simple pass‑through for now
// -----------------------------------------------------------------------------

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <cstring>
#include <array>
#include <vector>
#include <optional>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/strip.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore::kvstore::tiff_kvstore {
namespace jb = ::tensorstore::internal_json_binding;

// ─────────────────────────────────────────────────────────────────────────────
//  Endian helpers + header parser
// ─────────────────────────────────────────────────────────────────────────────
enum class Endian { kLittle, kBig };

inline uint16_t Read16(const char* p, Endian e) {
  return e == Endian::kLittle
             ? uint16_t(uint8_t(p[0])) | (uint16_t(uint8_t(p[1])) << 8)
             : uint16_t(uint8_t(p[1])) | (uint16_t(uint8_t(p[0])) << 8);
}

inline uint32_t Read32(const char* p, Endian e) {
  if (e == Endian::kLittle)
    return uint32_t(uint8_t(p[0])) | (uint32_t(uint8_t(p[1])) << 8) |
           (uint32_t(uint8_t(p[2])) << 16) | (uint32_t(uint8_t(p[3])) << 24);
  return uint32_t(uint8_t(p[3])) | (uint32_t(uint8_t(p[2])) << 8) |
         (uint32_t(uint8_t(p[1])) << 16) | (uint32_t(uint8_t(p[0])) << 24);
}

enum Tag : uint16_t {
  kImageWidth        = 256,
  kImageLength       = 257,
  kBitsPerSample     = 258,
  kCompression       = 259,
  kPhotometric       = 262,
  kStripOffsets      = 273,
  kRowsPerStrip      = 278,
  kStripByteCounts   = 279,
  kTileWidth         = 322,
  kTileLength        = 323,
  kTileOffsets       = 324,
  kTileByteCounts    = 325,
};

enum Type : uint16_t { kBYTE=1, kSHORT=3, kLONG=4 };

inline size_t TypeSize(Type t) {
  switch(t) {
    case kBYTE:  return 1;
    case kSHORT: return 2;
    case kLONG:  return 4;
    default:     return 0;
  }
}

struct IfdEntry {
  Tag      tag;
  Type     type;
  uint32_t count;
  uint32_t value_or_offset;   // raw
};

struct TiffHeader {
  Endian   endian;
  uint32_t first_ifd_offset;
};

struct ImageDirectory {
  // ───────── raw tags we keep ─────────
  uint32_t width            = 0;
  uint32_t height           = 0;
  uint32_t tile_width       = 0;          // 0 ⇒ striped
  uint32_t tile_length      = 0;          // 0 ⇒ striped
  uint32_t rows_per_strip   = 0;          // striped only
  std::vector<uint64_t> chunk_offsets;     // tile or strip
  std::vector<uint64_t> chunk_bytecounts;  // tile or strip
  bool      tiled           = false;

  // ───────── derived, filled after parsing ─────────
  uint32_t chunk_rows = 0;    // number of chunk rows
  uint32_t chunk_cols = 0;    // number of chunk cols
};

template <typename T>
static inline T CeilDiv(T a, T b) { return (a + b - 1) / b; }

inline absl::Status ParseHeader(const absl::Cord& c, TiffHeader& hdr) {
  if (c.size() < 8) return absl::DataLossError("Header truncated (<8 bytes)");
  char buf[8];
  std::string tmp(c.Subcord(0, 8));   // makes a flat copy of those 8 bytes
  std::memcpy(buf, tmp.data(), 8);

  if (buf[0] == 'I' && buf[1] == 'I')
    hdr.endian = Endian::kLittle;
  else if (buf[0] == 'M' && buf[1] == 'M')
    hdr.endian = Endian::kBig;
  else
    return absl::InvalidArgumentError("Bad byte‑order mark");

  if (Read16(buf + 2, hdr.endian) != 42)
    return absl::InvalidArgumentError("Missing 42 magic");

  hdr.first_ifd_offset = Read32(buf + 4, hdr.endian);
  return absl::OkStatus();
}

inline absl::Status ParseIfd(const absl::Cord& c,
                             size_t ifd_offset,
                             Endian e,
                             ImageDirectory& out) {
  // 1. copy 2 bytes count
  if (c.size() < ifd_offset + 2)
    return absl::DataLossError("IFD truncated (count)");
  char cnt_buf[2];
  std::string tmp(c.Subcord(0, 2));
  std::memcpy(cnt_buf, tmp.data(), 2);
//  c.CopyTo(cnt_buf, ifd_offset, 2);
  uint16_t entry_count = Read16(cnt_buf, e);

  // 2. copy entries (12 bytes each)
  size_t table_size = size_t(entry_count) * 12;
  if (c.size() < ifd_offset + 2 + table_size + 4)
    return absl::DataLossError("IFD truncated (entries)");

  std::string table(c.Subcord(ifd_offset + 2, table_size));
  const char* p = table.data();
  std::vector<IfdEntry> entries;
  entries.reserve(entry_count);
  for (uint16_t i=0;i<entry_count;++i, p+=12) {
    IfdEntry e2;
    e2.tag  = Tag(Read16(p, e));
    e2.type = Type(Read16(p+2, e));
    e2.count = Read32(p+4, e);
    e2.value_or_offset = Read32(p+8, e);
    entries.push_back(e2);
  }

  // helpers
  auto find = [&](Tag t)->const IfdEntry*{
    for(auto& v:entries) if (v.tag==t) return &v;
    return nullptr;
  };
  auto fetch_scalar = [&](Tag t, uint32_t* dst)->absl::Status{
    auto* ent=find(t);
    if(!ent) return absl::NotFoundError("Missing tag");
    if(ent->count!=1) return absl::InvalidArgumentError("Bad count");
    if(ent->type==kSHORT) *dst = ent->value_or_offset & 0xFFFFu;
    else if(ent->type==kLONG) *dst = ent->value_or_offset;
    else return absl::InvalidArgumentError("Unexpected type");
    return absl::OkStatus();
  };

  TENSORSTORE_RETURN_IF_ERROR(fetch_scalar(kImageWidth , &out.width ));
  TENSORSTORE_RETURN_IF_ERROR(fetch_scalar(kImageLength, &out.height));

  // Decide tiled vs strips
  if (find(kTileOffsets)) {
    out.tiled = true;
    TENSORSTORE_RETURN_IF_ERROR(fetch_scalar(kTileWidth , &out.tile_width ));
    TENSORSTORE_RETURN_IF_ERROR(fetch_scalar(kTileLength, &out.tile_length));
  } else {
    out.tiled = false;
    TENSORSTORE_RETURN_IF_ERROR(fetch_scalar(kRowsPerStrip, &out.rows_per_strip));
  }
  
  // Offsets & byte counts
  auto load_array = [&](const IfdEntry* ent,
                        std::vector<uint64_t>* vec)->absl::Status{
    if(!ent) return absl::NotFoundError("Missing required tag");
    size_t elem_sz = TypeSize(ent->type);
    if(!(ent->type==kSHORT || ent->type==kLONG))
      return absl::InvalidArgumentError("Unsupported type in array");
    size_t total = size_t(ent->count)*elem_sz;
    size_t src_off = (ent->count==1 && total<=4)
                     ? std::numeric_limits<size_t>::max()  // value in place
                     : ent->value_or_offset;
    std::string buf;
    if(src_off==std::numeric_limits<size_t>::max()) {
      buf.assign(reinterpret_cast<const char*>(&ent->value_or_offset),4);
    } else {
      if(c.size()<src_off+total)
        return absl::DataLossError("Array out of file bounds");
      buf.assign(std::string(c.Subcord(src_off, total)));
    }
    vec->resize(ent->count);
    for(uint32_t i=0;i<ent->count;++i) {
      if(ent->type==kSHORT)
        (*vec)[i] = Read16(buf.data()+i*elem_sz,e);
      else
        (*vec)[i] = Read32(buf.data()+i*elem_sz,e);
    }

    return absl::OkStatus();
  };

  TENSORSTORE_RETURN_IF_ERROR(
      load_array(find(out.tiled?kTileOffsets:kStripOffsets), &out.chunk_offsets));
  TENSORSTORE_RETURN_IF_ERROR(
      load_array(find(out.tiled?kTileByteCounts:kStripByteCounts),
                 &out.chunk_bytecounts));

  if(out.chunk_offsets.size()!=out.chunk_bytecounts.size())
    return absl::InvalidArgumentError("Offsets/ByteCounts length mismatch");

  // ------------------------------------------------------------------
  // Consistency & derived values
  // ------------------------------------------------------------------
  if (out.tiled) {
    out.chunk_cols = CeilDiv(out.width , out.tile_width );
    out.chunk_rows = CeilDiv(out.height, out.tile_length);
  } else {                           // striped
    out.tile_width  = out.width;     // pretend full‑width tiles
    out.tile_length = out.rows_per_strip;
    out.chunk_cols  = 1;
    out.chunk_rows  = out.chunk_offsets.size();
  }

  return absl::OkStatus();
}

// Expected key: "tile/<ifd>/<row>/<col>"
absl::Status ParseTileKey(std::string_view key,
                          uint32_t& ifd, uint32_t& row, uint32_t& col) {
  auto eat_number = [&](std::string_view& s, uint32_t& out) -> bool {
    if (s.empty()) return false;
    uint32_t v = 0;
    size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      v = v * 10 + (s[i] - '0');
      ++i;
    }
    if (i == 0) return false;           // no digits
    out = v;
    s.remove_prefix(i);
    return true;
  };

  if (!absl::ConsumePrefix(&key, "tile/")) {
    return absl::InvalidArgumentError("Key must start with \"tile/\"");
  }
  if (!eat_number(key, ifd) || !absl::ConsumePrefix(&key, "/") ||
      !eat_number(key, row) || !absl::ConsumePrefix(&key, "/") ||
      !eat_number(key, col) || !key.empty()) {
    return absl::InvalidArgumentError("Bad tile key format");
  }
  return absl::OkStatus();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Spec‑data (JSON parameters)
// ─────────────────────────────────────────────────────────────────────────────
struct TiffKvStoreSpecData {
  kvstore::Spec base;
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource> data_copy;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.base, x.cache_pool, x.data_copy);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base", jb::Projection<&TiffKvStoreSpecData::base>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&TiffKvStoreSpecData::cache_pool>()),
      jb::Member(internal::DataCopyConcurrencyResource::id,
                 jb::Projection<&TiffKvStoreSpecData::data_copy>()));
};

// ─────────────────────────────────────────────────────────────────────────────
//  Spec
// ─────────────────────────────────────────────────────────────────────────────
struct Spec
    : public internal_kvstore::RegisteredDriverSpec<Spec,
                                                    TiffKvStoreSpecData> {
  static constexpr char id[] = "tiff";

  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status ApplyOptions(kvstore::DriverSpecOptions&& o) override {
    return data_.base.driver.Set(std::move(o));
  }
  Result<kvstore::Spec> GetBase(std::string_view) const override {
    return data_.base;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Driver
// ─────────────────────────────────────────────────────────────────────────────
class TiffKeyValueStore
    : public internal_kvstore::RegisteredDriver<TiffKeyValueStore, Spec> {
 public:
 Future<ReadResult> Read(Key key, ReadOptions opts) {
    uint32_t ifd, row, col;
    if (auto st = ParseTileKey(key, ifd, row, col); !st.ok()) {
      return MakeReadyFuture<ReadResult>(st);          // fast fail
    }

    // 1. Bounds‑check against cached first IFD info
    if (ifd != 0) {   // we only cached IFD 0 so far
      return MakeReadyFuture<ReadResult>(
          absl::UnimplementedError("Only IFD 0 implemented"));
    }
    if (row >= first_ifd_.chunk_rows || col >= first_ifd_.chunk_cols) {
      return MakeReadyFuture<ReadResult>(
          absl::OutOfRangeError("Tile/strip index out of range"));
    }

    // 2. Compute byte range
    size_t tile_index   = row * first_ifd_.chunk_cols + col;
    uint64_t offset     = first_ifd_.chunk_offsets[tile_index];
    uint64_t byte_count = first_ifd_.chunk_bytecounts[tile_index];

    ReadOptions ro;
    ro.byte_range = OptionalByteRangeRequest::Range(offset, offset + byte_count);
    ro.staleness_bound = opts.staleness_bound;  // propagate

    return base_.driver->Read(base_.path, std::move(ro));
  }

  // ------------------------------------------------------------------
  // List  (unchanged)
  // ------------------------------------------------------------------
  void ListImpl(ListOptions options, ListReceiver receiver) override {
    options.range = KeyRange::AddPrefix(base_.path, options.range);
    base_.driver->ListImpl(std::move(options), std::move(receiver));
  }

  // ------------------------------------------------------------------
  // Misc helpers
  // ------------------------------------------------------------------
  std::string DescribeKey(std::string_view key) override {
    return StrCat(QuoteString(key), " in ",
                  base_.driver->DescribeKey(base_.path));
  }
  SupportedFeatures GetSupportedFeatures(const KeyRange& r) const override {
    return base_.driver->GetSupportedFeatures(
        KeyRange::AddPrefix(base_.path, r));
  }
  Result<KvStore> GetBase(std::string_view, const Transaction& t) const override {
    return KvStore(base_.driver, base_.path, t);
  }
  const Executor& executor() const { return spec_data_.data_copy->executor; }

  absl::Status GetBoundSpecData(TiffKvStoreSpecData& spec) const {
    spec = spec_data_;
    return absl::OkStatus();
  }

  // ------------------------------------------------------------------
  // Data members
  // ------------------------------------------------------------------
  TiffKvStoreSpecData spec_data_;
  kvstore::KvStore    base_;

  // Newly stored header information
  absl::Cord header_raw_;
  TiffHeader header_parsed_;
  ImageDirectory first_ifd_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Spec::DoOpen  (now reads & validates the 8‑byte header)
// ─────────────────────────────────────────────────────────────────────────────
Future<kvstore::DriverPtr> Spec::DoOpen() const {
  // 1. Open the underlying kvstore.
  auto base_future = kvstore::Open(data_.base);

  // 2. Once base opens, issue an 8‑byte range read, validate, then build driver.
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const Spec>(this)](
          kvstore::KvStore& base_kv) mutable -> Future<kvstore::DriverPtr> {
        // ---- read first 8 bytes
        ReadOptions hdr_opt;
        hdr_opt.byte_range =
            OptionalByteRangeRequest::Range(0, 8);  // header only
        auto hdr_future =
            base_kv.driver->Read(base_kv.path, std::move(hdr_opt));
        
            // ---- parse & construct driver
        return MapFutureValue(
            InlineExecutor{},
            [spec, base_kv](const ReadResult& hdr_rr)
                -> Future<kvstore::DriverPtr> {
              TiffHeader hdr;
              TENSORSTORE_RETURN_IF_ERROR(ParseHeader(hdr_rr.value, hdr));

              // Read 2‑byte count first
              ReadOptions cnt_opt;
              cnt_opt.byte_range =
                  OptionalByteRangeRequest::Range(hdr.first_ifd_offset, hdr.first_ifd_offset+2);
              auto cnt_future =
                  base_kv.driver->Read(base_kv.path, cnt_opt);

              return MapFutureValue(
                  InlineExecutor{},
                  [spec, base_kv, hdr, hdr_rr](const ReadResult& cnt_rr)
                      -> Future<kvstore::DriverPtr> {
                    
                    uint16_t n_entries =
                        Read16(std::string(cnt_rr.value).data(), hdr.endian);
                    size_t ifd_bytes = 2 + size_t(n_entries)*12 + 4;

                    ReadOptions ifd_opt;
                    ifd_opt.byte_range = OptionalByteRangeRequest::Range(
                        hdr.first_ifd_offset, hdr.first_ifd_offset + ifd_bytes);
                    auto ifd_future =
                        base_kv.driver->Read(base_kv.path, ifd_opt);

                    return MapFutureValue(
                        InlineExecutor{},
                        [spec, base_kv, hdr, hdr_rr](const ReadResult& ifd_rr)
                            -> Result<kvstore::DriverPtr> {
                          ImageDirectory dir;
                          TENSORSTORE_RETURN_IF_ERROR(
                              ParseIfd(ifd_rr.value, 0, hdr.endian, dir));

                          // Construct driver
                          auto drv   = internal::MakeIntrusivePtr<TiffKeyValueStore>();
                          drv->base_ = base_kv;
                          drv->spec_data_ = spec->data_;
                          drv->header_raw_    = hdr_rr.value;
                          drv->header_parsed_ = hdr;
                          drv->first_ifd_     = std::move(dir);
                          ABSL_LOG_IF(INFO, tiff_logging) << "TIFF open: "
                                         << drv->first_ifd_.width << "x"
                                         << drv->first_ifd_.height
                                         << (drv->first_ifd_.tiled?" tiled":" stripped");
                          return kvstore::DriverPtr(drv);
                        },
                        ifd_future);
                  },
                  cnt_future);
            },
            std::move(hdr_future));
      },
      std::move(base_future));
}

// ─────────────────────────────────────────────────────────────────────────────
//  GC declaration (driver holds no GC‑relevant objects)
// ─────────────────────────────────────────────────────────────────────────────
}  // namespace tensorstore::kvstore::tiff_kvstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::kvstore::tiff_kvstore::TiffKeyValueStore)

// ─────────────────────────────────────────────────────────────────────────────
//  Registration
// ─────────────────────────────────────────────────────────────────────────────
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::kvstore::tiff_kvstore::Spec>
    registration;
}  // namespace
