// tensorstore/kvstore/tiff/tiff_key_value_store_test.cc
//
// Tests for the TIFF kv‑store adapter, patterned after
// zip_key_value_store_test.cc.

#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"

#include <string>

#include "absl/strings/cord.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "tensorstore/context.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace kvstore = tensorstore::kvstore;
using ::tensorstore::Context;
using ::tensorstore::MatchesStatus;

/* -------------------------------------------------------------------------- */
/*                       Little‑endian byte helpers                           */
/* -------------------------------------------------------------------------- */
void PutLE16(std::string& dst, uint16_t v) {
  dst.push_back(static_cast<char>(v & 0xff));
  dst.push_back(static_cast<char>(v >> 8));
}
void PutLE32(std::string& dst, uint32_t v) {
  dst.push_back(static_cast<char>(v & 0xff));
  dst.push_back(static_cast<char>(v >> 8));
  dst.push_back(static_cast<char>(v >> 16));
  dst.push_back(static_cast<char>(v >> 24));
}

/* -------------------------------------------------------------------------- */
/*                     Minimal TIFF byte‑string builders                      */
/* -------------------------------------------------------------------------- */

// 512 × 512 image, one 256 × 256 tile at offset 128, payload “DATA”.
std::string MakeTinyTiledTiff() {
  std::string t;
  t += "II"; PutLE16(t, 42); PutLE32(t, 8);     // header

  PutLE16(t, 6);                                // 6 IFD entries
  auto E=[&](uint16_t tag,uint16_t type,uint32_t cnt,uint32_t val){
    PutLE16(t,tag); PutLE16(t,type); PutLE32(t,cnt); PutLE32(t,val);};
  E(256,3,1,512); E(257,3,1,512);               // width, length
  E(322,3,1,256); E(323,3,1,256);               // tile width/length
  E(324,4,1,128); E(325,4,1,4);                 // offset/bytecount
  PutLE32(t,0);                                 // next IFD

  if (t.size() < 128) t.resize(128,'\0');
  t += "DATA";
  return t;
}

std::string MakeTinyStripedTiff() {
  std::string t;

  // TIFF header
  t += "II"; PutLE16(t, 42); PutLE32(t, 8);

  // IFD
  PutLE16(t, 5);  // 5 IFD entries
  auto E=[&](uint16_t tag,uint16_t type,uint32_t cnt,uint32_t val){
    PutLE16(t,tag); PutLE16(t,type); PutLE32(t,cnt); PutLE32(t,val);};

  // entries
  E(256, 3, 1, 4);    // ImageWidth = 4
  E(257, 3, 1, 8);    // ImageLength = 8
  E(278, 3, 1, 8);    // RowsPerStrip = 8  (entire image = 1 strip)
  E(273, 4, 1, 128);  // StripOffsets = 128 (pointing to the data)
  E(279, 4, 1, 8);    // StripByteCounts = 8 bytes (DATASTR)
  PutLE32(t, 0);      // next IFD = 0 (no more IFDs)

  // Add padding up to offset 128
  if (t.size() < 128) t.resize(128, '\0');

  // The actual strip data (8 bytes)
  t += "DATASTR!";  // Example: 8 bytes of data

  return t;
}

std::string MakeTwoStripedTiff() {
  std::string t;

  // ─── Header: II + magic 42 + IFD at byte 8
  t += "II";
  PutLE16(t, 42);    // magic
  PutLE32(t, 8);     // first IFD offset

  // ─── IFD entry count = 6
  PutLE16(t, 6);

  // Helper: write one entry
  auto E = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
    PutLE16(t, tag);
    PutLE16(t, type);
    PutLE32(t, count);
    PutLE32(t, value);
  };

  // 1) ImageWidth=4, 2) ImageLength=8
  E(256, 3, 1, 4);  // SHORT=3
  E(257, 3, 1, 8);  // SHORT=3

  // 3) RowsPerStrip=4 => 2 total strips
  E(278, 3, 1, 4);

  // 4) StripOffsets array => 2 LONG => at offset 128
  E(273, 4, 2, 128);

  // 5) StripByteCounts => 2 LONG => at offset 136
  E(279, 4, 2, 136);

  // 6) Compression => none=1
  E(259, 3, 1, 1);

  // next‑IFD offset = 0
  PutLE32(t, 0);

  // ─── Arrive at offset 128
  if (t.size() < 128) t.resize(128, '\0');

  // two 4‑byte offsets in array => total 8 bytes
  // let’s say strip #0 data at offset=200, strip #1 at offset=208
  PutLE32(t, 200);  // 1st strip offset
  PutLE32(t, 208);  // 2nd strip offset

  // ─── Arrive at offset 136
  if (t.size() < 136) t.resize(136, '\0');

  // two 4‑byte bytecounts => total 8 bytes
  // each strip = 4
  PutLE32(t, 4); // strip #0 size
  PutLE32(t, 4); // strip #1 size

  // ─── Pad to 200, then write "AAAA"
  if (t.size() < 200) t.resize(200, '\0');
  t.replace(200, 4, "AAAA");

  // ─── Pad to 208, then write "BBBB"
  if (t.size() < 208) t.resize(208, '\0');
  t.replace(208, 4, "BBBB");

  return t;
}


/* -------------------------------------------------------------------------- */
/*                            Test‑fixture class                              */
/* -------------------------------------------------------------------------- */

class TiffKeyValueStoreTest : public ::testing::Test {
 public:
  TiffKeyValueStoreTest() : context_(Context::Default()) {}

  // Writes `value` to the in‑memory store at key "data.tif".
  void PrepareMemoryKvstore(absl::Cord value) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        tensorstore::KvStore memory,
        kvstore::Open({{"driver", "memory"}}, context_).result());

    TENSORSTORE_CHECK_OK(
        kvstore::Write(memory, "data.tif", value).result());
  }

  tensorstore::Context context_;
};

/* -------------------------------------------------------------------------- */
/*                                 Tests                                      */
/* -------------------------------------------------------------------------- */

// ─── Tiled TIFF ──────────────────────────────────────────────────────────────
TEST_F(TiffKeyValueStoreTest, Tiled_ReadSuccess) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store,"tile/0/0/0").result());
  EXPECT_EQ(std::string(rr.value), "DATA");
}

TEST_F(TiffKeyValueStoreTest, Tiled_OutOfRange) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  auto status = kvstore::Read(tiff_store,"tile/0/9/9").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kOutOfRange));
}

// ─── Striped TIFF ────────────────────────────────────────────────────────────
TEST_F(TiffKeyValueStoreTest, Striped_ReadOneStrip) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store,"tile/0/0/0").result());
  EXPECT_EQ(std::string(rr.value), "DATASTR!");
}

TEST_F(TiffKeyValueStoreTest, Striped_ReadSecondStrip) {
  PrepareMemoryKvstore(absl::Cord(MakeTwoStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rr, kvstore::Read(tiff_store,"tile/0/1/0").result());
  EXPECT_EQ(std::string(rr.value), "BBBB");
}

TEST_F(TiffKeyValueStoreTest, Striped_OutOfRangeRow) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyStripedTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  auto status = kvstore::Read(tiff_store,"tile/0/2/0").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kOutOfRange));
}

// ─── Bad key format ─────────────────────────────────────────────────────────
TEST_F(TiffKeyValueStoreTest, BadKeyFormat) {
  PrepareMemoryKvstore(absl::Cord(MakeTinyTiledTiff()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto tiff_store,
      kvstore::Open({{"driver","tiff"},
                     {"base",{{"driver","memory"},{"path","data.tif"}}}},
                    context_).result());

  auto status = kvstore::Read(tiff_store,"foo/bar").result().status();
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
