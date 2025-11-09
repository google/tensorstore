// NetCDF TensorStore Driver Integration Test
// Tests full end-to-end integration through TensorStore C++ API

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/netcdf/netcdf_driver_full.h"
#include "tensorstore/driver/netcdf/minidriver.h"
#include "tensorstore/open.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/data_type.h"
#include "tensorstore/static_cast.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using ::tensorstore::Context;
using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::nlohmann::json;

class NetCDFDriverIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a test NetCDF file using minidriver
    std::string err;
    test_file_ = "test_integration.nc";

    // Clean up any existing file
    std::remove(test_file_.c_str());

    // Create file with minidriver
    ASSERT_EQ(0, ts_netcdf::CreateFile(test_file_, true, &err)) << err;
    ASSERT_EQ(0, ts_netcdf::CreateDimension(test_file_, "x", 10, &err)) << err;
    ASSERT_EQ(0, ts_netcdf::CreateDimension(test_file_, "y", 20, &err)) << err;
    ASSERT_EQ(0, ts_netcdf::CreateVariable(test_file_, "data",
                                            ts_netcdf::DType::kFloat,
                                            {"x", "y"}, &err)) << err;

    // Write some initial data
    std::vector<float> init_data(10 * 20, 1.0f);
    ts_netcdf::Slice slice{{0, 0}, {10, 20}, {}};
    ASSERT_EQ(0, ts_netcdf::WriteFloats(test_file_, "data", slice,
                                         init_data.data(), &err)) << err;
  }

  void TearDown() override {
    std::remove(test_file_.c_str());
  }

  std::string test_file_;
};

// Test 1: Open existing NetCDF file through TensorStore
TEST_F(NetCDFDriverIntegrationTest, OpenExistingFile) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file_},
    {"variable", "data"}
  };

  auto context = Context::Default();
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(spec, context, tensorstore::OpenMode::open,
                       tensorstore::ReadWriteMode::read_write)
          .result());

  // Verify shape
  EXPECT_EQ(2, store.rank());
  EXPECT_EQ(10, store.domain().shape()[0]);
  EXPECT_EQ(20, store.domain().shape()[1]);
}

// Test 2: Read data through TensorStore API
TEST_F(NetCDFDriverIntegrationTest, ReadData) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file_},
    {"variable", "data"}
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(spec, tensorstore::OpenMode::open,
                       tensorstore::ReadWriteMode::read)
          .result());

  // Read all data
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto data,
      tensorstore::Read(store).result());

  // Verify data
  ASSERT_EQ(2, data.rank());
  EXPECT_EQ(10, data.shape()[0]);
  EXPECT_EQ(20, data.shape()[1]);

  // Cast to float array for element access
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto float_data,
      tensorstore::StaticDataTypeCast<float>(data));

  // Check values (should all be 1.0 from setup)
  for (Index i = 0; i < 10; ++i) {
    for (Index j = 0; j < 20; ++j) {
      EXPECT_FLOAT_EQ(1.0f, float_data({i, j}));
    }
  }
}

// Test 3: Write data through TensorStore API
TEST_F(NetCDFDriverIntegrationTest, WriteData) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file_},
    {"variable", "data"}
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(spec, tensorstore::OpenMode::open,
                       tensorstore::ReadWriteMode::read_write)
          .result());

  // Create test data
  auto write_data = MakeArray<float>({{2.0, 3.0, 4.0},
                                       {5.0, 6.0, 7.0}});

  // Write to a subregion [0:2, 0:3]
  auto write_result = tensorstore::Write(
      write_data,
      store | tensorstore::Dims(0).SizedInterval(0, 2) |
              tensorstore::Dims(1).SizedInterval(0, 3)
  );
  TENSORSTORE_ASSERT_OK(write_result.commit_future.result());

  // Read back and verify
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_back,
      tensorstore::Read(
          store | tensorstore::Dims(0).SizedInterval(0, 2) |
                  tensorstore::Dims(1).SizedInterval(0, 3)
      ).result());

  // Cast to float for element access
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto float_read_back,
      tensorstore::StaticDataTypeCast<float>(read_back));

  // Verify written data
  EXPECT_FLOAT_EQ(2.0f, float_read_back({0, 0}));
  EXPECT_FLOAT_EQ(3.0f, float_read_back({0, 1}));
  EXPECT_FLOAT_EQ(4.0f, float_read_back({0, 2}));
  EXPECT_FLOAT_EQ(5.0f, float_read_back({1, 0}));
  EXPECT_FLOAT_EQ(6.0f, float_read_back({1, 1}));
  EXPECT_FLOAT_EQ(7.0f, float_read_back({1, 2}));

  // Verify rest of data unchanged
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rest,
      tensorstore::Read(
          store | tensorstore::Dims(0).SizedInterval(2, 1) |
                  tensorstore::Dims(1).SizedInterval(0, 3)
      ).result());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto float_rest,
      tensorstore::StaticDataTypeCast<float>(rest));

  for (Index j = 0; j < 3; ++j) {
    EXPECT_FLOAT_EQ(1.0f, float_rest({0, j})) << "Position (2, " << j << ")";
  }
}

// Test 4: Write with stride
TEST_F(NetCDFDriverIntegrationTest, WriteWithStride) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file_},
    {"variable", "data"}
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(spec, tensorstore::OpenMode::open,
                       tensorstore::ReadWriteMode::read_write)
          .result());

  // Write to every other position
  auto write_data = MakeArray<float>({10.0, 20.0, 30.0, 40.0, 50.0});

  auto write_result = tensorstore::Write(
      write_data,
      store | tensorstore::Dims(0).SizedInterval(0, 5, 2) |
              tensorstore::Dims(1).SizedInterval(0, 1)
  );
  TENSORSTORE_ASSERT_OK(write_result.commit_future.result());

  // Read back with same stride
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto read_back,
      tensorstore::Read(
          store | tensorstore::Dims(0).SizedInterval(0, 5, 2) |
                  tensorstore::Dims(1).SizedInterval(0, 1)
      ).result());

  // Cast to float for element access
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto float_read_back,
      tensorstore::StaticDataTypeCast<float>(read_back));

  // Verify
  EXPECT_EQ(5, read_back.shape()[0]);
  EXPECT_FLOAT_EQ(10.0f, float_read_back({0, 0}));
  EXPECT_FLOAT_EQ(20.0f, float_read_back({1, 0}));
  EXPECT_FLOAT_EQ(30.0f, float_read_back({2, 0}));
  EXPECT_FLOAT_EQ(40.0f, float_read_back({3, 0}));
  EXPECT_FLOAT_EQ(50.0f, float_read_back({4, 0}));
}

// Test 5: Create new file through TensorStore
TEST_F(NetCDFDriverIntegrationTest, CreateNewFile) {
  std::string new_file = "test_create_new.nc";
  std::remove(new_file.c_str());

  auto spec = json{
    {"driver", "netcdf"},
    {"path", new_file},
    {"variable", "temperature"},
    {"mode", "w"},
    {"dimensions", json::array({
      json::object({{"time", 100}}),
      json::object({{"lat", 50}}),
      json::object({{"lon", 100}})
    })},
    {"dtype", "float32"},
    {"shape", json::array({100, 50, 100})}
  };

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto store,
      tensorstore::Open(spec, tensorstore::OpenMode::create)
          .result());

  // Write some data
  auto data = MakeArray<float>({{{1.0, 2.0}, {3.0, 4.0}}});

  auto write_result = tensorstore::Write(
      data,
      store | tensorstore::Dims(0).SizedInterval(0, 1) |
              tensorstore::Dims(1).SizedInterval(0, 2) |
              tensorstore::Dims(2).SizedInterval(0, 2)
  );
  TENSORSTORE_ASSERT_OK(write_result.commit_future.result());

  // Verify with minidriver
  std::string err;
  ts_netcdf::Info info;
  ASSERT_EQ(0, ts_netcdf::Inspect(new_file, "temperature", &info, &err)) << err;
  EXPECT_EQ(ts_netcdf::DType::kFloat, info.dtype);
  EXPECT_EQ(3, info.shape.size());
  EXPECT_EQ(100, info.shape[0]);
  EXPECT_EQ(50, info.shape[1]);
  EXPECT_EQ(100, info.shape[2]);

  std::remove(new_file.c_str());
}

// Test 6: Error handling - invalid file
TEST_F(NetCDFDriverIntegrationTest, ErrorInvalidFile) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", "nonexistent.nc"},
    {"variable", "data"}
  };

  auto result = tensorstore::Open(spec, tensorstore::OpenMode::open).result();
  EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kNotFound));
}

// Test 7: Error handling - invalid variable
TEST_F(NetCDFDriverIntegrationTest, ErrorInvalidVariable) {
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file_},
    {"variable", "nonexistent"}
  };

  auto result = tensorstore::Open(spec, tensorstore::OpenMode::open).result();
  EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kNotFound));
}

}  // namespace
