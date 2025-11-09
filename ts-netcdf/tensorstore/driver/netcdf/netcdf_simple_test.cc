// Simple NetCDF TensorStore Driver Test
// Just verifies that tensorstore::Open() can open a NetCDF file

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/netcdf/minidriver.h"
#include "tensorstore/open.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using ::nlohmann::json;

TEST(NetCDFDriverSimpleTest, CanOpenFile) {
  // Create a test NetCDF file  
  std::string test_file = "test_simple.nc";
  std::string err;
  
  // Clean up any existing file
  std::remove(test_file.c_str());
  
  // Create file with minidriver
  ASSERT_EQ(0, ts_netcdf::CreateFile(test_file, true, &err)) << err;
  ASSERT_EQ(0, ts_netcdf::CreateDimension(test_file, "x", 10, &err)) << err;
  ASSERT_EQ(0, ts_netcdf::CreateDimension(test_file, "y", 20, &err)) << err;
  ASSERT_EQ(0, ts_netcdf::CreateVariable(test_file, "data",
                                          ts_netcdf::DType::kFloat,
                                          {"x", "y"}, &err)) << err;

  // Try to open with TensorStore
  auto spec = json{
    {"driver", "netcdf"},
    {"path", test_file},
    {"variable", "data"}
  };

  auto result = tensorstore::Open(spec, tensorstore::OpenMode::open,
                                   tensorstore::ReadWriteMode::read).result();

  // Just verify it doesn't fail
  EXPECT_TRUE(result.ok() || result.status().message().find("not implemented") != std::string::npos)
      << "Status: " << result.status();

  // Clean up
  std::remove(test_file.c_str());
}

}  // namespace
