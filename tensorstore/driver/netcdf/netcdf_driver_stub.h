#ifndef TENSORSTORE_DRIVER_NETCDF_NETCDF_DRIVER_STUB_H_
#define TENSORSTORE_DRIVER_NETCDF_NETCDF_DRIVER_STUB_H_
#include <string>

namespace tensorstore {
namespace internal_netcdf {

enum class StatusCode {
  kOk = 0,
  kUnimplemented = 12,
  kInvalidArgument = 3,
};

struct Status {
  StatusCode code;
  std::string message;
  static Status Ok() { return {StatusCode::kOk, ""}; }
  static Status Unimplemented(std::string m) { return {StatusCode::kUnimplemented, std::move(m)}; }
  static Status InvalidArgument(std::string m) { return {StatusCode::kInvalidArgument, std::move(m)}; }
  explicit operator bool() const { return code == StatusCode::kOk; }
};

Status OpenFromJson(const std::string& json);

}  // namespace internal_netcdf
}  // namespace tensorstore
#endif
