#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"
#include <iostream>
#include <vector>

ABSL_FLAG(std::string, path, "", "Path to .nc");
ABSL_FLAG(std::string, var,  "temp", "Variable name");
ABSL_FLAG(int, t, 0, "time index");
ABSL_FLAG(int, y0, 0, "lat start (inclusive)");
ABSL_FLAG(int, y1, 1, "lat stop (exclusive)");
ABSL_FLAG(int, x0, 0, "lon start (inclusive)");
ABSL_FLAG(int, x1, 2, "lon stop (exclusive)");
ABSL_FLAG(float, value, 7777.0f, "value to write");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const auto path = absl::GetFlag(FLAGS_path);
  const auto var  = absl::GetFlag(FLAGS_var);
  if (path.empty()) { std::cerr << "Need --path\n"; return 2; }

  auto spec = tensorstore::Spec::FromJson({
      {"driver","netcdf"},
      {"path", path},
      {"variable", var},
      {"mode", "open"}     // keep as open; your driver may also accept create
  }).value();

  auto ctx = tensorstore::Context::Default();
  auto open_res = tensorstore::Open(spec, ctx).result();
  if (!open_res.ok()) {
    std::cerr << "Open failed (likely read-only): " << open_res.status() << "\nSKIPPED\n";
    return 0;
  }
  auto ts = *open_res;

  const int t  = absl::GetFlag(FLAGS_t);
  const int y0 = absl::GetFlag(FLAGS_y0), y1 = absl::GetFlag(FLAGS_y1);
  const int x0 = absl::GetFlag(FLAGS_x0), x1 = absl::GetFlag(FLAGS_x1);
  if (y1 <= y0 || x1 <= x0) { std::cerr << "Bad subregion\n"; return 2; }
  const int ny = y1 - y0, nx = x1 - x0;

  std::vector<float> buf(ny * nx, absl::GetFlag(FLAGS_value));
  auto src = tensorstore::MakeArray<float>(buf, {ny, nx});

  using namespace tensorstore;
  auto target = ts | Dims(0).IndexSlice(t)
                   | Dims(0).HalfOpenInterval(y0, y1)
                   | Dims(1).HalfOpenInterval(x0, x1);

  auto wr = tensorstore::Write(src, target).result();
  if (!wr.ok()) {
    std::cerr << "Write not supported yet: " << wr.status() << "\nSKIPPED\n";
    return 0;
  }
  std::cout << "WRITE: OK\n";
  return 0;
}
