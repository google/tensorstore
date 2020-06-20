
#include "tensorstore/internal/init_tensorstore.h"

#include "absl/flags/parse.h"

namespace tensorstore {

void InitTensorstore(int* argc, char*** argv) {
  // Parse flags via absl::ParseCommandLine.
  absl::ParseCommandLine(*argc, *argv);
}

}  // namespace tensorstore
