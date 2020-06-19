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

#include <iostream>

#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using tensorstore::Result;
using tensorstore::Status;

namespace {

Status ReturnStatus(int x) {
  if (x < 10) {
    // FIXME: We should adopt equivalent error codes as tensorflow and
    // avoid defining our own errorspace.
    return absl::UnknownError("Unknown error");
  }
  return absl::OkStatus();
}

Status ReturnStatusIfError(int y) {
  TENSORSTORE_RETURN_IF_ERROR(ReturnStatus(y + 1));
  TENSORSTORE_RETURN_IF_ERROR(ReturnStatus(y - 1));
  return ReturnStatus(y);
}

Result<int> ReturnResult(int x) {
  TENSORSTORE_RETURN_IF_ERROR(ReturnStatus(x + 1));
  TENSORSTORE_RETURN_IF_ERROR(ReturnStatus(x - 1));
  TENSORSTORE_RETURN_IF_ERROR(ReturnStatus(x));
  return x / 2;
}

struct MoveOnly {
  MoveOnly(int value) : value(value) {}

  MoveOnly(MoveOnly const&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&) = default;
  MoveOnly& operator=(MoveOnly&&) = default;

  int value;
};

Result<MoveOnly> MakeMoveOnly(int x) {
  if (x % 2 == 1) {
    return x;
  }
  if (x % 5 == 0) {
    MoveOnly y(x);
    return std::move(y);
  }
  return absl::UnknownError("no move");
}

struct CopyOnly {
  CopyOnly(int value) : value(value) {}

  CopyOnly(CopyOnly const&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&) = delete;
  CopyOnly& operator=(CopyOnly&&) = delete;

  int value;
};

Result<CopyOnly> MakeCopyOnly(int x) {
  if (x % 2 == 1) {
    return x;
  }
  if (x % 5 == 1) {
    CopyOnly y(x);
    return y;
  }
  return absl::UnknownError("no copy");
}

Result<std::unique_ptr<CopyOnly>> MakeCopyFactory(int x) {
  if (x % 2 == 1) {
    return Result<std::unique_ptr<CopyOnly>>{new CopyOnly(x)};
  }
  if (x % 5 == 1) {
    return absl::make_unique<CopyOnly>(x);
  }
  return absl::UnknownError("no factory");
}

}  // namespace

int main(int argc, char** argv) {
  // Using Status.
  {
    // Status result values must be used.
    ReturnStatus(3).IgnoreError();

    // FIXME: tensorflow::Status omits operator bool, opting instead for the
    // explicit .ok().  We should consider making our interface equivalent to
    // tensorflow::Status in this way.
    if (!ReturnStatus(15).ok()) {
      std::cout << std::endl << "Status failure";
    }

    auto status = ReturnStatusIfError(10);
    if (!status.ok()) {
      std::cout << std::endl << status;
    }
  }

  // Using Result<T>. Result<T> is similar in many ways to
  // std::expected<T, tensorstore::Status>.
  {
    ReturnResult(20).IgnoreResult();
    auto result = ReturnResult(15);
    if (!result) {
      std::cout << std::endl << result.status();
    } else {
      std::cout << std::endl << result.value();
    }

    result = ReturnResult(result.value());
    if (!result) {
      std::cout << std::endl << result.status();
    } else {
      std::cout << std::endl << result.value();
    }
    std::cout << std::endl;

    for (int i = 9; i < 12; i++) {
      std::cout << i << " ";
      auto x = MakeMoveOnly(i);
      if (x) {
        if (x.value().value != i) {
          std::cout << x.value().value;
        }
      } else {
        std::cout << x.status();
      }
      std::cout << " ";

      auto y = MakeCopyOnly(i);
      if (y) {
        if (y.value().value != i) {
          std::cout << x.value().value;
        }
      } else {
        std::cout << y.status();
      }
      std::cout << " ";

      auto z = MakeCopyFactory(i);
      if (!z) {
        std::cout << z.status();
      }
      std::cout << std::endl;
    }
  }
}
