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

#ifndef TENSORSTORE_INTERNAL_OS_GET_BIOS_INFO_H_
#define TENSORSTORE_INTERNAL_OS_GET_BIOS_INFO_H_

#include <string>

#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_os {

/// Returns the BIOS info if it is a GCP machines.
Result<std::string> GetGcpProductName();

}  // namespace internal_os
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_OS_GET_BIOS_INFO_H_
