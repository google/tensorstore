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

#ifndef TENSORSTORE_INTERNAL_TESTING_DYNAMIC_H_
#define TENSORSTORE_INTERNAL_TESTING_DYNAMIC_H_

#include <functional>
#include <string>

#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal_testing {

/// Registers a GoogleTest case dynamically.
void RegisterGoogleTestCaseDynamically(
    std::string test_suite_name, std::string test_name,
    std::function<void()> test_func,
    SourceLocation loc = tensorstore::SourceLocation::current());

}  // namespace internal_testing
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_TESTING_DYNAMIC_H_
