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

#include "tensorstore/util/element_pointer.h"

#include <string>

#include "tensorstore/data_type.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_element_pointer {

std::string DescribeForCast(DataType dtype) {
  return tensorstore::StrCat("pointer with ",
                             StaticCastTraits<DataType>::Describe(dtype));
}

}  // namespace internal_element_pointer
}  // namespace tensorstore
