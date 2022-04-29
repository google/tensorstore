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

#ifndef TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_METHOD_H_
#define TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_METHOD_H_

#include <iosfwd>

namespace tensorstore {

/// Specifies the method by which the index into a given output dimension of an
/// index transform is computed from the input indices.
///
/// \relates OutputIndexMapRef
enum class OutputIndexMethod {
  /// Output index is equal to the constant ``offset`` and does not depend on
  /// any input dimensions.
  constant,
  /// Output index is equal to ``offset + stride * input[input_dimension]``.
  single_input_dimension,
  /// Output index is equal to ``offset + stride * array(input)``.
  array
};

/// Prints to an output stream.
///
/// \relates OutputIndexMethod
/// \id OutputIndexMethod
std::ostream& operator<<(std::ostream& os, OutputIndexMethod method);

}  // namespace tensorstore

#endif  // TENSORSTORE_INDEX_SPACE_OUTPUT_INDEX_METHOD_H_
