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

#ifndef TENSORSTORE_CONTAINER_KIND_H_
#define TENSORSTORE_CONTAINER_KIND_H_

namespace tensorstore {

/// Specifies whether an object (e.g. strided layout) is stored by value or
/// view/reference.
///
/// \ingroup array
enum class ContainerKind {
  /// Object is stored by value.
  container,
  /// Object is stored by reference.
  view
};

/// \relates ContainerKind
constexpr ContainerKind container = ContainerKind::container;

/// \relates ContainerKind
constexpr ContainerKind view = ContainerKind::view;

}  // namespace tensorstore

#endif  // TENSORSTORE_CONTAINER_KIND_H_
