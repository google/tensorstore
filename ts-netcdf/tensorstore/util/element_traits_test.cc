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

#include "tensorstore/util/element_traits.h"

#include <type_traits>

namespace {
using ::tensorstore::AreElementTypesCompatible;
using ::tensorstore::IsElementTypeImplicitlyConvertible;
using ::tensorstore::IsElementTypeOnlyExplicitlyConvertible;

static_assert(IsElementTypeImplicitlyConvertible<int, int>);
static_assert(IsElementTypeImplicitlyConvertible<const int, const int>);
static_assert(IsElementTypeImplicitlyConvertible<int, const int>);
static_assert(IsElementTypeImplicitlyConvertible<const int, const int>);
static_assert(!IsElementTypeImplicitlyConvertible<const int, int>);
static_assert(!IsElementTypeImplicitlyConvertible<int, float>);
static_assert(!IsElementTypeImplicitlyConvertible<const int, const float>);

static_assert(IsElementTypeImplicitlyConvertible<int, void>);
static_assert(IsElementTypeImplicitlyConvertible<int, const void>);
static_assert(IsElementTypeImplicitlyConvertible<const int, const void>);
static_assert(!IsElementTypeImplicitlyConvertible<const int, void>);

static_assert(!IsElementTypeOnlyExplicitlyConvertible<int, int>);
static_assert(!IsElementTypeOnlyExplicitlyConvertible<int, void>);

static_assert(IsElementTypeOnlyExplicitlyConvertible<void, int>);
static_assert(IsElementTypeOnlyExplicitlyConvertible<void, const int>);
static_assert(IsElementTypeOnlyExplicitlyConvertible<const void, const int>);
static_assert(!IsElementTypeOnlyExplicitlyConvertible<const void, int>);

static_assert(AreElementTypesCompatible<int, int>);
static_assert(AreElementTypesCompatible<const int, int>);
static_assert(AreElementTypesCompatible<int, const int>);
static_assert(AreElementTypesCompatible<const int, const int>);
static_assert(AreElementTypesCompatible<const int, void>);
static_assert(AreElementTypesCompatible<const int, const void>);
static_assert(AreElementTypesCompatible<int, void>);
static_assert(AreElementTypesCompatible<int, const void>);
static_assert(AreElementTypesCompatible<void, const int>);
static_assert(AreElementTypesCompatible<const void, const int>);
static_assert(AreElementTypesCompatible<void, int>);
static_assert(AreElementTypesCompatible<const void, const int>);
static_assert(AreElementTypesCompatible<void, int>);
static_assert(AreElementTypesCompatible<const void, int>);
static_assert(!AreElementTypesCompatible<int, float>);
static_assert(!AreElementTypesCompatible<const int, float>);
static_assert(!AreElementTypesCompatible<int, const float>);
static_assert(!AreElementTypesCompatible<const int, const float>);

}  // namespace
