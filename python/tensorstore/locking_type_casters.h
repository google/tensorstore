// Copyright 2025 The TensorStore Authors
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

#ifndef PYTHON_TENSORSTORE_LOCKING_TYPE_CASTERS_H_
#define PYTHON_TENSORSTORE_LOCKING_TYPE_CASTERS_H_

/// \file
///
/// Defines pybind11 type_caster specializations which add critical sections
/// around the cast operations for some types.  This file *must* be included
/// in any place where the following types are used in python:
///
/// - `tensorstore.ArrayStorageStatistics`
/// - `tensorstore.Batch`
/// - `tensorstore.ChunkLayout.Grid`
/// - `tensorstore.ChunkLayout`
/// - `tensorstore.Dim`
/// - `tensorstore.KeyRange`
/// - `tensorstore.OpenMode`, defined in spec.h
/// - `tensorstore.Schema`
/// - `tensorstore.TimestampedStorageGeneration`

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <type_traits>

#include "python/tensorstore/critical_section.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/batch.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/schema.h"

namespace tensorstore {
namespace internal_python {

template <typename T>
struct locking_type_caster : public pybind11::detail::type_caster_base<T> {
  static_assert(!std::is_reference_v<T> && !std::is_pointer_v<T>);

  using Base = pybind11::detail::type_caster_base<T>;

  bool load(pybind11::handle src, bool convert) {
    src_ = src;
    return Base::load(src, convert);
  }

  explicit operator const T*() { return Base::operator T*(); }
  explicit operator const T&() { return Base::operator T&(); }
  explicit operator T*() { return Base::operator T*(); }
  explicit operator T&() { return Base::operator T&(); }

  // When passing by-value add critical sections.
  explicit operator T() {
    ScopedPyCriticalSection cs(src_.ptr());
    return Base::operator T&();
  }

  // cast_op_type determines which operator overload to call for a given c++
  // input parameter type.
  // clang-format off
  template <typename T_>
  using cast_op_type =
      std::conditional_t<
          std::is_same_v<std::remove_reference_t<T_>, const T*>, const T*,
      std::conditional_t<
          std::is_same_v<std::remove_reference_t<T_>, T*>, T*,
      std::conditional_t<
          std::is_same_v<T_, const T&>, const T&,
      std::conditional_t<
          std::is_same_v<T_, T&>, T&,
      T>>>>;  // Fall back to T
  // clang-format on

  pybind11::handle src_;
};

}  // namespace internal_python
}  // namespace tensorstore
namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::ChunkLayout>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::ChunkLayout> {};

template <>
struct type_caster<tensorstore::Schema>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::Schema> {};

template <>
struct type_caster<tensorstore::ArrayStorageStatistics>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::ArrayStorageStatistics> {};

template <>
struct type_caster<tensorstore::KeyRange>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::KeyRange> {};

template <>
struct type_caster<tensorstore::TimestampedStorageGeneration>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::TimestampedStorageGeneration> {};

template <>
struct type_caster<tensorstore::ChunkLayout::Grid>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::ChunkLayout::Grid> {};

template <>
struct type_caster<tensorstore::IndexDomainDimension<>>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::IndexDomainDimension<>> {};

template <>
struct type_caster<tensorstore::Batch>
    : public tensorstore::internal_python::locking_type_caster<
          tensorstore::Batch> {};

}  // namespace detail
}  // namespace pybind11

#endif  // PYTHON_TENSORSTORE_LOCKING_TYPE_CASTERS_H_
