// Copyright 2021 The TensorStore Authors
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

#ifndef TENSORSTORE_UTIL_GARBAGE_COLLECTION_FWD_H_
#define TENSORSTORE_UTIL_GARBAGE_COLLECTION_FWD_H_

/// \file
///
/// Forward declarations for garbage collection support.
///
/// This header is sufficient for declaring `GarbageCollection` specializations.

namespace tensorstore {
namespace garbage_collection {

/// Defines garbage collection support for a given type `T`.
///
/// The extra `SFINAE` parameter must always be void and may be used to define
/// partial specializations.
///
/// Specializations must either:
///
/// - Define a `constexpr static bool required() { return false; }` method to
///   indicate that garbage collection is not required.  If this is defined, the
///   `Visit` method need not be defined, and won't be called.
///
/// - Define a
///   `static void Visit(GarbageCollectionVisitor& visitor, const T& value)`
///   method that invokes `visitor` for each directly reachable object that may
///   reference garbage-collected objects.  It is not necessary to define a
///   `constexpr static bool required() { return true; }` method in this case,
///   as that is the default.
template <typename T, typename SFINAE = void>
struct GarbageCollection;

class GarbageCollectionVisitor;

}  // namespace garbage_collection
}  // namespace tensorstore

/// Defines a specialization of
/// `tensorstore::garbage_collection::GarbageCollection<TYPE>`, but does not
/// define the `Visit` function.
///
/// This is intended for use in header files at global scope (outside of any
/// namespaces).  In the corresponding source file, you should either define the
/// `Visit` function manually, or use
/// `TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION` to define it to
/// forward to an existing `GarbageCollection` implementation.  That allows you
/// to avoid a dependency from your header file on `garbage_collection.h`.
///
/// For types that do not require garbage collection, you should use
/// `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED` instead.
#define TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(TYPE)         \
  template <>                                                               \
  struct tensorstore::garbage_collection::GarbageCollection<TYPE> {         \
    static void Visit(                                                      \
        tensorstore::garbage_collection::GarbageCollectionVisitor& visitor, \
        const TYPE& value);                                                 \
  };                                                                        \
  /**/

/// Defines the `Visit` function of
/// `tensorstore::garbage_collection::GarbageCollection<TYPE>` to forward to
/// `TRAITS::Visit`.
///
/// This is intended to be used in a source file, to define the `Visit` function
/// declared by `TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION` in the
/// corresponding header file.
///
/// Possible `TRAITS` types include
/// `garbage_collection::ApplyMembersGarbageCollection<TYPE>`,
/// `garbage_collection::ContainerGarbageCollection<TYPE>`,
/// `garbage_collection::OptionalGarbageCollection<TYPE>`,
/// `garbage_collection::IndirectPointerGarbageCollection<TYPE>`,
/// `internal_kvs_backed_chunk_driver::DriverBase::GarbageCollectionBase`.
#define TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(TYPE, TRAITS) \
  void tensorstore::garbage_collection::GarbageCollection<TYPE>::Visit(    \
      tensorstore::garbage_collection::GarbageCollectionVisitor& visitor,  \
      const TYPE& value) {                                                 \
    return TRAITS::Visit(visitor, value);                                  \
  }                                                                        \
  /**/

/// Define a specialization of
/// `tensorstore::garbage_collection::GarbageCollection<TYPE>` that indicates
/// garbage collection is not required.
#define TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(TYPE)   \
  template <>                                                       \
  struct tensorstore::garbage_collection::GarbageCollection<TYPE> { \
    static constexpr bool required() { return false; }              \
  };                                                                \
  /**/

#endif  // TENSORSTORE_UTIL_GARBAGE_COLLECTION_FWD_H_
