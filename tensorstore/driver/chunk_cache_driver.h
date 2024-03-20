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

#ifndef TENSORSTORE_DRIVER_CHUNK_CACHE_DRIVER_H_
#define TENSORSTORE_DRIVER_CHUNK_CACHE_DRIVER_H_

#include <stddef.h>

#include <cassert>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {

/// TensorStore Driver mixin that implements `Read` and `Write` by forwarding to
/// a `ChunkCache`.
///
/// \tparam Derived Derived `Driver` type, that must define
///     `size_t component_index()`, `const ChunkCache *cache()`, and
///     `const StalenessBound &data_staleness_bound()`.  The `Derived` type can
///     inherit from `ChunkGridSpecificationDriver` to define those methods.
template <typename Derived, typename Parent>
class ChunkCacheReadWriteDriverMixin : public Parent {
 public:
  /// Simply forwards to `ChunkCache::Read`.
  void Read(Driver::ReadRequest request,
            AnyFlowReceiver<absl::Status, ReadChunk, IndexTransform<>> receiver)
      override {
    static_cast<Derived*>(this)->cache()->Read(
        {std::move(request), static_cast<Derived*>(this)->component_index(),
         static_cast<Derived*>(this)->data_staleness_bound().time},
        std::move(receiver));
  }

  /// Simply forwards to `ChunkCache::Write`.
  void Write(Driver::WriteRequest request,
             AnyFlowReceiver<absl::Status, WriteChunk, IndexTransform<>>
                 receiver) override {
    static_cast<Derived*>(this)->cache()->Write(
        {std::move(request), static_cast<Derived*>(this)->component_index()},
        std::move(receiver));
  }
};

/// Specifies the constructor arguments for `ChunkGridSpecificationDriver` (and
/// derived types like `ChunkCacheDriver`).
template <typename CacheType = internal::Cache>
struct ChunkCacheDriverInitializer {
  CachePtr<CacheType> cache;
  size_t component_index;
  StalenessBound data_staleness_bound;
};

/// TensorStore Driver mixin that stores a `CachePtr<ChunkCacheType>`, a
/// `size_t component_index`, and a `StalenessBound data_staleness_bound`.
///
/// This can be combined with `ChunkCacheReadWriteDriverMixin`, as in
/// `ChunkCacheDriver`, to implement `Read` and `Write` operations using the
/// cache.
///
/// This can also be used separately, when using a cache that makes use of a
/// `ChunkGridSpecification` but does not implement the same read/write
/// interface as `ChunkCache`.
///
/// \tparam ChunkCacheType The chunk cache type, must be compatible with
///     `internal::CachePtr` and define a
///     `const internal::ChunkGridSpecification &grid()` method and a
///     `const Executor &executor()` method.
template <typename ChunkCacheType, typename Parent>
class ChunkGridSpecificationDriver : public Parent {
 public:
  using Initializer = ChunkCacheDriverInitializer<ChunkCacheType>;

  template <typename CacheType = ChunkCacheType>
  explicit ChunkGridSpecificationDriver(
      ChunkCacheDriverInitializer<CacheType>&& initializer)
      : cache_(
            static_pointer_cast<ChunkCacheType>(std::move(initializer.cache))),
        component_index_(initializer.component_index),
        data_staleness_bound_(initializer.data_staleness_bound) {
    assert(cache_);
    assert(component_index_ < cache()->grid().components.size());
  }

  // This method is declared as both `virtual` and `final` because it may or may
  // not already be defined as `virtual` by `Parent`.
  //
  // NOLINTNEXTLINE(readability/inheritance)
  virtual ChunkCacheType* cache() const final { return cache_.get(); }

  // NOLINTNEXTLINE(readability/inheritance)
  virtual size_t component_index() const final { return component_index_; }

  Executor data_copy_executor() final { return cache()->executor(); }

  const ChunkGridSpecification::Component& component_spec() const {
    return cache()->grid().components[component_index()];
  }

  DataType dtype() final { return component_spec().dtype(); }

  DimensionIndex rank() final { return component_spec().rank(); }

  // This method is declared as both `virtual` and `final` because it may or may
  // not already be defined as `virtual` by `Parent`.
  //
  // NOLINTNEXTLINE(readability/inheritance)
  virtual const StalenessBound& data_staleness_bound() const final {
    return data_staleness_bound_;
  }

 private:
  CachePtr<ChunkCacheType> cache_;
  size_t component_index_;
  StalenessBound data_staleness_bound_;
};

/// Combines `ChunkGridSpecificationDriver` and
/// `ChunkCacheReadWriteDriverMixin`.
class ChunkCacheDriver
    : public ChunkGridSpecificationDriver<
          ChunkCache,
          ChunkCacheReadWriteDriverMixin<ChunkCacheDriver, Driver>> {
  using Base = ChunkGridSpecificationDriver<
      ChunkCache, ChunkCacheReadWriteDriverMixin<ChunkCacheDriver, Driver>>;

 public:
  using Base::Base;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_CHUNK_CACHE_DRIVER_H_
