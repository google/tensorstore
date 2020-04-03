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

#ifndef TENSORSTORE_INTERNAL_UNIQUE_WITH_INTRUSIVE_ALLOCATOR_H_
#define TENSORSTORE_INTERNAL_UNIQUE_WITH_INTRUSIVE_ALLOCATOR_H_

#include <memory>
#include <new>
#include <utility>

namespace tensorstore {
namespace internal {

/// Deleter for use with `std::unique_ptr` or `std::shared_ptr` that destroys
/// and deallocates an object using the allocator returned by its
/// `get_allocator()` method.
///
/// This avoids the need to separately store the allocator when the object needs
/// a copy of it anyway.
///
/// Note that this deleter is not suitable on its own for deleting a derived
/// class via a base class pointer, because `deallocate` is called with
/// `sizeof(T)`.  For that purpose, use `VirtualDestroyDeleter` in conjunction
/// with `IntrusiveAllocatorBase`.
template <typename T>
struct IntrusiveAllocatorDeleter {
  void operator()(T* p) {
    auto allocator = p->get_allocator();
    typename std::allocator_traits<decltype(
        allocator)>::template rebind_alloc<T>
        rebound_allocator(std::move(allocator));
    std::allocator_traits<decltype(rebound_allocator)>::destroy(
        rebound_allocator, p);
    std::allocator_traits<decltype(rebound_allocator)>::deallocate(
        rebound_allocator, p, 1);
  }
};

/// Variant of `std::make_unique` that uses an allocator to be embedded in the
/// constructed object.
///
/// Allocates an object of type `T` using the specified `allocator`, and
/// constructs it as `T(std::forward<Arg>(arg)..., allocator)`.  The type `T`
/// must define a `get_allocator()` method that returns a copy of the allocator
/// passed to the constructor; this method is used by
/// `IntrusiveAllocatorDeleter` to destroy the allocated object.
///
/// \tparam T Type of object to construct.
/// \param allocator Allocator to use.
/// \param arg... Constructor arguments.
template <typename T, typename Allocator, typename... Arg>
std::unique_ptr<T, IntrusiveAllocatorDeleter<T>>
MakeUniqueWithIntrusiveAllocator(Allocator allocator, Arg&&... arg) {
  using ReboundAllocator =
      typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
  ReboundAllocator rebound_allocator(std::move(allocator));
  auto temp_deleter = [&rebound_allocator](T* p) {
    std::allocator_traits<ReboundAllocator>::deallocate(rebound_allocator, p,
                                                        1);
  };
  std::unique_ptr<T, decltype(temp_deleter)> temp_ptr(
      std::allocator_traits<ReboundAllocator>::allocate(rebound_allocator, 1),
      temp_deleter);
  new (temp_ptr.get())
      T(std::forward<Arg>(arg)..., std::move(rebound_allocator));
  return std::unique_ptr<T, IntrusiveAllocatorDeleter<T>>(temp_ptr.release());
}

/// Deleter that calls `p->Destroy()`, intended to be used in conjunction with
/// `MakeUniqueWithVirtualIntrusiveAllocator`.
struct VirtualDestroyDeleter {
  template <typename T>
  void operator()(T* p) const {
    p->Destroy();
  }
};

/// CRTP base class that provides a `Destroy` method that calls
/// `IntrusiveAllocatorDeleter`.  This is intended to be used to destroy objects
/// allocated by `MakeUniqueWithVirtualIntrusiveAllocator`.
///
/// Example usage:
///
///     class Base {
///      public:
///       // Other methods.
///       virtual void Destroy() = 0;
///     };
///
///     class Derived : public IntrusiveAllocatorBase<Base> {
///      public:
///       using allocator_type = ...;
///
///       Derived(allocator_type allocator)
///         : allocator_(allocator) {}
///
///       allocator_type get_allocator() const { return allocator_; }
///
///       // Other definitions
///
///      private:
///       allocator_type allocator_;
///     };
///
///     std::unique_ptr<Base, VirtualDestroyDeleter> ptr =
///         MakeUniqueWithVirtualIntrusiveAllocator<Derived>(allocator);
///
/// \tparam Derived Derived class type that inherits from this class, must equal
///     be the actual run-time type (not a base class) for correct destruction
///     behavior.  Must define a `get_allocator()` method that returns the
///     allocator to use to destroy the object.
/// \tparam IntrusiveBase Base class to inherit from.  Must define a
///     `virtual void Destroy()` method.
template <typename Derived, typename IntrusiveBase>
class IntrusiveAllocatorBase : public IntrusiveBase {
 public:
  using IntrusiveBase::IntrusiveBase;
  void Destroy() override {
    IntrusiveAllocatorDeleter<Derived>()(static_cast<Derived*>(this));
  }
};

/// Variant of `std::make_unique` that uses an allocator to be embedded in the
/// constructed object.
///
/// Allocates an object of type `T` using the specified `allocator`, and
/// constructs it as `T(std::forward<Arg>(arg)..., allocator)`.  The type `T`
/// must define a `get_allocator()` method that returns a copy of the allocator
/// passed to the constructor; this method is used by
/// `IntrusiveAllocatorDeleter` to destroy the allocated object.
///
/// \tparam T Type of object to construct.  Must inherit from
///     `IntrusiveAllocatorBase<T, Base>` for some type `Base`.
/// \param allocator Allocator to use.
/// \param arg... Constructor arguments.
template <typename T, typename Allocator, typename... Arg>
std::unique_ptr<T, VirtualDestroyDeleter>
MakeUniqueWithVirtualIntrusiveAllocator(Allocator allocator, Arg&&... arg) {
  return std::unique_ptr<T, VirtualDestroyDeleter>(
      MakeUniqueWithIntrusiveAllocator<T>(std::move(allocator),
                                          std::forward<Arg>(arg)...)
          .release());
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_UNIQUE_WITH_INTRUSIVE_ALLOCATOR_H_
