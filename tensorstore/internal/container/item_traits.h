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

#ifndef TENSORSTORE_INTERNAL_CONTAINER_ITEM_TRAITS_H_
#define TENSORSTORE_INTERNAL_CONTAINER_ITEM_TRAITS_H_

#include <cstring>
#include <memory>
#include <type_traits>

#include "absl/meta/type_traits.h"

namespace tensorstore {
namespace internal_container {

// Defines how items are initialized/destroyed/transferred.
// Based on absl/container/internal/common_policy_traits.h
template <typename T>
struct ItemTraits {
  template <typename Allocator, class... Args>
  static void construct(Allocator* alloc, T* storage, Args&&... args) {
    using AllocatorTraits = std::allocator_traits<Allocator>;
    AllocatorTraits::construct(*alloc, storage, std::forward<Args>(args)...);
  }

  template <typename Allocator>
  static auto destroy(Allocator* alloc, T* storage) {
    using AllocatorTraits = std::allocator_traits<Allocator>;
    AllocatorTraits::destroy(*alloc, storage);
    return IsDestructionTrivial<Allocator>();
  }

  template <typename Allocator>
  static auto transfer(Allocator* alloc, T* new_storage, T* old_storage) {
    return transfer_impl(alloc, new_storage, old_storage, Rank1{});
  }

  static constexpr bool transfer_uses_memcpy() {
    return std::is_same<decltype(transfer_impl<std::allocator<char>>(
                            nullptr, nullptr, nullptr, Rank1{})),
                        std::true_type>::value;
  }

  // Returns true if destroy is trivial and can be omitted.
  template <class Alloc>
  static constexpr bool destroy_is_trivial() {
    return std::is_same<decltype(destroy<Alloc>(nullptr, nullptr)),
                        std::true_type>::value;
  }

 private:
  struct Rank0 {};
  struct Rank1 : Rank0 {};

  // This overload returns true_type for the trait below.
  // The conditional_t is to make the enabler type dependent.
  template <class Alloc,
            typename = std::enable_if_t<absl::is_trivially_relocatable<
                std::conditional_t<false, Alloc, T>>::value>>
  static std::true_type transfer_impl(Alloc*, T* new_storage, T* old_storage,
                                      Rank1) {
    std::memcpy(static_cast<void*>(std::launder(new_storage)),
                static_cast<const void*>(old_storage), sizeof(T));
    return {};
  }

  template <class Alloc>
  static std::false_type transfer_impl(Alloc* alloc, T* new_storage,
                                       T* old_storage, Rank0) {
    construct(alloc, new_storage, std::move(*old_storage));
    destroy(alloc, old_storage);
    return {};
  }

  // Returns true if the destruction of the value with given Allocator will be
  // trivial.
  template <class Allocator>
  static constexpr auto IsDestructionTrivial() {
    constexpr bool result =
        std::is_trivially_destructible<T>::value &&
        std::is_same<typename std::allocator_traits<
                         Allocator>::template rebind_alloc<char>,
                     std::allocator<char>>::value;
    return std::integral_constant<bool, result>();
  }
};

}  // namespace internal_container
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CONTAINER_ITEM_TRAITS_H_
