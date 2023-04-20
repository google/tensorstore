// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_REF_COUNTED_STRING_H_
#define TENSORSTORE_INTERNAL_REF_COUNTED_STRING_H_

#include <assert.h>
#include <stddef.h>

#include <atomic>
#include <new>
#include <string_view>
#include <utility>

namespace tensorstore {
namespace internal {

/// Reference-counted immutable string.
///
/// This is the size of a single pointer, and requires only a single heap
/// allocation.
class RefCountedString {
 public:
  RefCountedString() : data_(nullptr) {}
  RefCountedString(std::string_view s) : data_(AllocateCopy(s)) {}
  RefCountedString(const char* s) : RefCountedString(std::string_view(s)) {}

  RefCountedString(const RefCountedString& other) noexcept
      : data_(other.data_) {
    if (data_) {
      header().IncrementReferenceCount();
    }
  }

  RefCountedString(RefCountedString&& other) noexcept : data_(other.data_) {
    other.data_ = nullptr;
  }

  RefCountedString& operator=(const RefCountedString& other) noexcept;

  RefCountedString& operator=(RefCountedString&& other) noexcept {
    auto temp = other.data_;
    other.data_ = data_;
    data_ = temp;
    return *this;
  }

  RefCountedString& operator=(std::string_view s);
  RefCountedString& operator=(const char* s);

  ~RefCountedString() {
    if (!data_) return;
    header().DecrementReferenceCount();
  }

  bool empty() const { return data_ == nullptr; }
  const char* data() const { return data_; }
  size_t size() const { return data_ ? header().length : 0; }

  char operator[](size_t i) const {
    assert(i <= size());
    return data_[i];
  }

  const char* begin() const { return data_; }
  const char* end() const { return data_ + size(); }

  /// Implicitly converts to `string_view`, like `std::string`.
  operator std::string_view() const { return std::string_view(data(), size()); }

  /// Absl string conversion support.
  template <typename Sink>
  friend void AbslStringify(Sink&& sink, const RefCountedString& x) {
    sink.Append(std::string_view(x));
  }

  friend bool operator==(const RefCountedString& a, const RefCountedString& b) {
    return a.data_ == b.data_ || std::string_view(a) == std::string_view(b);
  }

  friend bool operator<(const RefCountedString& a, const RefCountedString& b) {
    return std::string_view(a) < std::string_view(b);
  }

  friend bool operator<=(const RefCountedString& a, const RefCountedString& b) {
    return std::string_view(a) <= std::string_view(b);
  }

  friend bool operator>(const RefCountedString& a, const RefCountedString& b) {
    return std::string_view(a) > std::string_view(b);
  }

  friend bool operator>=(const RefCountedString& a, const RefCountedString& b) {
    return std::string_view(a) >= std::string_view(b);
  }

  friend bool operator!=(const RefCountedString& a, const RefCountedString& b) {
    return !(a == b);
  }

  friend bool operator==(std::string_view a, const RefCountedString& b) {
    return a == std::string_view(b);
  }

  friend bool operator<(std::string_view a, const RefCountedString& b) {
    return a < std::string_view(b);
  }

  friend bool operator<=(std::string_view a, const RefCountedString& b) {
    return a <= std::string_view(b);
  }

  friend bool operator>(std::string_view a, const RefCountedString& b) {
    return a > std::string_view(b);
  }

  friend bool operator>=(std::string_view a, const RefCountedString& b) {
    return a >= std::string_view(b);
  }

  friend bool operator!=(std::string_view a, const RefCountedString& b) {
    return a != std::string_view(b);
  }

  friend bool operator==(const char* a, const RefCountedString& b) {
    return a == std::string_view(b);
  }

  friend bool operator<(const char* a, const RefCountedString& b) {
    return a < std::string_view(b);
  }

  friend bool operator<=(const char* a, const RefCountedString& b) {
    return a <= std::string_view(b);
  }

  friend bool operator>(const char* a, const RefCountedString& b) {
    return a > std::string_view(b);
  }

  friend bool operator>=(const char* a, const RefCountedString& b) {
    return a >= std::string_view(b);
  }

  friend bool operator!=(const char* a, const RefCountedString& b) {
    return a != std::string_view(b);
  }

  friend bool operator==(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) == b;
  }

  friend bool operator<(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) < b;
  }

  friend bool operator<=(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) <= b;
  }

  friend bool operator>(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) > b;
  }

  friend bool operator>=(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) >= b;
  }

  friend bool operator!=(const RefCountedString& a, std::string_view b) {
    return std::string_view(a) != b;
  }

  friend bool operator==(const RefCountedString& a, const char* b) {
    return std::string_view(a) == b;
  }

  friend bool operator<(const RefCountedString& a, const char* b) {
    return std::string_view(a) < b;
  }

  friend bool operator<=(const RefCountedString& a, const char* b) {
    return std::string_view(a) <= b;
  }

  friend bool operator>(const RefCountedString& a, const char* b) {
    return std::string_view(a) > b;
  }

  friend bool operator>=(const RefCountedString& a, const char* b) {
    return std::string_view(a) >= b;
  }

  friend bool operator!=(const RefCountedString& a, const char* b) {
    return std::string_view(a) != b;
  }

  template <typename H>
  friend H AbslHashValue(H h, const RefCountedString& s) {
    return H::combine_contiguous(std::move(h), s.data_, s.size());
  }

 private:
  friend class RefCountedStringWriter;

  struct Header {
    size_t length;
    mutable std::atomic<size_t> ref_count{1};

    void IncrementReferenceCount() const {
      ref_count.fetch_add(1, std::memory_order_relaxed);
    }

    void DecrementReferenceCount() const {
      if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        Deallocate();
      }
    }

    void Deallocate() const;
  };

  static char* Allocate(size_t size);
  static const char* AllocateCopy(std::string_view s);

  const Header& header() const {
    return reinterpret_cast<const Header*>(data_)[-1];
  }

  const char* data_;
};

class RefCountedStringWriter {
 public:
  RefCountedStringWriter() = default;
  explicit RefCountedStringWriter(size_t size) {
    string_.data_ = RefCountedString::Allocate(size);
  }
  RefCountedStringWriter(RefCountedStringWriter&& other) = default;
  RefCountedStringWriter(const RefCountedStringWriter& other) = delete;

  char* data() { return const_cast<char*>(string_.data()); }
  size_t size() const { return string_.size(); }

  operator RefCountedString() && { return std::move(string_); }

 private:
  RefCountedString string_;
};

template <typename T, typename SFINAE>
struct HeapUsageEstimator;

template <>
struct HeapUsageEstimator<RefCountedString, void> {
  static size_t EstimateHeapUsage(const RefCountedString& x, size_t max_depth) {
    return x.size();
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_REF_COUNTED_STRING_H_
