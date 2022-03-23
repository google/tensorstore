#ifndef TENSORSTORE_EXAMPLES_DATA_TYPE_INVOKE_H_
#define TENSORSTORE_EXAMPLES_DATA_TYPE_INVOKE_H_

#include "absl/status/status.h"
#include <half.hpp>
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore_examples {
namespace internal {

// Calls F with comparable data types.
template <typename Fn>
struct DataTypeInvokerFn {
 public:
  Fn fn;

  template <typename... Args>
  inline absl::Status operator()(tensorstore::DataTypeId id, Args&&... args) {
#define INVOKE_WITH_TYPE(T, ...)                               \
  case tensorstore::DataTypeId::T: {                           \
    return tensorstore::internal::InvokeForStatus(             \
        fn, tensorstore::T{}, std::forward<Args...>(args)...); \
  }

    switch (id) {
      TENSORSTORE_FOR_EACH_DATA_TYPE(INVOKE_WITH_TYPE)
      default:
        break;
    }
#undef INVOKE_WITH_TYPE
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Could not invoke with data type ", id));
  }
};

template <template <class...> class Trait, class AlwaysVoid, class... Args>
struct detector : std::false_type {};
template <template <class...> class Trait, class... Args>
struct detector<Trait, std::void_t<Trait<Args...>>, Args...> : std::true_type {
};

template <class T>
using invoke_dtype_t = decltype(std::declval<T*>()->dtype());

template <typename T>
using HasDTypeMethod = typename detector<invoke_dtype_t, void, T>::type;

}  // namespace internal

// Returns the DataType.id of a type.
template <typename T>
std::enable_if_t<std::is_same_v<T, tensorstore::DataTypeId>,
                 ::tensorstore::DataTypeId>
DataTypeIdOf(T id) {
  return id;
}

template <typename T>
std::enable_if_t<std::is_same_v<T, tensorstore::DataType>,
                 ::tensorstore::DataTypeId>
DataTypeIdOf(const T& t) {
  return t.id();
}

template <typename T>
std::enable_if_t<internal::HasDTypeMethod<T>::value, ::tensorstore::DataTypeId>
DataTypeIdOf(const T& t) {
  return t.dtype().id();
}

/// MakeDataTypeInvoker returns a function object which wraps a function
/// where the first paramter is the array datatype.
///
/// Example:
///   auto fn = MakeDataTypeInvoker([](auto t, const auto& a) {
///     using ArrayDataType = decltype(t);
///     T* data = static_cast<T*>(a.data());
///   });
///
template <typename Fn>
inline auto MakeDataTypeInvoker(Fn fn) {
  return internal::DataTypeInvokerFn<Fn>{std::move(fn)};
}

}  // namespace tensorstore_examples

#endif  // TENSORSTORE_EXAMPLES_DATA_TYPE_INVOKE_H_
