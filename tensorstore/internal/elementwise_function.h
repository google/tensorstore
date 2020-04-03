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

#ifndef TENSORSTORE_UTIL_ELEMENTWISE_FUNCTION_H_
#define TENSORSTORE_UTIL_ELEMENTWISE_FUNCTION_H_

/// \file
/// Type-erasure facility for `N`-ary element-wise functions to be invoked on
/// the corresponding elements from `N` arrays, similar to NumPy "universal
/// functions".
///
/// To avoid the overhead of an indirect function call for each element, to
/// type-erase a given element-wise function, we generate several vectorized
/// function variants.  Logically, each vectorized function variant is called
/// with `N` one-dimensional arrays (called "iteration buffers"), and invokes
/// the elementwise function with pointers to the corresponding elements of each
/// of the arrays.  A variant is generated for each of the following cases:
///
///   1. A variant used when all arguments are contiguous arrays
///      (`IterationBufferKind::kContiguous`).
///
///   2. A variant used when all arguments have a fixed stride (specified at
///      run-time) between consecutive elements (corresponds to
///      `IterationBufferKind::kStrided`).  (This variant is used if at least
///      one array argument is not contiguous.)
///
///   3. A variant used when offset of each element is determined by a separate
///      offset array for each argument (corresponds to
///      `IterationBufferKind::kIndexed`).  (This variant is used if at least
///      one array argument requires the indirection of an offset array.)
///
/// Each of the generated function variants returns a count of the number of
/// elements processed successfully.  The function indicates an error by
/// returning a count less than the element count provided to it.
///
/// This facility is used by other library components that provide mechanisms
/// for invoking element-wise functions over multi-dimensional arrays.  The
/// typical usage is as follows:
///
/// A library component defines a type-erased function in the `internal`
/// namespace that accepts `N` type-erased arrays in some form and an
/// `ElementwiseClosure<N>, and returns a `bool` value, e.g.:
///
///     template <std::size_t N>
///     bool SomeIterateFunction(const std::array<ArrayView<void>,N> &arrays,
///                              ElementwiseClosure<N> closure);
///
/// This function should call one of `closure.function->functions` for each
/// one-dimensional sub-array, passing in `closure.context` as the first
/// argument, and should return `false` as soon as a callback function returns a
/// number not equal to count passed to it.
///
/// The library component also defines a templated wrapper function intended for
/// public consumption that accepts `N` typed arrays in some form as well as an
/// arbitrary function object by universal reference, e.g.:
///
///     template <typename Func, typename... ArrayType,
///               typename ResultType =
///                 std::invoke_result_t<Func&, typename ArrayType::Element...>>
///     bool SomeIterateFunction(Func &&func, const ArrayType& array...) {
///       return internal::SomeIterateFunction(
///           {{EraseTypeAndConst(array)...}},
///           internal::SimpleElementwiseFunction<
///               Func(typename ArrayType::Element...)>::Closure(&func));
///     }
///
/// The type-erased function can be implemented outside of a header, and
/// explicitly instantiated for a fixed range of arities.  Only the small
/// wrapper function, which generates the vectorized versions of the
/// user-specified element-wise function, needs to be instantiated for each use.
/// Note that both the type and const qualification of the elements are erased.
///
/// The internal type-erased function, in addition to being used by the typed
/// public interface, can also be used directly by other type-erased functions
/// that have already been passed an `ElementwiseClosure`.
///
/// `LoopTemplate<Arity, ExtraArg...>` concept:
/// -----------------------------------
///
/// A type that models the `LoopTemplate<Arity, ExtraArg...>` concept has the
/// following members:
///
///     using ElementwiseFunctionType = ElementwiseFunction<Arity, ExtraArg...>;
///
///     template <typename ArrayAccessor>
///     static Index Loop(void *context, Index count,
///                       IterationBufferPointer pointer...,
///                       ExtraArg... extra_arg);
///
///  This function invokes an element-wise function for each of the `count`
///  elements, using the specified base pointers.  The number of `pointer`
///  parameters must be equal to `Arity`.  This function returns the number of
///  elements successfully processed.  Returning `count` indicates that all
///  elements were processed successfully and iteration should continue.
///  Returning a number less than `count` indicates that an error occurred and
///  that iteration should stop.
///
///  The `extra_arg...` parameters have no pre-specified purpose and are simply
///  passed through.
///
/// Wrapping/adapting an existing ElementwiseFunction:
/// --------------------------------------------------
///
/// To wrap or adapt an existing ElementwiseFunction with the overhead of only
/// one extra function call per 1-d block (rather than an extra function call
/// per element) is to define a struct as follows:
///
///     struct CallWithStatusContext {
///       using ElementwiseFunctionType = ElementwiseFunction<1>;
///       ElementwiseFunction<1, Status*> orig_function;
///       void *orig_context;
///       Status status;
///       struct LoopTemplate {
///         template <typename ArrayAccessor>
///         static Index Loop(void* void_context, Index count,
///                           typename ArrayAccessor::Array array) {
///           auto* context = static_cast<WrapContext*>(void_context);
///           return context->orig_function[ArrayAccessor::buffer_kind](
///               context->orig_context, count, array, &status);
///         }
///       };
///     };
///     CallWithStatusContext wrapper_context{orig_function, orig_context};
///     const ElementwiseFunction<1> *wrapper_function =
///         GetElementwiseFunction<CallWithStatusContext>();
///
/// Then `wrapper_function` can be used with a context pointer of
/// `&wrapper_context` to invoke `orig_function` with an extra pointer to a
/// `Status`.

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/void_wrapper.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/default_iteration_result.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

enum class IterationBufferKind {
  /// Contiguous array (stride between consecutive elements is `sizeof(T)`), and
  /// base pointer is aligned to `alignof(T)`.
  kContiguous,

  /// Fixed stride of `byte_stride` bytes between consecutive elements.  Base
  /// pointer and `byte_stride` are aligned to `alignof(T)`.
  kStrided,

  /// Offset array specifies the byte offset of each element.  Base pointer and
  /// every element of the offset array are aligned to `alignof(T)`.
  kIndexed,
};

constexpr size_t kNumIterationBufferKinds = 3;

/// Specifies a pointer to either a contiguous, strided, or indexed buffer
/// (corresponding to one of the values of `IterationBufferKind`).
///
/// The same type is used for all three pointer types to simplify dynamic
/// dispatch.  The `IterationBufferKind` is not stored within the
/// `IterationBufferPointer`; code that consumes an `IterationBufferPointer`
/// must either be specialized at compile time for a specified buffer kind or be
/// passed the buffer kind separately.
struct IterationBufferPointer {
  /// Constructs an invalid pointer.
  IterationBufferPointer() = default;

  /// Constructs a strided pointer (corresponding to
  /// `IterationBufferKind::kContiguous` or `IterationBufferKind::kStrided`).
  explicit IterationBufferPointer(ByteStridedPointer<void> pointer,
                                  Index byte_stride)
      : pointer(pointer), byte_stride(byte_stride) {}

  /// Constructs a pointer with an offset array (corresponding to
  /// `IterationBufferKind::kIndexed`).
  explicit IterationBufferPointer(ByteStridedPointer<void> pointer,
                                  const Index* byte_offsets)
      : pointer(pointer), byte_offsets(byte_offsets) {}

  /// Base pointer of the array.
  ByteStridedPointer<void> pointer;
  union {
    /// Stride in bytes between consecutive elements.  For `kContiguous` buffers
    /// of type `T`, this must be `sizeof(T)`.  For `kStrided` buffers, this
    /// must be an integer multiple of `alignof(T)`.  For `kIndexed` buffers,
    /// this member may not be accessed.
    Index byte_stride;

    /// For `kIndexed` buffers of size `n`, pointer to an array of length `n`.
    /// For `0 <= i < n`, element `i` is at address `pointer + offsets[i]`.  For
    /// `kContiguous` and `kStrided` buffers, this member may not be accessed.
    const Index* byte_offsets;
  };
};

/// Defines a static method template `GetPointerAtOffset` for accessing an
/// element of an iteration buffer:
///
///     template <typename Element>
///     static Element *GetPointerAtOffset(Array array, Index offset);
///
/// This function template returns a pointer to the element at the specified
/// offset in `array`, which must be of kind `BufferKind`.
template <IterationBufferKind BufferKind>
struct IterationBufferAccessor;

template <>
struct IterationBufferAccessor<IterationBufferKind::kStrided> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kStrided;
  template <typename Element>
  static Element* GetPointerAtOffset(IterationBufferPointer ptr, Index offset) {
    return static_cast<Element*>(
        ptr.pointer +
        internal::wrap_on_overflow::Multiply(ptr.byte_stride, offset));
  }
};

template <>
struct IterationBufferAccessor<IterationBufferKind::kContiguous> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kContiguous;
  template <typename Element>
  static Element* GetPointerAtOffset(IterationBufferPointer ptr, Index offset) {
    return static_cast<Element*>(
        ptr.pointer + internal::wrap_on_overflow::Multiply(
                          static_cast<Index>(sizeof(Element)), offset));
  }
};

template <>
struct IterationBufferAccessor<IterationBufferKind::kIndexed> {
  constexpr static IterationBufferKind buffer_kind =
      IterationBufferKind::kIndexed;
  template <typename Element>
  static Element* GetPointerAtOffset(IterationBufferPointer ptr, Index offset) {
    return static_cast<Element*>(ptr.pointer + ptr.byte_offsets[offset]);
  }
};

template <std::size_t Arity, typename... ExtraArg>
class ElementwiseFunction;

}  // namespace internal

namespace internal_elementwise_function {

/// Helper metafunction used by ElementwiseFunctionPointer.
template <typename SequenceType, typename... ExtraArg>
struct ElementwiseFunctionPointerHelper;

// Alias used by `ElementwiseFunctionPointerHelper` (in place of e.g.
// `FirstType`) to work around MSVC 2019 (v19.24) bug:
// https://developercommunity.visualstudio.com/content/problem/919634/ice-with-decltype-and-variadic-arguments.html
template <std::size_t I>
using IterationBufferPointerHelper = internal::IterationBufferPointer;

template <std::size_t... Is, typename... ExtraArg>
struct ElementwiseFunctionPointerHelper<std::index_sequence<Is...>,
                                        ExtraArg...> {
  using type = Index (*)(void*, Index, IterationBufferPointerHelper<Is>...,
                         ExtraArg...);
};

template <typename, typename...>
struct SimpleLoopTemplate;

/// LoopTemplate implementation for an element-wise function that returns either
/// `void` or `true`.
///
/// \tparam Func The element-wise function.
/// \tparam Element... The array element types.
/// \tparam ExtraArg Additional arguments supplied to `Func`.
///
/// If `Func` is an empty type (such as a capture-less lambda), the context
/// pointer is ignored and may be `nullptr`.  Otherwise, the context pointer
/// must be a valid pointer to an object of type `Func`.
template <typename Func, typename... Element, typename... ExtraArg>
struct SimpleLoopTemplate<Func(Element...), ExtraArg...> {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<sizeof...(Element), ExtraArg...>;
  /// \tparam ArrayAccessor The ArrayAccessor type.
  template <typename ArrayAccessor>
  static Index Loop(
      void* context, Index count,
      internal::FirstType<internal::IterationBufferPointer, Element>... pointer,
      ExtraArg... extra_arg) {
    internal::PossiblyEmptyObjectGetter<Func> func_helper;
    Func& func = func_helper.get(static_cast<Func*>(context));
    for (Index i = 0; i < count; ++i) {
      if (!internal::Void::CallAndWrap(
              func,
              ArrayAccessor::template GetPointerAtOffset<Element>(pointer,
                                                                  i)...,
              extra_arg...)) {
        return i;
      }
    }
    return count;
  }
};

}  // namespace internal_elementwise_function

namespace internal {

/// Type alias that evaluates to the type
/// `Index (*)(void *, Index, IterationBufferPointer..., ExtraArg...)`, where
/// the `IterationBufferPointer` parameter is repeated `Arity` times.
template <std::size_t Arity, typename... ExtraArg>
using SpecializedElementwiseFunctionPointer =
    typename internal_elementwise_function::ElementwiseFunctionPointerHelper<
        std::make_index_sequence<Arity>, ExtraArg...>::type;

/// Combines a pointer to an `ElementwiseFunction` with a context pointer with
/// which it can be invoked.
///
/// Typically, `function` points to a static constant.
template <std::size_t Arity, typename... ExtraArg>
struct ElementwiseClosure {
  using Function = ElementwiseFunction<Arity, ExtraArg...>;
  constexpr static std::size_t arity = Arity;
  const Function* function;
  void* context;
};

/// Representation of a type-erased function that can be invoked with
/// contiguous, fixed-strided arrays, and indirect arrays specified by an
/// additional offset array.
///
/// This representation is suitable for iterating over arrays with arbitrary
/// index space transforms.
template <std::size_t Arity, typename... ExtraArg>
class ElementwiseFunction {
 public:
  constexpr static std::size_t arity = Arity;

  using Closure = ElementwiseClosure<Arity, ExtraArg...>;

  using SpecializedFunctionPointer =
      SpecializedElementwiseFunctionPointer<Arity, ExtraArg...>;

  constexpr ElementwiseFunction() = default;

  template <typename LoopTemplate>
  constexpr explicit ElementwiseFunction(LoopTemplate)
      : functions_{
            &LoopTemplate::template Loop<
                IterationBufferAccessor<IterationBufferKind::kContiguous>>,
            &LoopTemplate::template Loop<
                IterationBufferAccessor<IterationBufferKind::kStrided>>,
            &LoopTemplate::template Loop<
                IterationBufferAccessor<IterationBufferKind::kIndexed>>} {}

  constexpr SpecializedFunctionPointer operator[](
      IterationBufferKind buffer_kind) const {
    return functions_[static_cast<size_t>(buffer_kind)];
  }

  constexpr SpecializedFunctionPointer& operator[](
      IterationBufferKind buffer_kind) {
    return functions_[static_cast<size_t>(buffer_kind)];
  }

 private:
  /// Pointer to the function to be invoked (along with the `context` pointer)
  /// by users of this facility for each buffer kind.
  SpecializedFunctionPointer functions_[kNumIterationBufferKinds];
};

/// Convenience interface for obtaining an `ElementwiseFunction` for a given
/// `LoopTemplate`.
///
/// Example usage:
///
///     struct MyFunc {
///       template <typename ArrayAccessor>
///       Index Loop(void *context, Index count,
///                  IterationBufferPointer pointer) {
///         MyFunc *obj = static_cast<MyFunc*>(context);
///         /// ...
///       }
///     };
///
///     // Implicitly converts to `ElementwiseFunction`.
///     ElementwiseFunction<1> func = GetElementwiseFunction<LoopTemplate>();
///     // Implicitly converts to `const ElementwiseFunction*` for use with
///     // `ElementwiseClosure`.
///     const ElementwiseFunction<1>* func_ptr =
///         GetElementwiseFunction<LoopTemplate>();
///     // Create closure directly.
///     MyFunc my_obj;
///     ElementwiseClosure<1> closure =
///         { GetElementwiseFunction<LoopTemplate>(), &my_obj };
///
/// \tparam LoopTemplate A model of the LoopTemplate concept.
template <typename LoopTemplate>
struct GetElementwiseFunction {
  using ElementwiseFunctionType =
      typename LoopTemplate::ElementwiseFunctionType;

  /// Defines a static constant `ElementwiseFunction` for use with
  /// `ElementwiseClosure`.
  constexpr static ElementwiseFunctionType function{LoopTemplate{}};

  /// Convenience conversion operator to `const ElementwiseFunction*` for use
  /// with `ElementwiseClosure`.
  constexpr operator const ElementwiseFunctionType*() const {
    return &function;
  }

  /// Convenience conversion operator to `ElementwiseFunction`.
  ///
  /// \tparam Arity An arity supported by `LoopTemplate`.
  /// \tparam ExtraArg... The additional arguments required by `LoopTemplate`.
  constexpr operator ElementwiseFunctionType() const { return function; }
};

template <typename LoopTemplate>
constexpr typename LoopTemplate::ElementwiseFunctionType
    GetElementwiseFunction<LoopTemplate>::function;

template <typename, typename...>
struct SimpleElementwiseFunction;

/// Convenience interface for creating a type-erased element-wise function from
/// a function object type.
///
/// This class inherits from `GetElementwiseFunction` (defined above) and is
/// therefore implicitly convertible to compatible `ElementwiseFunction` types.
/// Additionally, it provides convenience interfaces for obtaining an
/// `ElementwiseClosure`.
///
/// This differs from the more general `GetElementwiseFunction` in that the
/// `Func` argument is a regular, non-type-erased function object, rather than a
/// `LoopTemplate`.
///
/// Example usage:
///
///     struct MyFunc {
///       bool operator()(int *a, float *b);
///     };
///
///     MyFunc obj;
///     // Implicitly converts to `ElementwiseFunction`.
///     ElementwiseFunction<2> func =
///         SimpleElementwiseFunction<MyFunc, int, float>();
///
///     // Implicitly converts to `const ElementwiseFunction*` (static constant)
///     const ElementwiseFunction<2>* func_ptr =
///         SimpleElementwiseFunction<MyFunc, int, float>();
///
///     // Converts to the closure `{ func_ptr, &obj }`.
///     ElementwiseClosure<2> closure =
///         SimpleElementwiseFunction<MyFunc, int, float>::Closure(&obj);
///
///     // Converts to the closure `{ func_ptr, nullptr }` since `MyFunc` is
///     // empty.
///     ElementwiseClosure<2> stateless_closure =
///         SimpleElementwiseFunction<MyFunc, int, float>();
///
/// \tparam Func The function object type.  Must be callable with
///     `(Element*..., ExtraArg...)`, and return either a type explicitly
///     convertible to `bool`, or `void` (which is equivalent to returning
///     `true`).
/// \tparam Element... The element types of the arrays over which the
///     element-wise function operates.
/// \tparam ExtraArg... Extra parameter types.
template <typename Func, typename... Element, typename... ExtraArg>
struct SimpleElementwiseFunction<Func(Element...), ExtraArg...>
    : public GetElementwiseFunction<
          internal_elementwise_function::SimpleLoopTemplate<
              std::remove_reference_t<Func>(Element...), ExtraArg...>> {
  using ElementwiseFunctionType =
      internal::ElementwiseFunction<sizeof...(Element), ExtraArg...>;
  using ClosureType =
      internal::ElementwiseClosure<sizeof...(Element), ExtraArg...>;

  /// Convenience interface for obtaining an `ElementwiseClosure` with a
  /// `context` pointer of `func`.
  ///
  /// If `Func` is an empty type, the conversion operator defined below, which
  /// does not require specify a pointer to `Func`, can be used instead.
  constexpr static ClosureType Closure(std::remove_reference_t<Func>* func) {
    return ClosureType{SimpleElementwiseFunction{},
                       const_cast<remove_cvref_t<Func>*>(func)};
  }

  /// Convenience conversion operator to `ElementwiseClosure` with a `context`
  /// of `nullptr` for use with empty `Func` types.
  ///
  /// \tparam ExtraArg Extra arguments required by `Func`.
  /// \requires `std::is_empty<Func>::value`.
  template <
      int&... ExplicitArgumentBarrier,
      std::enable_if_t<(sizeof...(ExplicitArgumentBarrier) == 0 &&
                        std::is_empty<remove_cvref_t<Func>>::value)>* = nullptr>
  constexpr operator ClosureType() const {
    return {SimpleElementwiseFunction{}, nullptr};
  }
};
}  // namespace internal

namespace internal_elementwise_function {
template <std::size_t Arity, typename... ExtraArg, typename Pointers,
          std::size_t... Is>
inline Index InvokeElementwiseFunctionImpl(
    std::index_sequence<Is...>,
    internal::SpecializedElementwiseFunctionPointer<Arity, ExtraArg...>
        function,
    void* context, Index size, const Pointers& pointers,
    ExtraArg... extra_arg) {
  using std::get;
  return function(context, size, get<Is>(pointers)...,
                  std::forward<ExtraArg>(extra_arg)...);
}
}  // namespace internal_elementwise_function

/// Invokes an `ElementwiseClosure` with the specified buffers.
///
/// \param closure The `ElementwiseClosure` to invoke.
/// \param buffer_kind The buffer kind of `pointers`.
/// \param size The number of elements in each buffer.
/// \param pointers `get<I>`-compatible container of `IterationBufferPointer` of
///     length `Arity`, such as `std::array<IterationBufferPointer,Arity>`.
/// \param extra_arg... Extra arguments to pass to `closure`.
/// \returns The element count returned by the closure.
namespace internal {
template <std::size_t Arity, typename... ExtraArg, typename Pointers>
inline Index InvokeElementwiseClosure(
    ElementwiseClosure<Arity, ExtraArg...> closure,
    IterationBufferKind buffer_kind, Index size, const Pointers& pointers,
    internal::type_identity_t<ExtraArg>... extra_arg) {
  return internal_elementwise_function::InvokeElementwiseFunctionImpl<
      Arity, ExtraArg...>(std::make_index_sequence<Arity>{},
                          (*closure.function)[buffer_kind], closure.context,
                          size, pointers, std::forward<ExtraArg>(extra_arg)...);
}

/// Invokes a `SpecializedElementwiseFunctionPointer` with the specified
/// buffers.
///
/// \tparam Arity The number of arrays with which to invoke the element-wise
///     function.  Must be specified explicitly.
/// \param function The elementwise function to invoke.
/// \param size The number of elements in each buffer.
/// \param pointers `get<I>`-compatible container of `IterationBufferPointer` of
///     length `Arity`, such as `std::array<IterationBufferPointer,Arity>`.
/// \param extra_arg... Extra trailing arguments to pass to `function`.
/// \returns The element count returned by `function`.
template <std::size_t Arity, typename... ExtraArg, typename Pointers>
inline Index InvokeElementwiseFunction(
    SpecializedElementwiseFunctionPointer<Arity, ExtraArg...> function,
    void* context, Index size, const Pointers& pointers,
    ExtraArg... extra_arg) {
  return internal_elementwise_function::InvokeElementwiseFunctionImpl<
      Arity, ExtraArg...>(std::make_index_sequence<Arity>{}, function, context,
                          size, pointers, std::forward<ExtraArg>(extra_arg)...);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_ELEMENTWISE_FUNCTION_H_
