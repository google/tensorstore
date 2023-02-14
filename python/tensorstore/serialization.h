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

#ifndef THIRD_PARTY_PY_TENSORSTORE_SERIALIZATION_H_
#define THIRD_PARTY_PY_TENSORSTORE_SERIALIZATION_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <cstddef>
#include <memory>
#include <string>

#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/serialization/serialization.h"

/// \file
///
/// Integration between Python pickling and TensorStore serialization.
///
/// Python pickling works by recursively "reducing" (via `__reduce__` or
/// `__getstate__`) pickle-compatible objects to other pickle-compatible objects
/// until primitive types (like `bool`, `int`, `float`, `str`, `bytes`) and
/// tuples/dicts/lists thereof are reached.  If this reduction results in
/// multiple references to the same object (as determined by pointer equality),
/// the object is encoded only once in the pickled representation and the shared
/// references are reconstructed when unpickling.
///
/// TensorStore serialization supports a similar "indirect reference" mechanism
/// whereby objects with multiple shared references are encoded only once and
/// the shared references are correct reconstructed when decoding.
///
/// There are two forms of integration between Python pickling and TensorStore
/// serialization that are potentially useful:
///
/// 1. Pickling/unpickling of type provided by this library the Python API,
///    where the type supports serialization and additionally may transitively
///    hold a reference to a pickle-compatible Python object.
///
/// 2. Serialization of a type provided by this library from the C++ API, where
///    the type may transitively hold a reference to a pickle-compatible Python
///    object.
///
/// Currently only (1) is implemented, and the implementation is as follows:
///
/// - Bytes written directly to the `EncodeSink::writer` are simply accumulated
///   in an `absl::Cord` and then converted to a `PyBytes` object when done.
///
/// - For indirect object references that written using
///   `EncodeSink::DoIndirect`, we defer the actual encoding of the object, and
///   instead store a wrapper Python object of type `tensorstore._Encodable`
///   that holds a reference to the C++ smart pointer and the
///   `ErasedEncodeWrapperFunction`.  The actual encoding of the object does not
///   occur until the pickle implementation calls the `__reduce__` function of
///   the `tensorstore._Encodable` object.  In order to handle references to
///   actual Python objects, if the smart pointer type is `GilSafePythonHandle`,
///   indicating an actual Python object, then we just store the object directly
///   without creating any wrapper object.
///
/// - The actual pickled representation is a `PyList` where the first element is
///   the `PyBytes` object containing all directly-written data and the
///   remaining elements are the wrapper objects corresponding to indirect
///   references.
///
/// - We must ensure that if a given C++ smart pointer is specified as an
///   indirect reference multiple times, the identical Python wrapper object is
///   returned.  Since Python does not provide a way to store any state specific
///   to an individual pickling operation, we instead rely on a global hash
///   table that maps the C++ pointer address to the corresponding Python
///   wrapper object, similar to the hash table that pybind11 maintains
///   internally.  Once the Python object is destroyed (typically at the end of
///   the pickling operation), the entry is removed from the hash table,
///   preventing the hash table from growing without bound.  There is a
///   possibility that multiple concurrent pickling operations, in which case
///   the same wrapper object may be used by multiple pickling operations, but
///   that is not a problem provided that the `ErasedEncodeWrapperFunction`
///   values specified for a given smart pointer are all interchangeable.
///
/// - The pickled `tensorstore._Encodable` objects get unpickled as
///   `tensorstore._Decodable` objects.  These objects simply hold the pickled
///   representation, and also an initially null `std::shared_ptr<void>` and
///   pointer to `std::type_info` corresponding to the decoded object.  The
///   first time the wrapper object is referenced during decoding, we create the
///   C++ object using the specified `ErasedDecodeWrapperFunction`.  For
///   subsequent references, we just verify that the `type_info` matches and
///   return another reference to the already-decoded object.  The
///   `tensorstore._Decodable` objects will typically be destroyed at the end of
///   the unpicling operation.

namespace tensorstore {
namespace internal_python {

/// Converts a Cord to a Python bytes object.
///
/// If an error occurs, returns `nullptr` and sets the Python error indicator.
pybind11::object BytesFromCord(const absl::Cord& cord) noexcept;

/// Allows encoding serializable values to a pickle-compatible representation.
///
/// The specified `encode` callback should encode objects to the provided
/// `sink`, and return `true` to indicate success or `false` in the case of an
/// error.
///
/// The pickle-compatible representation is a `PyList` where the first element
/// is a `PyBytes` object containing any directly-encoded data, and the
/// remaining elements are Python objects corresponding to the indirect
/// references.
///
/// In the case of an error, either returns `nullptr` and sets the Python error
/// indicator, or returns an error status.
Result<pybind11::object> PickleEncodeImpl(
    absl::FunctionRef<bool(serialization::EncodeSink& sink)> encode) noexcept;

/// Same as `PickleEncodeImpl`, but all errors are reported by throwing an
/// exception, rather than via `absl::Status` or the Python error indicator.
pybind11::object PickleEncodeOrThrowImpl(
    absl::FunctionRef<bool(serialization::EncodeSink& sink)> encode);

/// Decodes the pickle-compatible representation produced by `PickleEncodeImpl`.
///
/// The specified `decode` callback should decode objects from the provided
/// `source`, and return `true` to indicate success or `false` in the case of an
/// error.
absl::Status PickleDecodeImpl(
    pybind11::handle rep,
    absl::FunctionRef<bool(serialization::DecodeSource& source)>
        decode) noexcept;

/// Converts a single serializable value to a pickle-compatible representation.
///
/// The actual representation is a PyList where the first element is a PyBytes
/// object and the remaining elements correspond to indirect object references.
template <typename T, typename ElementSerializer = serialization::Serializer<T>>
pybind11::object EncodePickle(const T& value,
                              const ElementSerializer& serializer = {}) {
  return PickleEncodeOrThrowImpl([&](serialization::EncodeSink& sink) {
    return serializer.Encode(sink, value);
  });
}

/// Decodes a single serializable value from its pickle-compatible
/// representation (as returned by `EncodePickle`).
template <typename T, typename ElementSerializer = serialization::Serializer<T>>
void DecodePickle(pybind11::handle obj, T& value,
                  const ElementSerializer& serializer = {}) {
  ThrowStatusException(
      PickleDecodeImpl(obj, [&](serialization::DecodeSource& source) {
        return serializer.Decode(source, value);
      }));
}

/// Returns a two-argument tuple `(callable, (arg,))`, suitable for returning
/// from a `__reduce__` implementation.
///
/// \param callable Python callable object that is picklable.
/// \param arg Python object that is picklable, specifies the single argument
///     with which `callable` will be invoked.
/// \returns The new tuple object on success.  If an error occurs, returns
///     `nullptr` and sets the Python error indicator.  Never throws exceptions.
pybind11::object MakeReduceSingleArgumentReturnValue(pybind11::object callable,
                                                     pybind11::object arg);

/// Defines pickling support for a pybind11-bound class using the specified
/// serializer.
///
/// \tparam Holder Either the bound class type, or its smart pointer holder
///     type.  Must be specified explicitly.
/// \param cls Pybind11 class binding on which to install the pickling support.
/// \param serializer Serializer to use, must be compatible with the `Holder`
///     type.
template <typename Holder, typename Cls,
          typename Serializer = serialization::Serializer<Holder>>
void EnablePicklingFromSerialization(Cls& cls, Serializer serializer = {}) {
  cls.def(pybind11::pickle(
      [serializer](const Holder& holder) {
        return EncodePickle(holder, serializer);
      },
      [serializer](pybind11::object rep) {
        Holder value;
        DecodePickle<Holder>(rep, value, serializer);
        return value;
      }));
}

/// Convenience overload of `EnablePicklingFromSerialization` that just uses the
/// bound class type `T`, rather than an explicitly specified `Holder` type
/// which could be a smart pointer.
template <int&... ExplicitArgumentBarrier, typename T, typename... Extra,
          typename Serializer = serialization::Serializer<T>>
void EnablePicklingFromSerialization(pybind11::class_<T, Extra...>& cls,
                                     Serializer serializer = {}) {
  EnablePicklingFromSerialization<T>(cls, serializer);
}

/// Defines the specified function as the `_unpickle` member of `cls`, and
/// ensures that this member is itself picklable as a global.
///
/// This is used by `EnableGarbageCollectedObjectPicklingFromSerialization` to
/// define the unpickle implementation that is referenced by the return value of
/// the `__reduce__` function that it also defines.  See that function for
/// details.
void DefineUnpickleMethod(pybind11::handle cls, pybind11::object function);

/// Defines pickling support for a class defined using
/// `GarbageCollectedPythonObject` using the specified serializer.
///
/// For a type to be compatible with the `pickle` module, it can define a
/// `__reduce__` method:
///
/// https://docs.python.org/3/library/pickle.html#object.__reduce__
///
/// In particular, the non-static `__reduce__` method must return a picklable
/// tuple, where the first element is a callable object, and the second element
/// is a tuple of arguments with which to invoke that callable.
///
/// Therefore, this function defines both a `__reduce__` method as well as a
/// static `_unpickle` method that is used as the callable.  The tuple of
/// arguments (that will be passed to the `_unpickle` method) always consists of
/// a single `PyList` object returned by `EncodePickle`.
///
/// To work around https://github.com/pybind/pybind11/issues/2722 we define the
/// `_unpickle` method in a special way to ensure it is itself picklable, see
/// the implementation of `DefineUnpickleMethod` for details.
///
/// \param cls Pybind11 class binding on which to install the pickling support.
/// \param serializer Serializer to use, must be compatible with the
///     `Self::ContainedValue` type.
template <typename Self, typename Serializer = serialization::Serializer<
                             typename Self::ContainedValue>>
void EnableGarbageCollectedObjectPicklingFromSerialization(
    pybind11::class_<Self>& cls, Serializer serializer = {}) {
  using ContainedValue = typename Self::ContainedValue;
  static_assert(
      std::is_base_of_v<GarbageCollectedPythonObject<Self, ContainedValue>,
                        Self>);

  cls.def("__reduce__", [serializer](Self& self) {
    auto cls = pybind11::handle(reinterpret_cast<PyObject*>(Self::python_type));
    auto reduce_val = MakeReduceSingleArgumentReturnValue(
        cls.attr("_unpickle"), EncodePickle(self.value, serializer));
    if (!reduce_val) throw pybind11::error_already_set();
    return reduce_val;
  });

  DefineUnpickleMethod(
      cls, pybind11::cpp_function([serializer](pybind11::object rep) {
        ContainedValue value;
        DecodePickle<ContainedValue>(rep, value, serializer);
        return typename Self::Handle(std::move(value));
      }));
}

struct PythonHandleDirectSerializer {
  [[nodiscard]] static bool Encode(serialization::EncodeSink& sink,
                                   const GilSafePythonHandle& value);
  [[nodiscard]] static bool Decode(serialization::DecodeSource& source,
                                   GilSafePythonHandle& value);
};

/// Serializer that ensures the GIL is held when encoding/decoding the contained
/// object, which is assumed to be "GIL-unsafe".
///
/// We can say a type is GIL-safe if it can be used without holding the GIL
/// (e.g. a regular C++ object, `GilSafeHolder`, or `GilSafePythonHandle`),
/// while a type is GIL-unsafe if it can only be used while holding the GIL
/// (e.g. `pybind11::object`).
///
/// Since TensorStore serialization may in general be used from a thread that is
/// not holding the GIL, if the serializer for a given type uses Python APIs, it
/// must ensure the GIL is held.
///
/// We use the convention that serializers for GIL-unsafe types may assume the
/// GIL is already held, while serializers for GIL-safe types must not assume
/// the GIL is held.  The `GilSafeSerializer` adapter transforms a GIL-unsafe
/// serializer for a GIL-unsafe type `T` into a GIL-safe serializer for
/// `GilSafeHolder<T>`.
///
/// This is used to define the default serializer for `GilSafeHolder`.
template <typename T, typename BaseSerializer = serialization::Serializer<T>>
struct GilSafeSerializer {
  [[nodiscard]] bool Encode(serialization::EncodeSink& sink,
                            const GilSafeHolder<T>& value) const noexcept {
    ExitSafeGilScopedAcquire gil;
    if (!gil.acquired()) {
      sink.Fail(PythonExitingError());
      return false;
    }
    return base_serializer.Encode(sink, *value);
  }

  [[nodiscard]] bool Decode(serialization::DecodeSource& source,
                            GilSafeHolder<T>& value) const noexcept {
    ExitSafeGilScopedAcquire gil;
    if (!gil.acquired()) {
      source.Fail(PythonExitingError());
      return false;
    }
    return base_serializer.Decode(source, *value);
  }
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS BaseSerializer base_serializer = {};
};

}  // namespace internal_python

namespace serialization {
template <typename T>
struct Serializer<internal_python::GilSafeHolder<T>>
    : public internal_python::GilSafeSerializer<T> {};

}  // namespace serialization

}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_python::PythonWeakRef)

#endif  // THIRD_PARTY_PY_TENSORSTORE_SERIALIZATION_H_
