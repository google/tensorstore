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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "absl/container/flat_hash_map.h"
#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/serialization.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/str_cat.h"

// Include this Python header last since it defines some macros that are not
// properly prefixed and we want to avoid problems with any other headers.
#include <structmember.h>

namespace py = ::pybind11;

namespace tensorstore {
namespace internal_python {

namespace {

using PickleObjectRegistry = absl::flat_hash_map<void*, PyObject*>;

/// Global mapping from C++ pointer value to corresponding Python wrapper object
/// of type `tensorstore._Encodable`.
///
/// This allows the shared object reference mechanism supported by pickling to
/// interoperate with the shared object mechanism used by TensorStore
/// serialization.
///
/// The value is always a pointer to an `EncodableObject`.  The key is always
/// equal to the `cpp_data.cpp_object.get()` value within the `EncodableObject`.
///
/// An entry is added by `PickleEncodeSink::DoIndirect` when a new wrapper
/// object needs to be created.
///
/// Entries are automatically removed when the Python wrapper object is
/// destroyed, which typically happens at the end of the pickling operation.
///
/// \threadsafety Must only be accessed while holding the GIL.
internal::NoDestructor<PickleObjectRegistry> pickle_object_registry;

/// Python object representation for `tensorstore._Encodable` wrapper objects.
///
/// Every object of this type is stored in `pickle_object_registry`.
///
/// Objects of this type are created only by `PickleEncodeSink`, and cannot be
/// created directly from Python.
struct EncodableObject {
  /// Holds the non-trivial members of `EncodableObject`, which allows
  /// construction and destruction independent of the `EncodableObject` as a
  /// whole, in order to better interoperate with Python's object allocation
  /// interface.
  struct CppData {
    /// C++ smart pointer passed to `PickleEncodeSink::DoIndirect`.  The pointer
    /// value is equal to the key under which the containing `EncodableObject`
    /// is stored.
    std::shared_ptr<void> cpp_object;

    /// Encode function to use when pickling the `EncodableObject`.
    serialization::EncodeSink::ErasedEncodeWrapperFunction encode;
  };

  // clang-format off
  PyObject_HEAD
  CppData cpp_data;
  // clang-format on
};

/// Python object representation for `tensorstore._Decodable` wrapper objects.
///
/// This is the counterpart to `tensorstore.Encodable`: the `__new__` method for
/// this type is returned by the `__reduce__` method of `tensorstore._Encodable`
/// to use for reconsructing the object.
///
/// Since objects of this type are created during unpickling before we know the
/// actual C++ type or decode function, we just store the pickled representation
/// and defer decoding until it is accessed via
/// `PickleDecodeSource::DoIndirect`.
///
/// This type is not itself picklable.
struct DecodableObject {
  struct CppData {
    /// Pickled representation of this object, if not yet decoded.  Set to
    /// `nullptr` once `cpp_object` has been set.
    pybind11::object python_object;

    /// Type info of the smart pointer type (not the pointee type) that was
    /// specified when `cpp_object` was decoded, or `nullptr` if not yet
    /// decoded.
    const std::type_info* type_info = nullptr;

    /// Decoded object, or `nullptr` if not yet decoded.
    std::shared_ptr<void> cpp_object;
  };

  // clang-format off
  PyObject_HEAD
  CppData cpp_data;
  // clang-format on
};

/// Static Python type object corresponding to `tensorstore._Decodable`.
///
/// This is not fully initialized until `RegisterSerializationBindings` is
/// called.
PyTypeObject DecodableObjectType = [] {
  PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
  t.tp_name = "tensorstore._Decodable";
  t.tp_basicsize = sizeof(DecodableObject);
  t.tp_itemsize = 0;
  t.tp_flags = Py_TPFLAGS_DEFAULT;
  // `tensorstore._Decodable.__new__` function, which is called to reconstruct a
  // pickled object.
  t.tp_new = [](PyTypeObject* type, PyObject* args,
                PyObject* kwds) -> PyObject* {
    // It is expected to be called with a single positional argument specifying
    // the pickled representation.  This should only be called by the Python
    // pickle implementation, with the arguments tuple returned by
    // `tensorstore._Encodable.__reduce__`.  Validation of the pickled
    // representation itself is deferred until the first call to
    // `PickleDecodeSource::DoIndirect`.
    if (!PyTuple_CheckExact(args) || PyTuple_GET_SIZE(args) != 1) {
      PyErr_SetString(PyExc_TypeError, "Expected single argument");
      return nullptr;
    }
    auto self = py::reinterpret_steal<py::object>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    auto& data = reinterpret_cast<DecodableObject*>(self.ptr())->cpp_data;
    new (&data) DecodableObject::CppData;
    data.python_object =
        py::reinterpret_borrow<py::object>(PyTuple_GET_ITEM(args, 0));
    return self.release().ptr();
  };
  t.tp_dealloc = [](PyObject* self) {
    reinterpret_cast<DecodableObject*>(self)->cpp_data.~CppData();
    Py_TYPE(self)->tp_free(self);
  };
  return t;
}();

PyMethodDef EncodableObject_methods[] = {
    // Function called to pickle a `tensorstore._Encodable` object.
    {"__reduce__",
     [](PyObject* self, PyObject* ignored) -> PyObject* {
       // As this is a Python C API callback, we must not throw exceptions, and
       // instead indicate errors by returning `nullptr` with the Python error
       // indicator set.
       auto& data = reinterpret_cast<EncodableObject*>(self)->cpp_data;
       TENSORSTORE_ASSIGN_OR_RETURN(
           auto reduce_list,
           PickleEncodeImpl([&](serialization::EncodeSink& sink) {
             return data.encode(sink, data.cpp_object);
           }),
           (SetErrorIndicatorFromStatus(_), nullptr));
       if (!reduce_list) return nullptr;
       // Use the `DecodableObjectType` type object itself as the callable for
       // unpickling.
       return MakeReduceSingleArgumentReturnValue(
                  py::reinterpret_borrow<py::object>(
                      reinterpret_cast<PyObject*>(&DecodableObjectType)),
                  std::move(reduce_list))
           .release()
           .ptr();
     },
     METH_NOARGS, ""},
    {nullptr}, /*sentinel*/
};

/// Static Python type object corresponding to `tensorstore.Encodable`.
///
/// This is not fully initialized until `RegisterSerializationBindings` is
/// called.
PyTypeObject EncodableObjectType = [] {
  PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
  t.tp_name = "tensorstore._Encodable";
  t.tp_basicsize = sizeof(EncodableObject);
  t.tp_itemsize = 0;
  t.tp_flags = Py_TPFLAGS_DEFAULT;
  t.tp_dealloc = [](PyObject* self) {
    auto& data = reinterpret_cast<EncodableObject*>(self)->cpp_data;
    pickle_object_registry->erase(data.cpp_object.get());
    data.~CppData();
    Py_TYPE(self)->tp_free(self);
  };
  t.tp_methods = EncodableObject_methods;
  return t;
}();

/// Python object representation used by `GlobalPicklableFunction`.  Refer to
/// the documentation of that function for details.
struct GlobalPicklableFunctionObject {
  // clang-format off
  PyObject_HEAD
  PyObject* module;
  PyObject* qualname;
  PyObject* function;
  // clang-format on
};

PyMemberDef GlobalPicklableFunction_members[] = {
    {"__module__", T_OBJECT, offsetof(GlobalPicklableFunctionObject, module),
     READONLY},
    {"__qualname__", T_OBJECT,
     offsetof(GlobalPicklableFunctionObject, qualname), READONLY},
    {nullptr},
};

PyMethodDef GlobalPicklableFunction_methods[] = {
    // Function called to pickle a `tensorstore._GlobalPicklableFunction`
    // object.
    {"__reduce__",
     [](PyObject* self, PyObject* ignored) -> PyObject* {
       auto& obj = *reinterpret_cast<GlobalPicklableFunctionObject*>(self);
       Py_INCREF(obj.qualname);
       return obj.qualname;
     },
     METH_NOARGS, ""},
    {nullptr}, /*sentinel*/
};

PyTypeObject GlobalPicklableFunctionObjectType = [] {
  PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
  t.tp_name = "tensorstore._GlobalPicklableFunction";
  t.tp_basicsize = sizeof(GlobalPicklableFunctionObject);
  t.tp_itemsize = 0;
  t.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  t.tp_traverse = [](PyObject* self, visitproc visit, void* arg) -> int {
    auto& obj = *reinterpret_cast<GlobalPicklableFunctionObject*>(self);
    Py_VISIT(obj.function);
    return 0;
  };
  t.tp_dealloc = [](PyObject* self) {
    PyObject_GC_UnTrack(self);
    auto& obj = *reinterpret_cast<GlobalPicklableFunctionObject*>(self);
    Py_XDECREF(obj.module);
    Py_XDECREF(obj.qualname);
    Py_XDECREF(obj.function);
    Py_TYPE(self)->tp_free(self);
  };
  t.tp_call = [](PyObject* self, PyObject* args,
                 PyObject* kwargs) -> PyObject* {
    auto& obj = *reinterpret_cast<GlobalPicklableFunctionObject*>(self);
    return PyObject_Call(obj.function, args, kwargs);
  };
  t.tp_methods = GlobalPicklableFunction_methods;
  t.tp_members = GlobalPicklableFunction_members;
  return t;
}();

class PickleEncodeSink final : public serialization::EncodeSink {
 public:
  PickleEncodeSink(riegeli::Writer& writer, pybind11::handle rep) noexcept
      : serialization::EncodeSink(writer), rep_(rep) {}

  bool DoIndirect(const std::type_info& type,
                  ErasedEncodeWrapperFunction encode,
                  std::shared_ptr<void> object) override {
    GilScopedAcquire gil_acquire;
    py::object python_object;
    if (type == typeid(PythonWeakRef)) {
      // Special case: reference to actual Python object.  Just store the Python
      // object directly.
      python_object = py::reinterpret_borrow<py::object>(
          static_cast<PyObject*>(object.get()));
    } else if (auto it = pickle_object_registry->find(object.get());
               it != pickle_object_registry->end()) {
      // Python wrapper object already exists.  Just return it.
      python_object = py::reinterpret_borrow<py::object>(it->second);
    } else {
      // Create a new Python wrapper object corresponding to `object`.
      auto* python_type = &EncodableObjectType;
      python_object = py::reinterpret_steal<py::object>(
          python_type->tp_alloc(python_type, 0));
      if (!python_object) goto error_indicator_set;
      auto& data =
          reinterpret_cast<EncodableObject*>(python_object.ptr())->cpp_data;
      new (&data) EncodableObject::CppData;
      pickle_object_registry->emplace(object.get(), python_object.ptr());
      data.cpp_object = std::move(object);
      data.encode = std::move(encode);
    }
    if (PyList_Append(rep_.ptr(), python_object.ptr()) != 0) {
      goto error_indicator_set;
    }
    return true;

  error_indicator_set:
    Fail(GetStatusFromPythonException());
    return false;
  }

 private:
  // PyList of indirect objects.  The first entry is reserved for the PyBytes
  // object that will contain the contents of `owned_buffer_`.
  pybind11::handle rep_;
};

class PickleDecodeSource final : public serialization::DecodeSource {
 public:
  PickleDecodeSource(riegeli::Reader& reader, pybind11::handle rep)
      : serialization::DecodeSource(reader), rep_(rep), indirect_index_(1) {}
  absl::Status Done() override {
    if (indirect_index_ != PyList_GET_SIZE(rep_.ptr())) {
      return serialization::DecodeError("Unused indirect object references");
    }
    return serialization::DecodeSource::Done();
  }

  bool DoIndirect(const std::type_info& type,
                  ErasedDecodeWrapperFunction decode,
                  std::shared_ptr<void>& value) override {
    GilScopedAcquire gil_acquire;
    if (indirect_index_ >= PyList_GET_SIZE(rep_.ptr())) {
      Fail(serialization::DecodeError(
          "Expected additional indirect object reference"));
      return false;
    }
    auto python_object = py::reinterpret_borrow<py::object>(
        PyList_GET_ITEM(rep_.ptr(), indirect_index_++));
    if (type == typeid(PythonWeakRef)) {
      // Special case: reference to actual Python object.  Just return the
      // Python object directly.
      value = internal::UnownedToShared(python_object.release().ptr());
      return true;
    }
    if (Py_TYPE(python_object.ptr()) != &DecodableObjectType) {
      Fail(serialization::DecodeError("Expected tensorstore._Decodable"));
      return false;
    }
    auto& data =
        reinterpret_cast<DecodableObject*>(python_object.ptr())->cpp_data;
    if (data.type_info) {
      if (*data.type_info != type) {
        Fail(absl::InvalidArgumentError(tensorstore::StrCat(
            "Type mismatch for indirect object, received ",
            data.type_info->name(), " but expected ", type.name())));
        return false;
      }
    } else {
      // Decode a shared C++ object from the representation returned by
      // `tensorstore._Encodable.__reduce__`.
      TENSORSTORE_RETURN_IF_ERROR(
          PickleDecodeImpl(data.python_object,
                           [&](serialization::DecodeSource& source) {
                             return decode(source, data.cpp_object);
                           }),
          (Fail(_), false));
      data.type_info = &type;
      data.python_object = {};
    }
    value = data.cpp_object;
    return true;
  }

 private:
  pybind11::handle rep_;
  size_t indirect_index_;
};

/// Returns a callable object that can be pickled as a global variable.
///
/// This is a workaround for https://github.com/pybind/pybind11/issues/2722
pybind11::object MakeGlobalPicklableFunction(pybind11::object module,
                                             pybind11::object qualname,
                                             pybind11::object function) {
  auto self = py::reinterpret_steal<py::object>(
      GlobalPicklableFunctionObjectType.tp_alloc(
          &GlobalPicklableFunctionObjectType, 0));
  if (!self) throw py::error_already_set();
  auto& obj = *reinterpret_cast<GlobalPicklableFunctionObject*>(self.ptr());
  obj.module = module.release().ptr();
  obj.qualname = qualname.release().ptr();
  obj.function = function.release().ptr();
  return self;
}

void RegisterSerializationBindings(pybind11::module_ m, Executor defer) {
  if (PyType_Ready(&DecodableObjectType) != 0) {
    throw py::error_already_set();
  }
  if (PyType_Ready(&EncodableObjectType) != 0) {
    throw py::error_already_set();
  }
  if (PyType_Ready(&GlobalPicklableFunctionObjectType) != 0) {
    throw py::error_already_set();
  }
  m.attr("_Decodable") = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject*>(&DecodableObjectType));
  m.attr("_Encodable") = py::reinterpret_borrow<py::object>(
      reinterpret_cast<PyObject*>(&EncodableObjectType));
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterSerializationBindings, /*priority=*/0);
}

}  // namespace

void DefineUnpickleMethod(pybind11::handle cls, pybind11::object function) {
  auto qualname = py::reinterpret_steal<py::str>(
      PyUnicode_FromFormat("%U._unpickle", cls.attr("__qualname__").ptr()));
  if (!qualname) throw py::error_already_set();
  cls.attr("_unpickle") = MakeGlobalPicklableFunction(
      cls.attr("__module__"), std::move(qualname), std::move(function));
}

pybind11::object BytesFromCord(const absl::Cord& cord) noexcept {
  auto obj = py::reinterpret_steal<py::object>(
      PyBytes_FromStringAndSize(nullptr, cord.size()));
  if (!obj) return obj;
  auto* buf = PyBytes_AS_STRING(obj.ptr());
  for (auto chunk : cord.Chunks()) {
    std::memcpy(buf, chunk.data(), chunk.size());
    buf += chunk.size();
  }
  return obj;
}

Result<pybind11::object> PickleEncodeImpl(
    absl::FunctionRef<bool(serialization::EncodeSink& sink)> encode) noexcept {
  auto rep = py::reinterpret_steal<py::object>(PyList_New(1));
  if (!rep) return {std::in_place};
  absl::Cord cord;
  riegeli::CordWriter writer(&cord);
  PickleEncodeSink sink(writer, rep);
  if (![&] {
        GilScopedRelease gil_release;
        return encode(sink);
      }() ||
      !sink.Close())
    return sink.status();
  auto bytes_obj = BytesFromCord(cord);
  if (!bytes_obj) return {std::in_place};
  PyList_SET_ITEM(rep.ptr(), 0, bytes_obj.release().ptr());
  return rep;
}

pybind11::object PickleEncodeOrThrowImpl(
    absl::FunctionRef<bool(serialization::EncodeSink& sink)> encode) {
  auto rep = ValueOrThrow(PickleEncodeImpl(encode));
  if (!rep) throw py::error_already_set();
  return rep;
}

absl::Status PickleDecodeImpl(
    pybind11::handle rep,
    absl::FunctionRef<bool(serialization::DecodeSource& source)>
        decode) noexcept {
  PyObject* s;
  if (!PyList_CheckExact(rep.ptr()) || (PyList_GET_SIZE(rep.ptr()) < 1) ||
      !PyBytes_CheckExact(s = PyList_GET_ITEM(rep.ptr(), 0))) {
    return absl::DataLossError(
        "Expected list of size >= 1, where first element is bytes");
  }
  riegeli::StringReader<std::string_view> reader{
      std::string_view(PyBytes_AS_STRING(s), PyBytes_GET_SIZE(s))};
  PickleDecodeSource source(reader, rep);
  if (GilScopedRelease gil_release; !decode(source)) {
    serialization::internal_serialization::FailEof(source);
    return source.status();
  }
  return source.Done();
}

pybind11::object MakeReduceSingleArgumentReturnValue(pybind11::object callable,
                                                     pybind11::object arg) {
  auto reduce_val = py::reinterpret_steal<py::object>(PyTuple_New(2));
  if (!reduce_val.ptr()) return {};
  auto reduce_args = py::reinterpret_steal<py::object>(PyTuple_New(1));
  if (!reduce_args.ptr()) return {};
  PyTuple_SET_ITEM(reduce_args.ptr(), 0, arg.release().ptr());
  PyTuple_SET_ITEM(reduce_val.ptr(), 0, callable.release().ptr());
  PyTuple_SET_ITEM(reduce_val.ptr(), 1, reduce_args.release().ptr());
  return reduce_val;
}

}  // namespace internal_python

namespace serialization {

bool Serializer<internal_python::PythonWeakRef>::Encode(
    EncodeSink& sink, const internal_python::PythonWeakRef& value) {
  internal_python::GilScopedAcquire gil_acquire;
  return sink.DoIndirect(
      typeid(internal_python::PythonWeakRef),
      [](EncodeSink& sink, const std::shared_ptr<void>& erased_value) {
        sink.Fail(absl::UnimplementedError(
            "Unsupported EncodeSink for Python object serialization"));
        return false;
      },
      internal::UnownedToShared(value.get_value_or_none().ptr()));
}

bool Serializer<internal_python::PythonWeakRef>::Decode(
    DecodeSource& source, internal_python::PythonWeakRef& value) {
  internal_python::GilScopedAcquire gil_acquire;
  std::shared_ptr<void> temp;
  if (!source.DoIndirect(
          typeid(internal_python::PythonWeakRef),
          [](DecodeSource& source, std::shared_ptr<void>& erased_value) {
            source.Fail(absl::UnimplementedError(
                "Unsupported DecodeSource for Python object serialization"));
            return false;
          },
          temp)) {
    return false;
  }
  value = py::reinterpret_steal<py::object>(static_cast<PyObject*>(temp.get()));
  return true;
}

}  // namespace serialization
}  // namespace tensorstore
