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

#include "python/tensorstore/tensorstore_class.h"

#include <optional>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/write_futures.h"
#include "pybind11/stl.h"
#include "tensorstore/cast.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

void RegisterTensorStoreBindings(pybind11::module m) {
  // TODO(jbms): Add documentation.
  py::class_<TensorStore<>, std::shared_ptr<TensorStore<>>> cls_tensorstore(
      m, "TensorStore");
  cls_tensorstore
      .def_property_readonly(
          "rank", [](const TensorStore<>& self) { return self.rank(); })
      .def_property_readonly("domain",
                             [](const TensorStore<>& self) -> IndexDomain<> {
                               return self.domain();
                             })
      .def_property_readonly(
          "dtype", [](const TensorStore<>& self) { return self.data_type(); })
      .def("spec", [](const TensorStore<>& self) { return self.spec(); })
      .def_property_readonly(
          "mode",
          [](const TensorStore<>& self) {
            std::string mode;
            if (!!(self.read_write_mode() & ReadWriteMode::read)) mode += "r";
            if (!!(self.read_write_mode() & ReadWriteMode::write)) mode += "w";
            return mode;
          })
      .def_property_readonly(
          "writable",
          [](const TensorStore<>& self) {
            return !!(self.read_write_mode() & ReadWriteMode::write);
          })
      .def_property_readonly(
          "readable",
          [](const TensorStore<>& self) {
            return !!(self.read_write_mode() & ReadWriteMode::read);
          })
      .def("__repr__",
           [](const TensorStore<>& self) -> std::string {
             if (auto spec_result = self.spec()) {
               return PrettyPrintSpec(*spec_result, "TensorStore(", ")");
             }
             return "TensorStore(...)";
           })
      .def(
          "__array__",
          [](const TensorStore<>& self, std::optional<py::dtype> dtype,
             std::optional<py::object> context) {
            py::gil_scoped_release gil_release;
            return ValueOrThrow(tensorstore::Read<zero_origin>(self).result());
          },
          py::arg("dtype") = std::nullopt, py::arg("context") = std::nullopt)
      .def(
          "read",
          [](const TensorStore<>& self,
             std::optional<ContiguousLayoutOrder> order) {
            py::gil_scoped_release gil_release;
            return tensorstore::Read<zero_origin>(self,
                                                  {order.value_or(c_order)});
          },
          py::arg("order") = "C")
      .def(
          "write",
          [](const TensorStore<>& self, const TensorStore<>& other) {
            py::gil_scoped_release gil_release;
            return tensorstore::Copy(other, self);
          },
          py::arg("source"))
      .def(
          "write",
          [](const TensorStore<>& self, ArrayArgumentPlaceholder source) {
            SharedArray<const void> source_array;
            ConvertToArray<const void, dynamic_rank, /*nothrow=*/false>(
                source.obj, &source_array, self.data_type(), 0, self.rank());
            py::gil_scoped_release gil_release;
            return tensorstore::Write(source_array, self);
          },
          py::arg("source"))
      .def(
          "resolve",
          [](const TensorStore<>& self, bool fix_resizable_bounds) {
            py::gil_scoped_release gil_release;
            ResolveBoundsOptions options = {};
            if (fix_resizable_bounds) {
              options.mode = options.mode | tensorstore::fix_resizable_bounds;
            }
            return tensorstore::ResolveBounds(self, options);
          },
          py::arg("fix_resizable_bounds") = false)
      .def(
          "astype",
          [](const TensorStore<>& self, DataType target_data_type) {
            return ValueOrThrow(tensorstore::Cast(self, target_data_type));
          },
          "Returns a read/write view as the specified data type.",
          py::arg("dtype"));
  cls_tensorstore.attr("__iter__") = py::none();

  DefineIndexTransformOperations(
      &cls_tensorstore,
      [](std::shared_ptr<TensorStore<>> self) {
        return internal::TensorStoreAccess::transform(*self);
      },
      [](std::shared_ptr<TensorStore<>> self, IndexTransform<> new_transform) {
        return internal::TensorStoreAccess::Construct<TensorStore<>>(
            internal::TensorStoreAccess::driver(*self),
            std::move(new_transform), self->read_write_mode());
      },
      [](TensorStore<> self, const TensorStore<>& source) {
        py::gil_scoped_release gil;
        return tensorstore::Copy(source, self).commit_future.result();
      },
      [](TensorStore<> self, ArrayArgumentPlaceholder source) {
        SharedArray<const void> source_array;
        ConvertToArray<const void, dynamic_rank, /*nothrow=*/false>(
            source.obj, &source_array, self.data_type(), 0, self.rank());
        {
          py::gil_scoped_release gil;
          return tensorstore::Write(std::move(source_array), self)
              .commit_future.result();
        }
      });

  m.def(
      "cast",
      [](const TensorStore<>& store, DataType target_data_type) {
        return ValueOrThrow(tensorstore::Cast(store, target_data_type));
      },
      "Returns a read/write TensorStore view as the specified data type.",
      py::arg("store"), py::arg("dtype"));

  m.def(
      "array",
      [](SharedArray<void> array,
         std::optional<Context> context) -> TensorStore<> {
        if (!context) context = Context::Default();
        return ValueOrThrow(FromArray(std::move(*context), array));
      },
      "Returns a TensorStore that reads/writes from an in-memory array.",
      py::arg("array"), py::arg("context") = std::nullopt);

  m.def(
      "array",
      [](ArrayArgumentPlaceholder array, DataType dtype,
         std::optional<Context> context) -> TensorStore<> {
        if (!context) context = Context::Default();
        SharedArray<void> converted_array;
        ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                       /*NoThrow=*/false, /*AllowCopy=*/true>(
            array.obj, &converted_array, dtype);
        return ValueOrThrow(
            FromArray(std::move(*context), std::move(converted_array)));
      },
      "Returns a TensorStore that reads/writes from an in-memory array.",
      py::arg("array"), py::arg("dtype"), py::arg("context") = std::nullopt);

  m.def(
      "open",
      [](const Spec& spec, std::optional<bool> read, std::optional<bool> write,
         std::optional<bool> open, std::optional<bool> create,
         std::optional<bool> delete_existing,
         std::optional<bool> allow_option_mismatch,
         std::optional<Context> context) {
        if (!context) context = Context::Default();
        OpenOptions options;
        if (!read && !write) {
          read = true;
          write = true;
        }
        options.read_write_mode = ReadWriteMode{};
        if (read && *read == true) {
          options.read_write_mode =
              options.read_write_mode | ReadWriteMode::read;
        }
        if (write && *write == true) {
          options.read_write_mode =
              options.read_write_mode | ReadWriteMode::write;
        }
        if (open || create || delete_existing || allow_option_mismatch) {
          OpenMode open_mode = OpenMode{};
          if (open && *open == true) {
            open_mode = open_mode | OpenMode::open;
          }
          if (create && *create == true) {
            open_mode = open_mode | OpenMode::create;
          }
          if (delete_existing && *delete_existing == true) {
            open_mode = open_mode | OpenMode::delete_existing;
          }
          if (allow_option_mismatch && *allow_option_mismatch == true) {
            open_mode = open_mode | OpenMode::allow_option_mismatch;
          }
          options.open_mode = open_mode;
        }
        return tensorstore::Open(std::move(*context), spec, std::move(options));
      },
      "Opens a TensorStore", py::arg("spec"), py::arg("read") = std::nullopt,
      py::arg("write") = std::nullopt, py::arg("open") = std::nullopt,
      py::arg("create") = std::nullopt,
      py::arg("delete_existing") = std::nullopt,
      py::arg("allow_option_mismatch") = std::nullopt,
      py::arg("context") = std::nullopt);
}

}  // namespace internal_python
}  // namespace tensorstore
