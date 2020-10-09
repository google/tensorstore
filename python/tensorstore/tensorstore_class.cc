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

#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include "python/tensorstore/array_type_caster.h"
#include "python/tensorstore/context.h"
#include "python/tensorstore/data_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/index_space.h"
#include "python/tensorstore/json_type_caster.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/spec.h"
#include "python/tensorstore/transaction.h"
#include "python/tensorstore/write_futures.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/array.h"
#include "tensorstore/cast.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/array/array.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/json_pprint_python.h"
#include "tensorstore/open.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/progress.h"
#include "tensorstore/rank.h"
#include "tensorstore/resize_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

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
             return internal_python::PrettyPrintJsonAsPythonRepr(
                 self.spec() |
                     [](const auto& spec) {
                       return spec.ToJson(IncludeDefaults{false});
                     },
                 "TensorStore(", ")");
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
          py::arg("dtype"))
      .def(py::pickle(
          [](const TensorStore<>& self) -> py::tuple {
            auto builder = internal::ContextSpecBuilder::Make();
            auto spec = ValueOrThrow(self.spec({}, builder));
            auto pickled_context =
                internal_python::PickleContextSpecBuilder(std::move(builder));
            auto json_spec = ValueOrThrow(spec.ToJson());
            return py::make_tuple(py::cast(json_spec),
                                  std::move(pickled_context));
          },
          [](py::tuple t) -> tensorstore::TensorStore<> {
            auto json_spec = py::cast<::nlohmann::json>(t[0]);
            auto context =
                WrapImpl(internal_python::UnpickleContextSpecBuilder(t[1]));
            py::gil_scoped_release gil_release;
            return ValueOrThrow(
                tensorstore::Open(std::move(context), std::move(json_spec))
                    .result());
          }));

  cls_tensorstore.attr("__iter__") = py::none();

  cls_tensorstore.def_property_readonly(
      "transaction",
      [](const TensorStore<>& self) {
        return internal::TransactionState::ToCommitPtr(self.transaction());
      },
      R"(Associated transaction used for read/write operations.)");

  cls_tensorstore.def(
      "with_transaction",
      [](const TensorStore<>& self,
         internal::TransactionState::CommitPtr transaction) {
        return ValueOrThrow(self | internal::TransactionState::ToTransaction(
                                       std::move(transaction)));
      },
      R"(Returns a transaction-bound view of this TensorStore.

The returned view may be used to perform transactional read/write operations.
)");

  DefineIndexTransformOperations(
      &cls_tensorstore,
      [](std::shared_ptr<TensorStore<>> self) {
        return internal::TensorStoreAccess::handle(*self).transform;
      },
      [](std::shared_ptr<TensorStore<>> self, IndexTransform<> new_transform) {
        auto handle = internal::TensorStoreAccess::handle(*self);
        handle.transform = std::move(new_transform);
        return internal::TensorStoreAccess::Construct<TensorStore<>>(
            std::move(handle));
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
         internal_context::ContextImplPtr context) -> TensorStore<> {
        if (!context) {
          context = internal_context::Access::impl(Context::Default());
        }
        return ValueOrThrow(FromArray(WrapImpl(std::move(context)), array));
      },
      "Returns a TensorStore that reads/writes from an in-memory array.",
      py::arg("array"), py::arg("context") = nullptr);

  m.def(
      "array",
      [](ArrayArgumentPlaceholder array, DataType dtype,
         internal_context::ContextImplPtr context) -> TensorStore<> {
        if (!context) {
          context = internal_context::Access::impl(Context::Default());
        }
        SharedArray<void> converted_array;
        ConvertToArray</*Element=*/void, /*Rank=*/dynamic_rank,
                       /*NoThrow=*/false, /*AllowCopy=*/true>(
            array.obj, &converted_array, dtype);
        return ValueOrThrow(FromArray(WrapImpl(std::move(context)),
                                      std::move(converted_array)));
      },
      "Returns a TensorStore that reads/writes from an in-memory array.",
      py::arg("array"), py::arg("dtype"), py::arg("context") = nullptr);

  m.def(
      "open",
      [](const Spec& spec, std::optional<bool> read, std::optional<bool> write,
         std::optional<bool> open, std::optional<bool> create,
         std::optional<bool> delete_existing,
         std::optional<bool> allow_option_mismatch,
         internal_context::ContextImplPtr context,
         internal::TransactionState::CommitPtr transaction) {
        if (!context) {
          context = internal_context::Access::impl(Context::Default());
        }
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
        return tensorstore::Open(
            WrapImpl(std::move(context)),
            internal::TransactionState::ToTransaction(std::move(transaction)),
            spec, std::move(options));
      },
      "Opens a TensorStore", py::arg("spec"), py::arg("read") = std::nullopt,
      py::arg("write") = std::nullopt, py::arg("open") = std::nullopt,
      py::arg("create") = std::nullopt,
      py::arg("delete_existing") = std::nullopt,
      py::arg("allow_option_mismatch") = std::nullopt,
      py::arg("context") = nullptr, py::arg("transaction") = nullptr);
}

}  // namespace internal_python
}  // namespace tensorstore
