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

#include "tensorstore/kvstore/spec.h"

#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using ::tensorstore::internal::IntrusivePtr;

namespace tensorstore {
namespace kvstore {

void intrusive_ptr_increment(const DriverSpec* p) {
  intrusive_ptr_increment(
      static_cast<const internal::AtomicReferenceCount<DriverSpec>*>(p));
}

void intrusive_ptr_decrement(const DriverSpec* p) {
  intrusive_ptr_decrement(
      static_cast<const internal::AtomicReferenceCount<DriverSpec>*>(p));
}

DriverSpec::~DriverSpec() = default;

absl::Status DriverSpec::NormalizeSpec(std::string& path) {
  return absl::OkStatus();
}

Result<std::string> DriverSpec::ToUrl(std::string_view path) const {
  return absl::UnimplementedError("URL representation not supported");
}

absl::Status DriverSpec::ApplyOptions(DriverSpecOptions&& options) {
  return absl::OkStatus();
}

ContextBindingState DriverSpecPtr::context_binding_state() const {
  return get()->context_binding_state_;
}

void EncodeCacheKeyAdl(std::string* out, const DriverSpecPtr& ptr) {
  return ptr->EncodeCacheKey(out);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(Spec, [](auto is_loading,
                                                const auto& options, auto* obj,
                                                auto* j) {
  if constexpr (is_loading) {
    if (auto* s = j->template get_ptr<const std::string*>()) {
      TENSORSTORE_ASSIGN_OR_RETURN(*obj, Spec::FromUrl(*s));
      return absl::OkStatus();
    }
  } else {
    if (!obj->valid()) {
      *j = ::nlohmann::json::value_t::discarded;
      return absl::OkStatus();
    }
  }
  namespace jb = tensorstore::internal_json_binding;
  auto& registry = internal_kvstore::GetDriverRegistry();
  return jb::NestedContextJsonBinder(jb::Object(
      jb::Member("driver", jb::Projection<&Spec::driver>(registry.KeyBinder())),
      jb::Initialize([](Spec* p) {
        const_cast<DriverSpec&>(*p->driver).context_binding_state_ =
            ContextBindingState::unbound;
      }),
      jb::Member("context", jb::Projection(
                                [](const Spec& p) -> Context::Spec& {
                                  return const_cast<Context::Spec&>(
                                      p.driver->context_spec_);
                                },
                                internal::ContextSpecDefaultableJsonBinder)),
      jb::Member("path", jb::Projection(
                             [](auto& p) -> decltype(auto) { return (p.path); },
                             jb::DefaultInitializedValue())),
      [&](auto is_loading, const auto& options, auto* obj, auto* j) {
        if constexpr (is_loading) {
          TENSORSTORE_RETURN_IF_ERROR(registry.RegisteredObjectBinder()(
              is_loading, {options, obj->path}, &obj->driver, j));
          return const_cast<DriverSpec&>(*obj->driver).NormalizeSpec(obj->path);
        } else {
          return registry.RegisteredObjectBinder()(is_loading, options,
                                                   &obj->driver, j);
        }
      }))(is_loading, options, obj, j);
})

absl::Status DriverSpecPtr::Set(DriverSpecOptions&& options) {
  if (options.minimal_spec) {
    if ((*this)->use_count() != 1) *this = (*this)->Clone();
    TENSORSTORE_RETURN_IF_ERROR(
        const_cast<DriverSpec*>(get())->ApplyOptions(std::move(options)));
  }
  return absl::OkStatus();
}

absl::Status DriverSpecPtr::Set(SpecConvertOptions&& options) {
  internal::ApplyContextBindingMode(
      *this, options.context_binding_mode,
      /*default_mode=*/ContextBindingMode::retain);
  if (options.context) {
    TENSORSTORE_RETURN_IF_ERROR(BindContext(options.context));
  }
  return Set(static_cast<DriverSpecOptions&&>(options));
}

absl::Status DriverSpecPtr::BindContext(const Context& context) {
  return internal::BindContextCopyOnWriteWithNestedContext(*this, context);
}

absl::Status Spec::Set(SpecConvertOptions&& options) {
  return driver.Set(std::move(options));
}

void DriverSpecPtr::UnbindContext(
    const internal::ContextSpecBuilder& context_builder) {
  internal::UnbindContextCopyOnWriteWithNestedContext(*this, context_builder);
}

void DriverSpecPtr::StripContext() {
  internal::StripContextCopyOnWriteWithNestedContext(*this);
}

absl::Status Spec::BindContext(const Context& context) {
  return driver.BindContext(context);
}

void Spec::UnbindContext(const internal::ContextSpecBuilder& context_builder) {
  driver.UnbindContext(context_builder);
}

void Spec::StripContext() { driver.StripContext(); }

Result<std::string> Spec::ToUrl() const {
  if (!driver) {
    return absl::InvalidArgumentError("Invalid kvstore spec");
  }
  return driver->ToUrl(path);
}

}  // namespace kvstore

namespace serialization {

namespace {

using DriverSpecPtrNonNullDirectSerializer =
    RegistrySerializer<internal::IntrusivePtr<const kvstore::DriverSpec>>;

using DriverSpecPtrSerializer =
    IndirectPointerSerializer<internal::IntrusivePtr<const kvstore::DriverSpec>,
                              DriverSpecPtrNonNullDirectSerializer>;

using DriverSpecPtrNonNullSerializer = NonNullIndirectPointerSerializer<
    internal::IntrusivePtr<const kvstore::DriverSpec>,
    DriverSpecPtrNonNullDirectSerializer>;

}  // namespace

}  // namespace serialization

namespace internal_json_binding {
TENSORSTORE_DEFINE_JSON_BINDER(
    KvStoreSpecAndPathJsonBinder,
    Sequence(Member("kvstore", DefaultInitializedPredicate([](auto* obj) {
                      return !obj->valid();
                    })),
             // DEPRECATED: "path" is supported for backward compatibility only.
             LoadSave(OptionalMember(
                 "path",
                 Compose<std::string>([](auto is_loading, const auto& options,
                                         auto* obj, std::string* j) {
                   if (!obj->valid()) {
                     return absl::InvalidArgumentError(
                         "\"path\" must be specified in conjunction with "
                         "\"kvstore\"");
                   }
                   obj->AppendPathComponent(*j);
                   return absl::OkStatus();
                 })))))
}  // namespace internal_json_binding

}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr,
    tensorstore::serialization::DriverSpecPtrSerializer())

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::kvstore::Spec,
    tensorstore::serialization::ApplyMembersSerializer<
        tensorstore::kvstore::Spec>())

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpec,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::kvstore::DriverSpec>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::Spec,
    tensorstore::garbage_collection::ApplyMembersGarbageCollection<
        tensorstore::kvstore::Spec>)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::kvstore::DriverSpecPtr,
    tensorstore::garbage_collection::IndirectPointerGarbageCollection<
        tensorstore::kvstore::DriverSpecPtr>)
