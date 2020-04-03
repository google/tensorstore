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

#include "tensorstore/kvstore/key_value_store.h"

#include <functional>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include <nlohmann/json.hpp>
#include "tensorstore/context.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/collecting_sender.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/sync_flow_sender.h"
#include "tensorstore/util/to_string.h"

namespace tensorstore {

KeyValueStoreSpec::~KeyValueStoreSpec() = default;
KeyValueStoreSpec::Bound::~Bound() = default;

Result<KeyValueStoreSpec::Ptr> KeyValueStore::spec(
    const internal::ContextSpecBuilder& context_builder) const {
  return absl::UnimplementedError(
      "KeyValueStore does not support JSON representation");
}

Result<KeyValueStoreSpec::BoundPtr> KeyValueStore::GetBoundSpec() const {
  return absl::UnimplementedError(
      "KeyValueStore does not support JSON representation");
}

void KeyValueStore::EncodeCacheKey(std::string* out) const {
  internal::EncodeCacheKey(out, reinterpret_cast<std::uintptr_t>(this));
}

Result<KeyValueStoreSpec::BoundPtr> KeyValueStoreSpec::Bind(
    const Context& context) const {
  return absl::UnimplementedError("Driver not registered");
}

Result<KeyValueStoreSpec::Ptr> KeyValueStoreSpec::Convert(
    const RequestOptions& options) const {
  return absl::UnimplementedError("Driver not registered");
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    KeyValueStoreSpec::Ptr,
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      namespace jb = tensorstore::internal::json_binding;
      auto& registry = internal::GetKeyValueStoreDriverRegistry();
      return jb::Object(
          jb::Member("driver", registry.KeyBinder()),
          jb::Member("context",
                     jb::Projection(
                         [](const KeyValueStoreSpec::Ptr& p) -> decltype(auto) {
                           return (p->context_spec_);
                         },
                         jb::DefaultInitializedValue<
                             /*DisallowIncludeDefaults=*/true>())),
          registry.RegisteredObjectBinder())(is_loading, options, obj, j);
    })

Future<KeyValueStore::Ptr> KeyValueStoreSpec::Open(
    const Context& context, const OpenOptions& options) const {
  return ChainResult(
      Convert(options), [&](const Ptr& spec) { return spec->Bind(context); },
      [](const BoundPtr& bound_spec) { return bound_spec->Open(); });
}

std::ostream& operator<<(std::ostream& os, const KeyValueStore::ReadResult& x) {
  return os << "{value=" << (x.value ? QuoteString(*x.value) : "nullopt")
            << ", generation=" << x.generation << "}";
}

namespace internal {

KeyValueStoreDriverRegistry& GetKeyValueStoreDriverRegistry() {
  static internal::NoDestructor<KeyValueStoreDriverRegistry> registry;
  return *registry;
}

}  // namespace internal

KeyValueStore::~KeyValueStore() = default;

Future<KeyValueStore::Ptr> KeyValueStore::Open(
    const Context& context, ::nlohmann::json j,
    const KeyValueStoreSpec::OpenOptions& options) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, Spec::Ptr::FromJson(std::move(j)));
  return spec->Open(context, options);
}

AnyFlowSender<Status, KeyValueStore::Key> KeyValueStore::List(
    ListOptions options) {
  struct ListSender {
    Ptr self;
    ListOptions options;
    void submit(AnyFlowReceiver<Status, Key> receiver) {
      self->ListImpl(options, std::move(receiver));
    }
  };
  return ListSender{Ptr(this), std::move(options)};
}

Future<std::vector<KeyValueStore::Key>> ListFuture(
    KeyValueStore* store, KeyValueStore::ListOptions options) {
  return tensorstore::MakeSenderFuture<std::vector<KeyValueStore::Key>>(
      tensorstore::internal::MakeCollectingSender<
          std::vector<KeyValueStore::Key>>(
          tensorstore::MakeSyncFlowSender(store->List(options))));
}

}  // namespace tensorstore
