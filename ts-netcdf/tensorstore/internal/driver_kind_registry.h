// Copyright 2025 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_DRIVER_KIND_REGISTRY_H_
#define TENSORSTORE_INTERNAL_DRIVER_KIND_REGISTRY_H_

#include <iosfwd>
#include <optional>
#include <string_view>

#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

// Indicates the kind of a registered KvStore or TensorStore driver.
//
// The registry of driver kinds enables better error messages when a TensorStore
// driver is accidentally specified when creating a KvStore spec, and enables
// format auto-detection to be used implicitly when a KvStore driver is
// specified when creating a TensorStore spec.
enum class DriverKind {
  kKvStore,
  kTensorStore,
};

// Returns the name of a driver kind for use in error messages.
std::string_view DriverKindToStringView(DriverKind x);

template <typename Sink>
void AbslStringify(Sink& sink, DriverKind x) {
  return sink.Append(DriverKindToStringView(x));
}
std::ostream& operator<<(std::ostream& os, DriverKind x);

// Registers a driver identifier.  These are called automatically by the kvstore
// and TensorStore driver registration classes.
void RegisterDriverKind(std::string_view id, DriverKind kind);
void RegisterDriverKind(std::string_view id, DriverKind kind,
                        tensorstore::span<const std::string_view> aliases);

// Returns the kind of a registered driver id, or `std::nullopt` if not
// registered.
std::optional<DriverKind> GetDriverKind(std::string_view id);

// Indicates the kind of a registered URL scheme in the tensorstore library.
//
// The kvstore URL scheme handlers are registered in
// tensorstore/kvstore/registry.h and the TensorStore URL scheme
// handlers are registered in tensorstore/driver/registry.h.
enum class UrlSchemeKind {
  // KvStore driver that directly accesses storage, e.g. "file"
  //
  // This must be first in a URL pipeline.
  kKvStoreRoot,
  // KvStore driver that adapts a base kvstore, e.g. "zip" or "ocdbt".
  //
  // This must immediately follow a `kKvStoreBase` component in a URL pipeline.
  kKvStoreAdapter,
  // TensorStore driver that directly accesses storage.
  //
  // This must be first in a URL pipeline.
  kTensorStoreRoot,
  // TensorStore driver that adapts a base kvstore, e.g. "zarr3"
  //
  // This must immediately follow a `kKvStoreBase` component in a URL pipeline.
  kTensorStoreKvStoreAdapter,
  // TensorStore driver that adapts a base TensorStore, e.g. "cast"
  //
  // This must immediately follow a `kTensorStoreBase` or
  // `kTensorStoreKvStoreAdapter` component in a URL pipeline.
  kTensorStoreAdapter,
};

// Returns the name of a URL scheme kind for use in error messages.
std::string_view UrlSchemeKindToStringView(UrlSchemeKind x);

template <typename Sink>
void AbslStringify(Sink& sink, UrlSchemeKind x) {
  return sink.Append(UrlSchemeKindToStringView(x));
}
std::ostream& operator<<(std::ostream& os, UrlSchemeKind x);

// Registers a URL scheme.  This is called automatically by
// `internal_kvstore::UrlSchemeRegistration` and
// `internal::UrlSchemeRegistration`.
void RegisterUrlSchemeKind(std::string_view scheme, UrlSchemeKind scheme_kind);

// Returns the kind of a registered URL scheme, or `std::nullopt` if not
// registered.
std::optional<UrlSchemeKind> GetUrlSchemeKind(std::string_view scheme);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DRIVER_KIND_REGISTRY_H_
