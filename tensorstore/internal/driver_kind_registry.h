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

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DRIVER_KIND_REGISTRY_H_
