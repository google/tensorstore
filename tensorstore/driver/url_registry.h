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

#ifndef TENSORSTORE_DRIVER_URL_REGISTRY_H_
#define TENSORSTORE_DRIVER_URL_REGISTRY_H_

#include <string_view>

#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal {

using UrlSchemeRootHandler =
    Result<TransformedDriverSpec> (*)(std::string_view url);
using UrlSchemeKvStoreAdapterHandler = Result<TransformedDriverSpec> (*)(
    std::string_view url, kvstore::Spec&& base);
using UrlSchemeAdapterHandler = Result<TransformedDriverSpec> (*)(
    std::string_view url, TransformedDriverSpec&& base);

/// Registers a TensorStore URL handler for the specified `scheme`.
///
/// This is intended to be defined as a global/namespace-scope constant.
class UrlSchemeRegistration {
 public:
  UrlSchemeRegistration(std::string_view scheme,
                        UrlSchemeKvStoreAdapterHandler handler);
  UrlSchemeRegistration(std::string_view scheme, UrlSchemeRootHandler handler);
  UrlSchemeRegistration(std::string_view scheme,
                        UrlSchemeAdapterHandler handler);
};

// The following internal helper functions are intended for use by the
// higher-level `GetTransformedDriverSpecFromUrl` function defined in
// `tensorstore/driver/driver_spec.h`.

// Attempts to parse `url`, which is assumed to be a *single* URL
// pipeline component, into a TensorStore driver spec.
//
// Returns an error if `url` does not specify a valid root TensorStore driver
// scheme.
Result<TransformedDriverSpec> GetTransformedDriverRootSpecFromUrl(
    std::string_view url);

// Attempts to parse `url`, which is assumed to be a *single* URL
// pipeline component, into a TensorStore driver spec that adapts the specified
// `base` driver spec.
//
// Returns an error if `url` does not specify a valid TensorStore
// adapter driver scheme.
Result<TransformedDriverSpec> GetTransformedDriverAdapterSpecFromUrl(
    std::string_view url, TransformedDriverSpec&& base);

// Attempts to parse `url`, which is assumed to be a *single* URL
// pipeline component, into a TensorStore driver spec that adaps the specified
// `base` kvstore.
//
// Returns an error if `url` does not specify a valid TensorStore
// KvStore adapter driver scheme.
Result<TransformedDriverSpec> GetTransformedDriverKvStoreAdapterSpecFromUrl(
    std::string_view url, kvstore::Spec&& base);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_URL_REGISTRY_H_
