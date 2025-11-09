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

#ifndef TENSORSTORE_KVSTORE_URL_REGISTRY_H_
#define TENSORSTORE_KVSTORE_URL_REGISTRY_H_

#include <string_view>

#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvstore {

using UrlSchemeRootHandler = Result<kvstore::Spec> (*)(std::string_view url);
using UrlSchemeAdapterHandler = Result<kvstore::Spec> (*)(std::string_view url,
                                                          kvstore::Spec base);

/// Registers a kvstore URL handler for the specified `scheme`.
///
/// This is intended to be defined as a global/namespace-scope constant.
class UrlSchemeRegistration {
 public:
  UrlSchemeRegistration(std::string_view scheme, UrlSchemeRootHandler handler);
  UrlSchemeRegistration(std::string_view scheme,
                        UrlSchemeAdapterHandler handler);
};

Result<kvstore::Spec> GetRootSpecFromUrl(std::string_view url);
Result<kvstore::Spec> GetAdapterSpecFromUrl(std::string_view url,
                                            kvstore::Spec &&base);

}  // namespace internal_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_URL_REGISTRY_H_
