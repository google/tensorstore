// Copyright 2021 The TensorStore Authors
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

namespace tensorstore {
namespace internal_kvstore {

using UrlSchemeHandler = Result<kvstore::Spec> (*)(std::string_view url);

/// Registers a kvstore URL handler for the specified `scheme`.
///
/// This handler is used for URLs that start with `scheme + "://"`.
///
/// This is intended to be defined as a global/namespace-scope constant.
class UrlSchemeRegistration {
 public:
  UrlSchemeRegistration(std::string_view scheme, UrlSchemeHandler handler);
};

}  // namespace internal_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_URL_REGISTRY_H_
