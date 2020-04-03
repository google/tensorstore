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

#include "tensorstore/internal/file_io_concurrency_resource.h"

#include <algorithm>
#include <thread>  // NOLINT

#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/concurrency_resource_provider.h"

namespace tensorstore {
namespace internal {
namespace {

struct FileIoConcurrencyResourceTraits
    : public ConcurrencyResourceTraits,
      public ContextResourceTraits<FileIoConcurrencyResource> {
  // TODO(jbms): use beter method of picking concurrency limit
  FileIoConcurrencyResourceTraits()
      : ConcurrencyResourceTraits(
            std::max(size_t(4), size_t(std::thread::hardware_concurrency()))) {}
};

const ContextResourceRegistration<FileIoConcurrencyResourceTraits> registration;

}  // namespace
}  // namespace internal
}  // namespace tensorstore
