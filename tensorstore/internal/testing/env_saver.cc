// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/internal/testing/env_saver.h"

#include "tensorstore/internal/env.h"

namespace tensorstore {
namespace internal_testing {

EnvSaver::EnvSaver() : saved_env_(internal::GetEnvironmentMap()) {}

EnvSaver::~EnvSaver() {
  // NOTE: On windows the native environment variables are wide strings, and
  // conversion to/from absl::string_view may fail if the strings are not
  // representable in the filesystem encoding. If this matters in the future,
  // we could condition on windows and store/set std::wstring variables using
  // GetEnvironmentStringsW() / ::_wputenv_s().
  auto current_env = internal::GetEnvironmentMap();
  for (const auto& [name, value] : current_env) {
    if (!saved_env_.contains(name)) {
      internal::UnsetEnv(name.c_str());
    }
  }
  for (const auto& [name, value] : saved_env_) {
    if (auto it = current_env.find(name);
        it == current_env.end() || it->second != value) {
      internal::SetEnv(name.c_str(), value.c_str());
    }
  }
}

}  // namespace internal_testing
}  // namespace tensorstore
