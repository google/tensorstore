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

#ifndef TENSORSTORE_CODEC_SPEC_REGISTRY_H_
#define TENSORSTORE_CODEC_SPEC_REGISTRY_H_

/// \file
///
/// Interface for registering an `internal::CodecDriverSpec`.

#include "tensorstore/codec_spec.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/json_serialization_options.h"

namespace tensorstore {
namespace internal {

using CodecSpecRegistry =
    JsonRegistry<internal::CodecDriverSpec, JsonSerializationOptions,
                 JsonSerializationOptions,
                 IntrusivePtr<const internal::CodecDriverSpec>>;

/// Returns the global encoding registry.
CodecSpecRegistry& GetCodecSpecRegistry();

/// Registers an encoding specification.
///
/// Example usage:
///
///     class MyCodec : public internal::CodecDriverSpec {
///      public:
///       // ...
///       constexpr static char id[] = "my_driver";
///       absl::Status MergeFrom(const internal::CodecDriverSpec &other)
///       override; bool EqualTo(const internal::CodecDriverSpec &other); Ptr
///       Clone() const override {
///         return Ptr(new MyCodec(*this));
///       }
///       TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(
///           MyCodec,
///           FromJsonOptions, ToJsonOptions,
///           ::nlohmann::json::object_t)
///     };
///
///     TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
///         MyCodec,
///         jb::Sequence(...))
///
///     const internal::CodecSpecRegistration<MyCodec> registration;
///
/// \tparam Derived The derived codec type, must be a subclass of
///     `internal::CodecDriverSpec` and must define an `id` member and a
///     `default_json_binder` member.  The `default_json_binder` member is
///     typically defined using the `TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER`
///     macro.
template <typename Derived>
class CodecSpecRegistration {
  static_assert(std::is_base_of_v<internal::CodecDriverSpec, Derived>);

 public:
  CodecSpecRegistration() {
    GetCodecSpecRegistry().Register<Derived>(Derived::id,
                                             Derived::default_json_binder);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_CODEC_SPEC_REGISTRY_H_
