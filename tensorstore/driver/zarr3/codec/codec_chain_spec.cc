// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"

#include <stddef.h>

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/codec_spec.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/driver/zarr3/codec/bytes.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/registry.h"
#include "tensorstore/driver/zarr3/codec/transpose.h"
#include "tensorstore/driver/zarr3/name_configuration_json_binder.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

namespace jb = ::tensorstore::internal_json_binding;

namespace {
struct ZarrCodecJsonBinderImpl {
  static absl::Status FromJson(const ZarrCodecSpec::FromJsonOptions& options,
                               ZarrCodecSpec::Ptr* obj, ::nlohmann::json* j);
  static absl::Status ToJson(const ZarrCodecSpec::ToJsonOptions& options,
                             const ZarrCodecSpec* const* obj,
                             ::nlohmann::json* j);
  absl::Status operator()(std::true_type is_loading,
                          const ZarrCodecSpec::FromJsonOptions& options,
                          ZarrCodecSpec::Ptr* obj, ::nlohmann::json* j) const {
    return FromJson(options, obj, j);
  }

  template <typename T>
  absl::Status operator()(std::false_type is_loading,
                          const ZarrCodecSpec::ToJsonOptions& options, T* obj,
                          ::nlohmann::json* j) const {
    static_assert(
        std::is_convertible_v<decltype(&**obj), const ZarrCodecSpec*>);
    const ZarrCodecSpec* ptr = &**obj;
    return ToJson(options, &ptr, j);
  }
};

constexpr inline ZarrCodecJsonBinderImpl ZarrCodecJsonBinder{};

constexpr auto ZarrCodecJsonBinderImplBase =
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      const auto& registry = GetCodecRegistry();
      if constexpr (is_loading) {
        if (options.constraints && j->is_string()) {
          ::nlohmann::json::object_t j_obj;
          j_obj.emplace("name", std::move(*j));
          *j = std::move(j_obj);
        }
      }
      return jb::Object(NameConfigurationJsonBinder(
          registry.KeyBinder(), registry.RegisteredObjectBinder()))  //
          (is_loading, options, obj, j);
    };

absl::Status ZarrCodecJsonBinderImpl::FromJson(
    const ZarrCodecSpec::FromJsonOptions& options, ZarrCodecSpec::Ptr* obj,
    ::nlohmann::json* j) {
  return ZarrCodecJsonBinderImplBase(std::true_type{}, options, obj, j);
}

absl::Status ZarrCodecJsonBinderImpl::ToJson(
    const ZarrCodecSpec::ToJsonOptions& options,
    const ZarrCodecSpec* const* obj, ::nlohmann::json* j) {
  return ZarrCodecJsonBinderImplBase(std::false_type{}, options, obj, j);
}

constexpr auto ZarrCodecChainSpecJsonBinderImpl = jb::Compose<
    std::vector<ZarrCodecSpec::Ptr>>(
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      if constexpr (is_loading) {
        auto it = j->begin(), end = j->end();
        for (; it != end && (*it)->kind() == ZarrCodecKind::kArrayToArray;
             ++it) {
          obj->array_to_array.push_back(
              internal::static_pointer_cast<const ZarrArrayToArrayCodecSpec>(
                  std::move(*it)));
        }
        if (it != end && (*it)->kind() == ZarrCodecKind::kArrayToBytes) {
          obj->array_to_bytes =
              internal::static_pointer_cast<const ZarrArrayToBytesCodecSpec>(
                  std::move(*it));
          ++it;
        } else if (!options.constraints) {
          return absl::InvalidArgumentError(
              "array -> bytes codec must be specified");
        }
        for (; it != end; ++it) {
          if ((*it)->kind() != ZarrCodecKind::kBytesToBytes) {
            return absl::InvalidArgumentError(tensorstore::StrCat(
                "Expected bytes -> bytes codec, but received: ",
                jb::ToJson(*it, ZarrCodecJsonBinder).value().dump()));
          }
          obj->bytes_to_bytes.push_back(
              internal::static_pointer_cast<const ZarrBytesToBytesCodecSpec>(
                  std::move(*it)));
        }
      } else {
        j->insert(j->end(), obj->array_to_array.begin(),
                  obj->array_to_array.end());
        if (obj->array_to_bytes) {
          j->push_back(obj->array_to_bytes);
        }
        j->insert(j->end(), obj->bytes_to_bytes.begin(),
                  obj->bytes_to_bytes.end());
      }
      return absl::OkStatus();
    },
    jb::Array(ZarrCodecJsonBinder));

}  // namespace

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(ZarrCodecChainSpec,
                                       ZarrCodecChainSpecJsonBinderImpl);

namespace {
Result<ZarrArrayToBytesCodecSpec::Ptr> GetDefaultArrayToBytesCodecSpec(
    const ArrayCodecResolveParameters& decoded) {
  if (internal::IsTrivialDataType(decoded.dtype)) {
    return DefaultBytesCodec();
  }
  return absl::InternalError(tensorstore::StrCat(
      "No default codec defined for data type ", decoded.dtype));
}

absl::Status CodecResolveError(const ZarrCodecSpec& codec_spec,
                               std::string_view message,
                               const absl::Status& status) {
  return tensorstore::MaybeAnnotateStatus(
      status, tensorstore::StrCat(
                  "Error ", message, " through ",
                  jb::ToJson(&codec_spec, ZarrCodecJsonBinder).value().dump()));
}
}  // namespace

size_t ZarrCodecChainSpec::sharding_height() const {
  return array_to_bytes ? array_to_bytes->sharding_height() : 0;
}

absl::Status ZarrCodecChainSpec::GetDecodedChunkLayout(
    const ArrayDataTypeAndShapeInfo& array_info,
    ArrayCodecChunkLayoutInfo& decoded) const {
  // First compute data type and shape info.
  absl::FixedArray<ArrayDataTypeAndShapeInfo, 2> array_infos(
      array_to_array.size());
  const ArrayDataTypeAndShapeInfo* decoded_array_info = &array_info;
  for (size_t i = 0; i < array_to_array.size(); ++i) {
    const auto& codec_spec = *array_to_array[i];
    auto& encoded_array_info = array_infos[i];
    TENSORSTORE_RETURN_IF_ERROR(
        codec_spec.PropagateDataTypeAndShape(*decoded_array_info,
                                             encoded_array_info),
        CodecResolveError(codec_spec, "propagating data type and shape", _));
    decoded_array_info = &encoded_array_info;
  }
  std::optional<ArrayCodecChunkLayoutInfo> temp_info[2];
  const ArrayCodecChunkLayoutInfo* encoded_info;
  if (array_to_bytes) {
    auto& decoded_info = array_infos.empty() ? decoded : temp_info[0].emplace();
    TENSORSTORE_RETURN_IF_ERROR(
        array_to_bytes->GetDecodedChunkLayout(
            array_infos.empty() ? array_info : array_infos.back(),
            decoded_info),
        CodecResolveError(*array_to_bytes, "propagating chunk layout", _));
    encoded_info = &decoded_info;
  } else if (!array_to_array.empty()) {
    encoded_info = &temp_info[0].emplace();
  }
  for (size_t i = array_to_array.size(); i--;) {
    auto& decoded_info =
        i == 0 ? decoded : temp_info[(array_to_array.size() - i) % 2].emplace();
    const auto& codec_spec = *array_to_array[i];
    TENSORSTORE_RETURN_IF_ERROR(
        codec_spec.GetDecodedChunkLayout(
            array_infos[i], *encoded_info,
            i == 0 ? array_info : array_infos[i - 1], decoded_info),
        CodecResolveError(codec_spec, "propagating chunk layout", _));
    encoded_info = &decoded_info;
  }
  return absl::OkStatus();
}

Result<internal::IntrusivePtr<const ZarrCodecChain>>
ZarrCodecChainSpec::Resolve(ArrayCodecResolveParameters&& decoded,
                            BytesCodecResolveParameters& encoded,
                            ZarrCodecChainSpec* resolved_spec) const {
  auto chain = internal::MakeIntrusivePtr<ZarrCodecChain>();

  std::optional<ArrayCodecResolveParameters> temp_array_resolve_params[2];

  chain->array_to_array.reserve(array_to_array.size());
  chain->bytes_to_bytes.reserve(bytes_to_bytes.size());

  if (resolved_spec) {
    assert(resolved_spec != this);
    assert(resolved_spec->array_to_array.empty());
    resolved_spec->array_to_array.reserve(array_to_array.size());
    assert(!resolved_spec->array_to_bytes);
    assert(resolved_spec->bytes_to_bytes.empty());
    resolved_spec->bytes_to_bytes.reserve(bytes_to_bytes.size());
  }

  ArrayCodecResolveParameters* decoded_params = &decoded;

  size_t temp_i = 0;

  const auto resolve_array_to_array =
      [&](const ZarrArrayToArrayCodecSpec& codec_spec) -> absl::Status {
    auto& encoded_params = temp_array_resolve_params[(temp_i++) % 2].emplace();
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto codec,
        codec_spec.Resolve(std::move(*decoded_params), encoded_params,
                           resolved_spec
                               ? &resolved_spec->array_to_array.emplace_back()
                               : nullptr),
        CodecResolveError(codec_spec, "resolving codec spec", _));
    chain->array_to_array.push_back(std::move(codec));
    decoded_params = &encoded_params;
    return absl::OkStatus();
  };
  for (size_t i = 0; i < array_to_array.size(); ++i) {
    TENSORSTORE_RETURN_IF_ERROR(resolve_array_to_array(*array_to_array[i]));
  }

  std::optional<BytesCodecResolveParameters> temp_bytes_resolve_params[2];

  auto* bytes_decoded_params = &temp_bytes_resolve_params[0].emplace();

  ZarrArrayToBytesCodecSpec::Ptr temp_array_to_bytes_codec;
  auto* array_to_bytes_codec_ptr = this->array_to_bytes.get();
  if (!array_to_bytes_codec_ptr) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        temp_array_to_bytes_codec,
        GetDefaultArrayToBytesCodecSpec(*decoded_params));
    array_to_bytes_codec_ptr = temp_array_to_bytes_codec.get();
  }

  DimensionIndex preferred_order[kMaxRank];

  if (DimensionIndex rank = decoded_params->rank;
      decoded_params->inner_order &&
      !array_to_bytes_codec_ptr->SupportsInnerOrder(
          *decoded_params, span<DimensionIndex>(&preferred_order[0], rank))) {
    const auto& existing_inner_order = *decoded_params->inner_order;
    std::vector<DimensionIndex> new_order(rank);
    // Need to select `new_order` such that:
    // preferred_order[i] == inv(new_order)[existing_inner_order[i]]
    //
    // i.e.
    //
    // new_order[preferred_order[i]] = existing_inner_order[i]
    for (DimensionIndex i = 0; i < rank; ++i) {
      new_order[preferred_order[i]] = existing_inner_order[i];
    }
    TENSORSTORE_RETURN_IF_ERROR(
        resolve_array_to_array(*internal::MakeIntrusivePtr<TransposeCodecSpec>(
            TransposeCodecSpec::Options{std::move(new_order)})));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(
      chain->array_to_bytes,
      array_to_bytes_codec_ptr->Resolve(
          std::move(*decoded_params), *bytes_decoded_params,
          resolved_spec ? &resolved_spec->array_to_bytes : nullptr),
      CodecResolveError(*array_to_bytes, "resolving codec spec", _));

  if (chain->array_to_bytes->is_sharding_codec() && !bytes_to_bytes.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Sharding codec %s is not compatible with subsequent bytes -> "
        "bytes codecs %s that apply to the entire shard.  Instead, "
        "bytes -> bytes codecs may be specified as inner codecs that apply "
        "to each sub-chunk individually.",
        jb::ToJson(array_to_bytes_codec_ptr, ZarrCodecJsonBinder)
            .value()
            .dump(),
        jb::ToJson(bytes_to_bytes, jb::Array(ZarrCodecJsonBinder))
            .value()
            .dump()));
  }

  for (size_t i = 0; i < bytes_to_bytes.size(); ++i) {
    auto& encoded_params = temp_bytes_resolve_params[(i + 1) % 2].emplace();
    const auto& codec_spec = *bytes_to_bytes[i];
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto codec,
        codec_spec.Resolve(std::move(*bytes_decoded_params), encoded_params,
                           resolved_spec
                               ? &resolved_spec->bytes_to_bytes.emplace_back()
                               : nullptr),
        CodecResolveError(codec_spec, "resolving codec spec", _));
    bytes_decoded_params = &encoded_params;
    chain->bytes_to_bytes.push_back(std::move(codec));
  }

  encoded = std::move(*bytes_decoded_params);
  return chain;
}

namespace {
template <typename T, typename Binder>
std::string MergeErrorMessage(const T& a, const T& b, const Binder& binder) {
  return absl::StrFormat("Cannot merge zarr codec constraints %s and %s",
                         jb::ToJson(a, binder).value().dump(),
                         jb::ToJson(b, binder).value().dump());
}
std::string MergeErrorMessage(const ZarrCodecSpec& a, const ZarrCodecSpec& b) {
  return MergeErrorMessage(ZarrCodecSpec::Ptr(&a), ZarrCodecSpec::Ptr(&b),
                           ZarrCodecJsonBinder);
}

template <typename T>
void EnsureMutableCodecSpec(internal::IntrusivePtr<const T>& ptr) {
  static_assert(std::is_base_of_v<ZarrCodecSpec, T>);
  assert(ptr);
  if (ptr->use_count() > 1) {
    ptr = internal::static_pointer_cast<const T>(ptr->Clone());
  }
}

absl::Status MergeZarrCodecSpecs(ZarrCodecSpec::Ptr& target,
                                 const ZarrCodecSpec* source, bool strict) {
  if (!source) {
    return absl::OkStatus();
  }
  if (!target) {
    target.reset(source);
    return absl::OkStatus();
  }
  absl::Status status;
  const auto& target_ref = *target;
  const auto& source_ref = *source;
  if (typeid(target_ref) != typeid(source_ref)) {
    status = absl::FailedPreconditionError("");
  } else {
    EnsureMutableCodecSpec(target);
    status = const_cast<ZarrCodecSpec&>(*target).MergeFrom(*source, strict);
  }
  if (status.ok()) return absl::OkStatus();
  return tensorstore::MaybeAnnotateStatus(status,
                                          MergeErrorMessage(*target, *source));
}

template <typename T>
absl::Status MergeZarrCodecSpecs(typename T::Ptr& target, const T* source,
                                 bool strict) {
  static_assert(std::is_base_of_v<ZarrCodecSpec, T>);
  ZarrCodecSpec::Ptr target_base = std::move(target);
  auto status = MergeZarrCodecSpecs(target_base, source, strict);
  target = internal::static_pointer_cast<const T>(std::move(target_base));
  TENSORSTORE_RETURN_IF_ERROR(status);
  return absl::OkStatus();
}

template <typename T>
absl::Status MergeZarrCodecSpecs(std::vector<T>& targets,
                                 const std::vector<T>& sources, bool strict) {
  constexpr bool kIsArrayToArray =
      std::is_same_v<ZarrArrayToArrayCodecSpec::Ptr, T>;
  size_t merge_count = targets.size();
  bool size_mismatch = targets.size() != sources.size();
  if constexpr (kIsArrayToArray) {
    if (!strict) {
      // Allow `sources` or `targets` to contain an extra `TransposeCodecSpec`
      // at the end.
      if (sources.size() == targets.size() + 1 &&
          typeid(*sources.back()) == typeid(TransposeCodecSpec)) {
        targets.push_back(sources.back());
        size_mismatch = false;
      } else if (sources.size() + 1 == targets.size() &&
                 typeid(*targets.back()) == typeid(TransposeCodecSpec)) {
        --merge_count;
        size_mismatch = false;
      }
    }
  }
  if (size_mismatch) {
    return tensorstore::MaybeAnnotateStatus(
        absl::FailedPreconditionError(absl::StrFormat(
            "Mismatch in number of %s codecs (%d vs %d)",
            kIsArrayToArray ? "array -> array" : "bytes -> bytes",
            targets.size(), sources.size())),
        MergeErrorMessage(targets, sources, jb::Array(ZarrCodecJsonBinder)));
  }
  for (size_t i = 0; i < merge_count; ++i) {
    TENSORSTORE_RETURN_IF_ERROR(
        MergeZarrCodecSpecs(targets[i], sources[i].get(), strict));
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ZarrCodecChainSpec::MergeFrom(const ZarrCodecChainSpec& other,
                                           bool strict) {
  if (!strict) {
    size_t self_sharding_height = sharding_height();
    size_t other_sharding_height = other.sharding_height();
    if (self_sharding_height > other_sharding_height &&
        array_to_array.empty() && bytes_to_bytes.empty()) {
      EnsureMutableCodecSpec(array_to_bytes);
      return static_cast<ZarrShardingCodecSpec&>(
                 const_cast<ZarrArrayToBytesCodecSpec&>(*array_to_bytes))
          .MergeSubChunkCodecsFrom(other, strict);
    }
    if (self_sharding_height < other_sharding_height &&
        other.array_to_array.empty() && other.bytes_to_bytes.empty()) {
      auto new_array_to_bytes_codec =
          internal::static_pointer_cast<const ZarrShardingCodecSpec>(
              other.array_to_bytes->Clone());
      TENSORSTORE_RETURN_IF_ERROR(
          const_cast<ZarrShardingCodecSpec&>(*new_array_to_bytes_codec)
              .MergeSubChunkCodecsFrom(*this, strict));
      array_to_array.clear();
      bytes_to_bytes.clear();
      array_to_bytes = std::move(new_array_to_bytes_codec);
      return absl::OkStatus();
    }
  }
  TENSORSTORE_RETURN_IF_ERROR(
      MergeZarrCodecSpecs(array_to_array, other.array_to_array, strict));
  TENSORSTORE_RETURN_IF_ERROR(
      MergeZarrCodecSpecs(array_to_bytes, other.array_to_bytes.get(), strict));
  TENSORSTORE_RETURN_IF_ERROR(
      MergeZarrCodecSpecs(bytes_to_bytes, other.bytes_to_bytes, strict));
  return absl::OkStatus();
}

absl::Status MergeZarrCodecSpecs(
    std::optional<ZarrCodecChainSpec>& target,
    const std::optional<ZarrCodecChainSpec>& source, bool strict) {
  if (!target) {
    if (source) {
      target = *source;
    }
    return absl::OkStatus();
  }
  if (!source) {
    return absl::OkStatus();
  }
  return target->MergeFrom(*source, strict);
}

bool ZarrShardingCodecSpec::SupportsInnerOrder(
    const ArrayCodecResolveParameters& decoded,
    span<DimensionIndex> preferred_inner_order) const {
  return true;
}

size_t ZarrShardingCodecSpec::sharding_height() const {
  auto* sub_chunk_codecs = this->GetSubChunkCodecs();
  return 1 + (sub_chunk_codecs ? sub_chunk_codecs->sharding_height() : 0);
}

CodecSpec TensorStoreCodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<TensorStoreCodecSpec>(*this);
}

absl::Status TensorStoreCodecSpec::DoMergeFrom(
    const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(TensorStoreCodecSpec)) {
    return absl::InvalidArgumentError("");
  }
  auto& other = static_cast<const TensorStoreCodecSpec&>(other_base);
  return MergeZarrCodecSpecs(codecs, other.codecs, /*strict=*/false);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    TensorStoreCodecSpec,
    jb::Sequence(  //
        jb::Member("codecs",
                   jb::Projection<&TensorStoreCodecSpec::codecs>(jb::Optional(
                       ZarrCodecChainJsonBinder</*Constraints=*/true>)))  //
        ))

namespace {
const internal::CodecSpecRegistration<TensorStoreCodecSpec>
    encoding_registration;

}  // namespace

}  // namespace internal_zarr3
namespace internal {
void CacheKeyEncoder<internal_zarr3::ZarrCodecChainSpec>::Encode(
    std::string* out, const internal_zarr3::ZarrCodecChainSpec& value) {
  internal::EncodeCacheKey(out, value.ToJson().value().dump());
}
}  // namespace internal
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrCodecChainSpec,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_zarr3::ZarrCodecChainSpec>())
