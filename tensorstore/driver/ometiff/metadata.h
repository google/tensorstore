#ifndef TENSORSTORE_DRIVER_OMETIFF_METADATA_H_
#define TENSORSTORE_DRIVER_OMETIFF_METADATA_H_

#include <string>
#include <map>
#include <tuple>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ometiff {
class OmeTiffMetadata {
    public: 
        DimensionIndex rank = dynamic_rank;
        std::vector<Index> shape;
        std::vector<Index> chunk_shape; 
        DataType dtype;
        StridedLayout<> chunk_layout;
        bool tiled;
        short dim_order;
        std::map<std::tuple<size_t, size_t, size_t>, size_t> ifd_lookup_table;
          /// Contains all additional attributes, excluding attributes parsed into the
  /// data members above.
        ::nlohmann::json::object_t extra_attributes;
        TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OmeTiffMetadata,
                                                internal_json_binding::NoOptions,
                                                tensorstore::IncludeDefaults)

        std::string GetCompatibilityKey() const;
        size_t GetIfdIndex(size_t, size_t, size_t) const;
};


/// Representation of partial metadata/metadata constraints specified as the
/// "metadata" member in the DriverSpec.
class OmeTiffMetadataConstraints {
 public:

  /// Length of `shape`, `axes` and `chunk_shape` if any are specified.  If none
  /// are specified, equal to `dynamic_rank`.
  DimensionIndex rank = dynamic_rank;

  /// Specifies the current shape of the full volume.
  std::optional<std::vector<Index>> shape;


  /// Specifies the chunk size (corresponding to the `"blockSize"` attribute)
  /// and the in-memory layout of a full chunk (always C order).
  std::optional<std::vector<Index>> chunk_shape;
  std::optional<DataType> dtype;
  std::optional<short> dim_order;
  /// Contains all additional attributes, excluding attributes parsed into the
  /// data members above.
  ::nlohmann::json::object_t extra_attributes;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OmeTiffMetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

/// Validates that `metadata` is consistent with `constraints`.
absl::Status ValidateMetadata(const OmeTiffMetadata& metadata,
                              const OmeTiffMetadataConstraints& constraints);

/// Sets chunk layout constraints implied by `rank` and `chunk_shape`.
absl::Status SetChunkLayoutFromMetadata(
    DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
    ChunkLayout& chunk_layout);

/// Returns the combined domain from `metadata_constraints` and `schema`.
///
/// If the domain is unspecified, returns a null domain.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<IndexDomain<>> GetEffectiveDomain(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the combined chunk layout from `metadata_constraints` and `schema`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<ChunkLayout> GetEffectiveChunkLayout(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema);


/// Decodes a chunk.
///
/// The layout of the returned array is only valid as long as `metadata`.
Result<SharedArrayView<const void>> DecodeChunk(const OmeTiffMetadata& metadata,
                                                absl::Cord buffer);



/// Encodes a chunk.
Result<absl::Cord> EncodeChunk(span<const Index> chunk_indices,
                               const OmeTiffMetadata& metadata,
                               ArrayView<const void> array);

/// Converts `metadata_constraints` to a full metadata object.
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const OmeTiffMetadata>> GetNewMetadata(
    const OmeTiffMetadataConstraints& metadata_constraints, const Schema& schema);

/// Validates that `schema` is compatible with `metadata`.
absl::Status ValidateMetadataSchema(const OmeTiffMetadata& metadata,
                                    const Schema& schema);

/// Validates that `dtype` is supported by OMETiff.
///
/// \dchecks `dtype.valid()`
absl::Status ValidateDataType(DataType dtype);
}  // namespace internal_ometiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_ometiff::OmeTiffMetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_ometiff::OmeTiffMetadataConstraints)
#endif  // TENSORSTORE_DRIVER_OMETIFF_METADATA_H_
