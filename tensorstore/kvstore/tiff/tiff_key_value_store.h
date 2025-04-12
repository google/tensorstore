// tensorstore/kvstore/tiff/tiff_key_value_store.h
//
// Tensorstore driver for readonly tiled TIFF files.

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_

#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/kvstore.h"

namespace tensorstore {
namespace kvstore {
namespace tiff_kvstore {

/// Opens a TIFF-backed KeyValueStore treating each tile as a separate key.
/// @param base_kvstore Base kvstore (e.g., local file, GCS, HTTP-backed).
/// @returns DriverPtr wrapping the TIFF store.
DriverPtr GetTiffKeyValueStore(DriverPtr base_kvstore);

}  // namespace tiff_kvstore
}  // namespace kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_KEY_VALUE_STORE_H_
