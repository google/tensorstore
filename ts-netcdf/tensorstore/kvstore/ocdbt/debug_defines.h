// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DEBUG_DEFINES_H_
#define TENSORSTORE_KVSTORE_OCDBT_DEBUG_DEFINES_H_

// Use as a logging condition via:
//  ABSL_LOG_IF(INFO, TENSORSTORE_INTERNAL_OCDBT_DEBUG)
//
// To enable debug checks, specify:
//   bazel build --//tensorstore/kvstore/ocdbt:debug
#ifndef TENSORSTORE_INTERNAL_OCDBT_DEBUG
#define TENSORSTORE_INTERNAL_OCDBT_DEBUG 0
#endif

#endif  // TENSORSTORE_KVSTORE_OCDBT_DEBUG_LOG_H_
