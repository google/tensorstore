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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_IDENTIFIER_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_IDENTIFIER_H_

#include <iosfwd>
#include <string>
#include <string_view>

#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"

namespace tensorstore {
namespace internal_ocdbt {

// Identifies an existing B+tree node for the purpose of lease assignment.
//
// A node is specified by a key range and height, which uniquely identifies a
// node within a given B+tree, and is hopefully somewhat stable even across
// multiple generations.
struct BtreeNodeIdentifier {
  BtreeNodeHeight height;
  KeyRange range;

  friend bool operator==(const BtreeNodeIdentifier& a,
                         const BtreeNodeIdentifier& b) {
    return a.height == b.height && a.range == b.range;
  }

  // Special identifier used to query a lease on the root node and manifest.
  static BtreeNodeIdentifier Root() {
    BtreeNodeIdentifier identifier;
    identifier.height = 0;
    return identifier;
  }

  // Computes a lease key for this node identifier.
  //
  // The lease key is a cryptographic hash of:
  //
  // - node height, unless the key range is the full range.
  // - key range
  // - database_identifier
  //
  // The database identifier (i.e. hash of the storage location) is included
  // because the same coordinator server may be used with multiple databases.
  //
  // As a special case, if the key range is the full range (i.e. the root node),
  // then the height does not affect the lease key (a height of 0 is always used
  // in computing the key).  This same lease key is also used for the manifest,
  // to ensure that the same cooperator owns both the root node and the
  // manifest.
  std::string GetKey(std::string_view database_identifier) const;

  friend std::ostream& operator<<(std::ostream& os,
                                  const BtreeNodeIdentifier& x);
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_BTREE_NODE_IDENTIFIER_H_
