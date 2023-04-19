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

#ifndef TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COORDINATOR_SERVER_H_
#define TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COORDINATOR_SERVER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/ocdbt/distributed/rpc_security.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace ocdbt {

/// Interface for starting a coordinator server.
///
/// .. warning::
///
///    Encryption and authentication are not currently supported, which makes
///    this unsuitable for use on untrusted networks.
///
/// Example:
///
///     tensorstore::ocdbt::CoordinatorServer::Options options;
///     TENSORSTORE_ASSIGN_OR_RETURN(
///       auto coordinator,
///       tensorstore::ocdbt::CoordinatorServer::Start(options));
///
///     int port = coordinator.port();
///     // If not fixed, communicate `port` to all cooperating processes.
///
///     // In each cooperating process, create `Context` that specifies this
///     // coordinator.
///     TENSORSTORE_ASSIGN_OR_RETURN(
///       auto context,
///       tensorstore::Context::FromJson({
///         {"ocdbt_coordinator", {
///           {"address", tensorstore::StrCat(server_hostname, ":", port)}
///         }}
///       }));
///
///     // Open kvstore in each cooperating process.
///     TENSORSTORE_ASSIGN_OR_RETURN(
///       auto kvs,
///       tensorstore::kvstore::Open({
///         {"driver", "ocdbt"},
///         {"base", "gs://bucket/path"},
///       }, context).result());
class CoordinatorServer {
 public:
  struct Spec {
    TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, JsonSerializationOptions,
                                            JsonSerializationOptions);

    /// Security method to use.
    internal_ocdbt::RpcSecurityMethod::Ptr security;

    /// List of addresses to bind.
    ///
    /// Each bind address must include a port number, but the port number may be
    /// `0` to indicate that any available port number should be chosen
    /// automatically.  The chosen port number can then be queried by calling
    /// `CoordinatorServer::port` or `CoordinatorServer::ports`.
    ///
    /// If none are specified, binds to `[::]:0`.
    std::vector<std::string> bind_addresses;
  };
  using Clock = std::function<absl::Time()>;
  struct Options {
    Spec spec;

    /// Specifies a custom clock, intended for testing.
    Clock clock;
  };

  /// Constructs a null handle to a coordinator server.
  CoordinatorServer();

  /// Starts the coordinator server.
  static Result<CoordinatorServer> Start(Options options);

  CoordinatorServer(CoordinatorServer&&);
  CoordinatorServer& operator=(CoordinatorServer&&);

  /// Stops the coordinator server, if this is not a null handle.
  ~CoordinatorServer();

  /// Returns the port number corresponding to the first bind address, or the
  /// default bind address of `[::]` if no bind address was specified.
  int port() const;

  /// Returns the list of port numbers corresponding to the bind addresses.
  span<const int> ports() const;

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_DISTRIBUTED_COORDINATOR_SERVER_H_
