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

#ifndef TENSORSTORE_KVSTORE_TSGRPC_KVSTORE_SERVER_H_
#define TENSORSTORE_KVSTORE_TSGRPC_KVSTORE_SERVER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace grpc_kvstore {

/// Interface for starting a grpc kvstore server.
///
/// .. warning::
///
///    Encryption and authentication are not currently supported, which makes
///    this unsuitable for use on untrusted networks.
///
/// Example:
///
///     tensorstore::grpc_kvstore::KvStoreServer::Spec spec;
///     TENSORSTORE_ASSIGN_OR_RETURN(
///       auto server,
///       tensorstore::grpc_kvstore::KvStoreServer::Start(spec));
///
class KvStoreServer {
 public:
  struct Spec {
    TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(Spec, JsonSerializationOptions,
                                            JsonSerializationOptions);

    /// List of addresses to bind.
    ///
    /// Each bind address must include a port number, but the port number may be
    /// `0` to indicate that any available port number should be chosen
    /// automatically.  The chosen port number can then be queried by calling
    /// `CoordinatorServer::port` or `CoordinatorServer::ports`.
    ///
    /// If none are specified, binds to `[::]:0`.
    std::vector<std::string> bind_addresses;

    /// Underlying kvstore used by the server.
    kvstore::Spec base;
  };

  /// Starts the kvstore server server.
  static Result<KvStoreServer> Start(Spec spec,
                                     Context context = Context::Default());

  /// Constructs a null handle to a coordinator server.
  KvStoreServer();

  KvStoreServer(KvStoreServer&&);
  KvStoreServer& operator=(KvStoreServer&&);

  /// Stops the coordinator server, if this is not a null handle.
  ~KvStoreServer();

  /// Waits for the server to shutdown.
  void Wait();

  /// Returns the port number corresponding to the first bind address, or the
  /// default bind address of `[::]` if no bind address was specified.
  int port() const;

  /// Returns the list of port numbers corresponding to the bind addresses.
  tensorstore::span<const int> ports() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace grpc_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TSGRPC_KVSTORE_SERVER_H_
