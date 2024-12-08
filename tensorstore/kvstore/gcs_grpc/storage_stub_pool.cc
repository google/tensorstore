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

#include "tensorstore/kvstore/gcs_grpc/storage_stub_pool.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpc/grpc.h"
#include "grpcpp/create_channel.h"  // third_party
#include "grpcpp/security/credentials.h"  // third_party
#include "grpcpp/support/channel_arguments.h"  // third_party
#include "grpcpp/support/client_interceptor.h"  // third_party
#include "grpcpp/support/interceptor.h"  // third_party
#include "grpcpp/support/status.h"  // third_party
#include "google/protobuf/message.h"
#include "tensorstore/internal/env.h"
#include "tensorstore/internal/grpc/utils.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/proto/proto_util.h"

// Protos
#include "google/protobuf/empty.pb.h"
#include "google/storage/v2/storage.grpc.pb.h"
#include "google/storage/v2/storage.pb.h"

ABSL_FLAG(std::optional<uint32_t>, tensorstore_gcs_grpc_channels, std::nullopt,
          "Default (and maximum) channels to use in gcs_grpc driver. "
          "Overrides TENSORSTORE_GCS_GRPC_CHANNELS.");

using ::grpc::experimental::ClientInterceptorFactoryInterface;
using ::grpc::experimental::ClientRpcInfo;
using ::grpc::experimental::InterceptionHookPoints;
using ::grpc::experimental::Interceptor;
using ::grpc::experimental::InterceptorBatchMethods;
using ::tensorstore::internal::GetFlagOrEnvValue;
using ::tensorstore::internal::GrpcStatusToAbslStatus;

using ::google::storage::v2::DeleteObjectRequest;
using ::google::storage::v2::ReadObjectRequest;
using ::google::storage::v2::WriteObjectRequest;

namespace tensorstore {
namespace internal_gcs_grpc {
namespace {

ABSL_CONST_INIT absl::Mutex global_mu(absl::kConstInit);

ABSL_CONST_INIT internal_log::VerboseFlag gcs_grpc_logging("gcs_grpc");

enum class Method {
  kReadObject = 0,
  kWriteObject,
  kDeleteObject,
  kListObjects
};

static const std::string_view kNames[] = {"ReadObject", "WriteObject",
                                          "DeleteObject", "ListObjects"};

// Implements detailed logging for gRPC calls to GCS.
class LoggingInterceptor : public Interceptor {
 public:
  LoggingInterceptor(Method method, bool verbose)
      : method_(method), verbose_(verbose) {}

  std::string_view method_name() const {
    return kNames[static_cast<size_t>(method_)];
  }

  void Intercept(InterceptorBatchMethods* methods) override {
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::PRE_SEND_MESSAGE)) {
      auto* message =
          static_cast<const google::protobuf::Message*>(methods->GetSendMessage());
      bool is_first_message = !started_;
      if (!started_) {
        started_ = true;
        switch (method_) {
          case Method::kReadObject:
            object_name_ =
                static_cast<const ReadObjectRequest*>(message)->object();
            break;
          case Method::kWriteObject:
            object_name_ = static_cast<const WriteObjectRequest*>(message)
                               ->write_object_spec()
                               .resource()
                               .name();
            break;
          case Method::kDeleteObject:
            object_name_ =
                static_cast<const DeleteObjectRequest*>(message)->object();
            break;
          case Method::kListObjects:
            break;
        }
      }
      if (verbose_) {
        ABSL_LOG(INFO) << "Begin: " << method_name() << " " << object_name_
                       << " " << ConciseDebugString(*message);
      } else if (is_first_message) {
        ABSL_LOG(INFO) << "Begin: " << method_name() << " " << object_name_;
      }
    }
    if (verbose_ && methods->QueryInterceptionHookPoint(
                        InterceptionHookPoints::POST_RECV_MESSAGE)) {
      ABSL_LOG(INFO) << method_name() << " "
                     << ConciseDebugString(*static_cast<const google::protobuf::Message*>(
                            methods->GetRecvMessage()));
    }
    if (methods->QueryInterceptionHookPoint(
            InterceptionHookPoints::POST_RECV_STATUS)) {
      if (auto* status = methods->GetRecvStatus();
          status != nullptr && !status->ok()) {
        ABSL_LOG(INFO) << "Error: " << method_name() << " " << object_name_
                       << " " << GrpcStatusToAbslStatus(*status);
      } else {
        ABSL_LOG(INFO) << "End: " << method_name() << " " << object_name_;
      }
    }
    methods->Proceed();
  }

 private:
  const Method method_;
  const bool verbose_;
  bool started_ = false;
  std::string object_name_;
};

// Creates a logging interceptor.
class LoggingInterceptorFactory : public ClientInterceptorFactoryInterface {
 public:
  Interceptor* CreateClientInterceptor(ClientRpcInfo* info) override {
    if (!gcs_grpc_logging.Level(1)) {
      return nullptr;
    }
    std::string_view method = info->method();
    bool verbose = gcs_grpc_logging.Level(2);
    if (absl::EndsWith(method, "Storage/ReadObject")) {
      return new LoggingInterceptor(Method::kReadObject, verbose);
    } else if (absl::EndsWith(method, "Storage/WriteObject")) {
      return new LoggingInterceptor(Method::kWriteObject, verbose);
    } else if (absl::EndsWith(method, "Storage/DeleteObject")) {
      return new LoggingInterceptor(Method::kDeleteObject, verbose);
    } else if (absl::EndsWith(method, "Storage/ListObjects")) {
      return new LoggingInterceptor(Method::kListObjects, verbose);
    }
    return nullptr;
  }
};

bool IsDirectPathAddress(std::string_view address) {
  return absl::StartsWith(address, "google:///") ||
         absl::StartsWith(address, "google-c2p:///") ||
         absl::StartsWith(address, "google-c2p-experimental:///");
}

/// Returns the number of channels to use.
/// See also the channel construction in googleapis/google-cloud-cpp repository:
/// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/grpc_client.cc#L188
uint32_t ChannelsForAddress(std::string_view address, uint32_t num_channels) {
  // If num_channels was explicitly requested, use that.
  if (num_channels != 0) {
    return num_channels;
  }

  if (IsDirectPathAddress(address)) {
    // google-c2p are direct-path addresses; multiple channels are handled
    // internally in gRPC.
    return 1;
  }

  // Use the flag --tensorstore_gcs_grpc_channels if present.
  // Use the environment variable TENSORSTORE_GCS_GRPC_CHANNELS if present.
  auto opt = GetFlagOrEnvValue(FLAGS_tensorstore_gcs_grpc_channels,
                               "TENSORSTORE_GCS_GRPC_CHANNELS");
  if (opt && *opt > 0) {
    return *opt;
  }

  // Otherwise multiplex over a multiple channels.
  return std::max(4u, std::thread::hardware_concurrency());
}

// Create a gRPC channel. See google cloud storage client in:
// https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/storage_stub_factory.cc
std::shared_ptr<grpc::Channel> CreateChannel(
    const std::string& address,
    const std::shared_ptr<grpc::ChannelCredentials>& creds, int channel_id) {
  grpc::ChannelArguments args;

  if (!IsDirectPathAddress(address)) {
    args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
    args.SetInt(GRPC_ARG_CHANNEL_ID, channel_id);
    args.SetInt(GRPC_ARG_DNS_ENABLE_SRV_QUERIES, 0);

    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                absl::ToInt64Milliseconds(absl::Minutes(5)));
  }

  // TODO: Consider adding the following gcp API flags:
  // args.SetInt(GRPC_ARG_TCP_TX_ZEROCOPY_ENABLED, 1);
  // args.SetCompressionAlgorithm(GRPC_COMPRESS_NONE);

  // The gRPC interceptor implements detailed logging.
  std::vector<
      std::unique_ptr<grpc::experimental::ClientInterceptorFactoryInterface>>
      interceptors;
  interceptors.push_back(std::make_unique<LoggingInterceptorFactory>());

  return grpc::experimental::CreateCustomChannelWithInterceptors(
      address, creds, args, std::move(interceptors));
}

}  // namespace

StorageStubPool::StorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<::grpc::ChannelCredentials> creds)
    : address_(std::move(address)) {
  size = ChannelsForAddress(address_, size);
  channels_.resize(size);
  stubs_.resize(size);

  ABSL_LOG_IF(INFO, gcs_grpc_logging)
      << "Connecting to " << address_ << " with " << size << " channels";

  for (int id = 0; id < channels_.size(); id++) {
    // See google cloud storage client in:
    // https://github.com/googleapis/google-cloud-cpp/blob/main/google/cloud/storage/internal/storage_stub_factory.cc
    channels_[id] = CreateChannel(address_, creds, id);
    stubs_[id] = Storage::NewStub(channels_[id]);
  }
}

void StorageStubPool::WaitForConnected(absl::Duration duration) {
  for (auto& channel : channels_) {
    channel->GetState(true);
  }
  if (duration > absl::ZeroDuration()) {
    auto timeout = absl::ToChronoTime(absl::Now() + duration);
    for (auto& channel : channels_) {
      channel->WaitForConnected(timeout);
    }
  }
  ABSL_LOG_IF(INFO, gcs_grpc_logging)
      << "Connection established to " << address_ << " in state "
      << channels_[0]->GetState(false);
}

std::shared_ptr<StorageStubPool> GetSharedStorageStubPool(
    std::string address, uint32_t size,
    std::shared_ptr<::grpc::ChannelCredentials> creds,
    absl::Duration wait_for_connected) {
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, std::shared_ptr<StorageStubPool>>>
      shared_pool;
  size = ChannelsForAddress(address, size);
  std::string key = absl::StrFormat("%d/%s", size, address);

  absl::MutexLock lock(&global_mu);
  auto& pool = (*shared_pool)[key];
  if (pool == nullptr) {
    pool = std::make_shared<StorageStubPool>(std::move(address), size,
                                             std::move(creds));
    pool->WaitForConnected(wait_for_connected);
  }
  return pool;
}

}  // namespace internal_gcs_grpc
}  // namespace tensorstore
