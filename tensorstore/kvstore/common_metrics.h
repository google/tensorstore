// Copyright 2020 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_COMMON_METRICS_H_
#define TENSORSTORE_KVSTORE_COMMON_METRICS_H_

#include <stdint.h>

#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/metadata.h"  // IWYU pragma: keep
#include "tensorstore/internal/metrics/registry.h"  // IWYU pragma: keep

// Defines the common metrics containers and registration macros for
// key-value store drivers.
//
// Example usage in a key-value store driver `.cc` file:
//
//   #include "tensorstore/internal/global_initializer.h"
//   #include "tensorstore/kvstore/common_metrics.h"
//
//   namespace {
//   static tensorstore::internal_kvstore::CommonMetrics g_metrics;
//
//   TENSORSTORE_GLOBAL_INITIALIZER {
//     TENSORSTORE_KVSTORE_REGISTER_COMMON_METRICS(&g_metrics, my_driver);
//   }
//   }  // namespace
//
namespace tensorstore {
namespace internal_kvstore {

// Holds the common read metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/read
//   /tensorstore/kvstore/driver/list
struct CommonReadMetrics {
  internal_metrics::Counter<int64_t> read;
  internal_metrics::Counter<int64_t> list;
};

// Holds the common write metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/write
//   /tensorstore/kvstore/driver/delete_range
struct CommonWriteMetrics {
  internal_metrics::Counter<int64_t> write;
  internal_metrics::Counter<int64_t> delete_range;
};

// Holds the detailed read metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/batch_read
//   /tensorstore/kvstore/driver/bytes_read
//   /tensorstore/kvstore/driver/read_latency_ms
struct DetailedReadMetrics {
  internal_metrics::Counter<int64_t> batch_read;
  internal_metrics::Counter<int64_t> bytes_read;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>
      read_latency_ms;
};

// Holds the detailed write metrics for a kvstore driver.
//   /tensorstore/kvstore/driver/bytes_written
//   /tensorstore/kvstore/driver/write_latency_ms
struct DetailedWriteMetrics {
  internal_metrics::Counter<int64_t> bytes_written;
  internal_metrics::Histogram<internal_metrics::DefaultBucketer>
      write_latency_ms;
};

// Combined struct for all common metrics.
struct CommonMetrics : public CommonReadMetrics,
                       public CommonWriteMetrics,
                       public DetailedReadMetrics,
                       public DetailedWriteMetrics {
  // no additional members
};

// Registers the common metrics.
// Each macro corresponds to one of the common metrics structs, above.
//
// \param metrics_ptr Pointer to the relevant metrics struct.
// \param KVSTORE Identifier (e.g., `file` or `gcs`).
//
#define TENSORSTORE_KVSTORE_REGISTER_COMMON_READ_METRICS(metrics_ptr, KVSTORE) \
  do {                                                                         \
    auto& r = ::tensorstore::internal_metrics::GetMetricRegistry();            \
    r.Register(&(metrics_ptr)->read,                                           \
               ::tensorstore::internal_metrics::MetricMetadata(                \
                   "/tensorstore/kvstore/" #KVSTORE "/read",                   \
                   #KVSTORE " kvstore::Read calls"));                          \
    r.Register(&(metrics_ptr)->list,                                           \
               ::tensorstore::internal_metrics::MetricMetadata(                \
                   "/tensorstore/kvstore/" #KVSTORE "/list",                   \
                   #KVSTORE " kvstore::List calls"));                          \
  } while (false)

#define TENSORSTORE_KVSTORE_REGISTER_COMMON_WRITE_METRICS(metrics_ptr, \
                                                          KVSTORE)     \
  do {                                                                 \
    auto& r = ::tensorstore::internal_metrics::GetMetricRegistry();    \
    r.Register(&(metrics_ptr)->write,                                  \
               ::tensorstore::internal_metrics::MetricMetadata(        \
                   "/tensorstore/kvstore/" #KVSTORE "/write",          \
                   #KVSTORE " kvstore::Write calls"));                 \
    r.Register(&(metrics_ptr)->delete_range,                           \
               ::tensorstore::internal_metrics::MetricMetadata(        \
                   "/tensorstore/kvstore/" #KVSTORE "/delete_range",   \
                   #KVSTORE " kvstore::DeleteRange calls"));           \
  } while (false)

#define TENSORSTORE_KVSTORE_REGISTER_DETAILED_READ_METRICS(metrics_ptr,   \
                                                           KVSTORE)       \
  do {                                                                    \
    auto& r = ::tensorstore::internal_metrics::GetMetricRegistry();       \
    r.Register(&(metrics_ptr)->batch_read,                                \
               ::tensorstore::internal_metrics::MetricMetadata(           \
                   "/tensorstore/kvstore/" #KVSTORE "/batch_read",        \
                   #KVSTORE " kvstore::Read after batching"));            \
    r.Register(                                                           \
        &(metrics_ptr)->bytes_read,                                       \
        ::tensorstore::internal_metrics::MetricMetadata(                  \
            "/tensorstore/kvstore/" #KVSTORE "/bytes_read", "bytes read") \
            .WithUnits(::tensorstore::internal_metrics::Units::kBytes));  \
    r.Register(                                                           \
        &(metrics_ptr)->read_latency_ms,                                  \
        ::tensorstore::internal_metrics::MetricMetadata(                  \
            "/tensorstore/kvstore/" #KVSTORE "/read_latency_ms",          \
            #KVSTORE " kvstore::Read latency (ms)")                       \
            .WithUnits(                                                   \
                ::tensorstore::internal_metrics::Units::kMilliseconds));  \
  } while (false)

#define TENSORSTORE_KVSTORE_REGISTER_DETAILED_WRITE_METRICS(metrics_ptr, \
                                                            KVSTORE)     \
  do {                                                                   \
    auto& r = ::tensorstore::internal_metrics::GetMetricRegistry();      \
    r.Register(                                                          \
        &(metrics_ptr)->bytes_written,                                   \
        ::tensorstore::internal_metrics::MetricMetadata(                 \
            "/tensorstore/kvstore/" #KVSTORE "/bytes_written",           \
            "bytes written")                                             \
            .WithUnits(::tensorstore::internal_metrics::Units::kBytes)); \
    r.Register(                                                          \
        &(metrics_ptr)->write_latency_ms,                                \
        ::tensorstore::internal_metrics::MetricMetadata(                 \
            "/tensorstore/kvstore/" #KVSTORE "/write_latency_ms",        \
            #KVSTORE " kvstore::Write latency (ms)")                     \
            .WithUnits(                                                  \
                ::tensorstore::internal_metrics::Units::kMilliseconds)); \
  } while (false)

// Combined registration macro for all common metrics.
#define TENSORSTORE_KVSTORE_REGISTER_COMMON_METRICS(metrics_ptr, KVSTORE)      \
  do {                                                                         \
    TENSORSTORE_KVSTORE_REGISTER_COMMON_READ_METRICS(metrics_ptr, KVSTORE);    \
    TENSORSTORE_KVSTORE_REGISTER_COMMON_WRITE_METRICS(metrics_ptr, KVSTORE);   \
    TENSORSTORE_KVSTORE_REGISTER_DETAILED_READ_METRICS(metrics_ptr, KVSTORE);  \
    TENSORSTORE_KVSTORE_REGISTER_DETAILED_WRITE_METRICS(metrics_ptr, KVSTORE); \
  } while (false)

}  // namespace internal_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_COMMON_METRICS_H_
