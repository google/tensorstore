# Tensorstore Benchmarks

This directory includes benchmarking utilities for  the tensorstore kvstore,
tensorstore reading/writing and concurrently reading/writing multiple tensorstores.


## kvstore benchmarks

The tensorstore/kvstore layer is the tensorstore component which handles
basic io.  Two benchmarks are available:

* `kvstore_benchmark` to benchmark io operations with a target size.

```
bazel run -c opt \
  //tensorstore/internal/benchmark:kvstore_benchmark -- \
  --kvstore_spec='"memory://abc/"' \
  --chunk_size=4194304 \
  --total_bytes=4294967296 \
  --read_blowup=500 \
  --repeat_writes=10 \
  --repeat_reads=100
```

* `kvstore_duration` to benchmark io operations over a designated duration.

```
bazel run -c opt \
  //tensorstore/internal/benchmark:kvstore_duration -- \
  --kvstore_spec='"file:///tmp/kvstore"' --duration=1m
```

## tensorstore benchmarks

The integrated `ts_benchmark` benchmarks reading and writing a single
tensorstore in a loop.

```
bazel run -c opt \
  //tensorstore/internal/benchmark:ts_benchmark -- \
  --alsologtostderr       \
  --strategy=sequential   \
  --total_read_bytes=-10  \
  --total_write_bytes=-2  \
  --chunk_bytes=2097152   \
  --repeat_reads=16       \
  --repeat_writes=8
```

## multi-tensorstore benchmarks

Benchmarks which read or write to multiplie tensorstores, which is similar
to the behavior in handling ai workload io, also provided by the following:

* `multi_genspec` generates a spec for the other multi-tensorstore tests.
* `multi_read_benchmark` benchmarks concurrent reading.
* `multi_write_benchmark` benchmarks concurrent writing.


###  Generate a spec/config file for shards.

```
bazel run -c opt \
  //tensorstore/internal/benchmark/multi_genspec  \
  --base_spec='{
    "driver": "zarr3",
    "kvstore": { "driver": "ocdbt", "base": "file:///tmp/checkpoint" }
  }'  \
  --config='[
    { "name": "array1", "dtype":"float32", "shape": [8192], "chunks": [2048] },
    { "name": "array2", "dtype":"float32", "shape": [8192,64,28672], "chunks": [512,32,7168] },
    { "name": "array3", "dtype":"float32", "shape": [32000,8192], "chunks":[8000,512] },
    { "name": "array4", "dtype":"float32", "shape": [28672,64,8192], "chunks":[7168,32,512] }
  ]'  > /tmp/config.json
```


### Run the write benchmark.

```
bazel run -c opt \
  //tensorstore/internal/benchmark/multi_write_benchmark  \
  --context_spec='{"file_io_concurrency": { "limit": 128 }}'  \
  --base_spec='{
    "driver": "zarr3",
    "kvstore": { "driver": "ocdbt", "base": "file:///tmp/checkpoint" }
  }'  \
  --repeat_writes=25 \
  --write_config=/tmp/config.json
```

### Run the read benchmark.

```
bazel run -c opt \
  //tensorstore/internal/benchmark/multi_read_benchmark  \
  --context_spec='{"file_io_concurrency": { "limit": 128 }}'  \
  --base_spec='{
    "driver": "zarr3",
    "kvstore": { "driver": "ocdbt", "base": "file:///tmp/checkpoint" }
  }'  \
  --repeat_reads=25 \
  --read_config=/tmp/config.json
```
