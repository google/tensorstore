# Copyright 2023 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import ml_dtypes
import numpy as np
import pytest
import tensorstore as ts

TESTING_DTYPES = [
    ml_dtypes.int4.dtype,
    # ml_dtypes.uint4, # TODO(ChromeHearts) implement uint4
    ml_dtypes.float8_e4m3fn.dtype,
    ml_dtypes.float8_e4m3fnuz.dtype,
    ml_dtypes.float8_e4m3b11fnuz.dtype,
    ml_dtypes.float8_e5m2.dtype,
    ml_dtypes.float8_e5m2fnuz.dtype,
    ml_dtypes.bfloat16.dtype,
    ts.int4.numpy_dtype,
    ts.float8_e4m3fn.numpy_dtype,
    ts.float8_e4m3fnuz.numpy_dtype,
    ts.float8_e4m3b11fnuz.numpy_dtype,
    ts.float8_e5m2.numpy_dtype,
    ts.float8_e5m2fnuz.numpy_dtype,
    ts.bfloat16.numpy_dtype,
    np.dtype("int4"),
    np.dtype("float8_e4m3fn"),
    np.dtype("float8_e4m3fnuz"),
    np.dtype("float8_e4m3b11fnuz"),
    np.dtype("float8_e5m2"),
    np.dtype("float8_e5m2fnuz"),
    np.dtype("bfloat16"),
]


@pytest.mark.parametrize("custom_type", TESTING_DTYPES)
def test_save_and_restore(custom_type):
  """test save and restore dtypes."""
  np.random.seed(0)
  shape = [3, 4, 5]

  t = ts.open({
      "driver": "zarr",
      "kvstore": "memory://",
      "key_encoding": ".",
      "metadata": {
          "shape": shape,
          "dtype": custom_type.name,
          "order": "C",
      },
      "create": True,
      "delete_existing": True,
  }).result()

  data = np.random.normal(scale=8, size=shape).astype(custom_type)
  print(f"{data=}")

  # blocking write
  t.write(data).result()

  read_data = t.read().result()

  assert np.array_equal(read_data, data)


@pytest.mark.parametrize("custom_type", TESTING_DTYPES)
def test_save_and_restore_cast_driver(custom_type):
  """Test save and restore by casting to a bigger dtype and make sure no precision loss."""

  np.random.seed(0)
  shape = [3, 4, 5]

  with tempfile.TemporaryDirectory() as temp_dir:
    write_store = ts.open({
        "driver": "cast",
        "dtype": custom_type.name,
        "base": {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": temp_dir,
            },
            "key_encoding": ".",
            "metadata": {
                "shape": shape,
                "dtype": (
                    # float32 is big enough to store all the custom types
                    "<f4"
                ),
                "order": "C",
            },
            "create": True,
            "delete_existing": True,
        },
    }).result()

    data = np.random.normal(scale=8, size=shape).astype(custom_type)
    print(f"{data=}")

    # blocking write
    write_store.write(data).result()

    # cast back to the custom_type
    read_store = ts.open({
        "driver": "cast",
        "dtype": custom_type.name,
        "base": {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": temp_dir,
            },
            "key_encoding": ".",
        },
    }).result()

    read_data = read_store.read().result()
    assert read_data.dtype == custom_type
    np.testing.assert_equal(read_data, data)
