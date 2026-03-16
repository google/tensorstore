.. _open-as-void:

Raw Byte Access (``open_as_void``)
==================================

The ``open_as_void`` option provides raw byte-level access to zarr arrays with
structured data types, bypassing the normal field interpretation. This feature
is available for both the :ref:`driver/zarr2` and :ref:`driver/zarr3` drivers.

Supported Data Types
--------------------

The ``open_as_void`` option is only valid for structured data types:

- **Zarr v2**: ``structured`` dtype (NumPy-style structured arrays)
- **Zarr v3**: ``struct`` and ``structured`` dtypes

Attempting to use ``open_as_void`` with non-structured data types will result
in an error.

Purpose
-------

When opening an array with :json:`"open_as_void": true`, TensorStore exposes
the underlying byte representation of the array data rather than interpreting
it according to the stored field structure.

Behavior
--------

When ``open_as_void`` is enabled:

1. **Data type becomes byte**: The resulting TensorStore has dtype
   :json:schema:`~dtype.byte` regardless of the original structured data type.

2. **Additional dimension added**: A new innermost dimension is appended to
   represent the byte layout of each element. The size of this dimension
   equals the number of bytes per element in the original structured type.

3. **Codecs are preserved**: All encoding/decoding (including compression)
   is still applied. The raw bytes exposed are the *decoded* element bytes,
   not the raw compressed chunk data.

Dimension Transformation
~~~~~~~~~~~~~~~~~~~~~~~~

For an array with shape ``[D0, D1, ..., Dn]`` and a structured data type of
size ``B`` bytes per element, opening with ``open_as_void`` produces a
TensorStore with:

- Shape: ``[D0, D1, ..., Dn, B]``
- Rank: original rank + 1
- Data type: ``byte``

.. admonition:: Example: Zarr v2 structured dtype
   :class: example

   A zarr v2 array with structured dtype ``[("x", "|u1"), ("y", "<i2")]``
   (total 3 bytes per element) and shape ``[100, 200]`` becomes a ``byte``
   array with shape ``[100, 200, 3]`` when opened with ``open_as_void``.

   The byte layout follows the original field ordering, so for each element
   position ``[i, j]``:

   - Byte ``[i, j, 0]`` contains field ``x`` (1 byte)
   - Bytes ``[i, j, 1:3]`` contain field ``y`` (2 bytes, little-endian)

.. admonition:: Example: Zarr v3 struct dtype
   :class: example

   A zarr v3 array with a ``struct`` dtype containing two int32 fields
   (total 8 bytes per element) and shape ``[50, 50]`` becomes a ``byte``
   array with shape ``[50, 50, 8]`` when opened with ``open_as_void``.

Usage
-----

Zarr v2 (``zarr2`` driver)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "driver": "zarr",
     "kvstore": {"driver": "file", "path": "/path/to/array/"},
     "open_as_void": true
   }

Zarr v3 (``zarr3`` driver)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "driver": "zarr3",
     "kvstore": {"driver": "file", "path": "/path/to/array/"},
     "open_as_void": true
   }

Python Example
~~~~~~~~~~~~~~

.. code-block:: python

   import tensorstore as ts

   # Open an existing array with structured dtype as raw bytes
   store = ts.open({
       'driver': 'zarr',
       'kvstore': {'driver': 'file', 'path': '/path/to/array/'},
       'open_as_void': True,
   }).result()

   # Original array shape [100, 200] with structured dtype (3 bytes per element)
   # becomes shape [100, 200, 3] with byte dtype
   print(store.shape)  # (100, 200, 3)
   print(store.dtype)  # dtype("byte")

   # Read raw bytes for a single element at position [10, 20]
   element_bytes = store[10, 20, :].read().result()
   print(element_bytes)  # Array of 3 bytes representing the structured element

Constraints and Limitations
---------------------------

Mutual Exclusivity with Field Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``open_as_void`` option is mutually exclusive with the ``field`` option
used to select a specific field from structured data types. Specifying both
results in an error:

.. code-block:: json

   {
     "driver": "zarr",
     "kvstore": "...",
     "field": "x",
     "open_as_void": true
   }

This is invalid because ``open_as_void`` provides access to the *entire*
byte representation of all fields combined, while ``field`` selects a single
field's typed data.

URL Syntax Not Supported
~~~~~~~~~~~~~~~~~~~~~~~~

The TensorStore URL syntax (e.g., ``file:///path/|zarr2:``) does not support
the ``open_as_void`` option. Use the full JSON spec format instead.

Spec Round-Trip Preservation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``open_as_void`` flag is preserved when converting an opened TensorStore
back to a spec. This ensures that specs obtained from void-mode stores
correctly reflect their access mode.

Internal Implementation
-----------------------

The ``open_as_void`` feature is implemented through a synthesized "void field"
mechanism:

1. **Void Field Synthesis**: When ``open_as_void`` is requested, the driver
   creates a synthetic field descriptor that represents the entire structured
   element as raw bytes. This field has:

   - Data type: ``byte`` (single unsigned byte)
   - Field shape: ``[bytes_per_element]``
   - No field offset (covers all fields)

2. **Grid Specification**: The chunk grid is modified to include the
   additional bytes dimension in the component shape while preserving
   the original chunked dimensions.

3. **Encoding/Decoding**: The codec chain still operates on the original
   structured data representation. The void field transformation happens at
   the interface level, presenting decoded chunk data as raw bytes.

4. **Cache Separation**: Void-mode and normal-mode access to the same
   underlying array use separate cache entries to prevent data corruption
   from mixing typed and byte-level views.

Compatibility
-------------

Compression and Codecs
~~~~~~~~~~~~~~~~~~~~~~

``open_as_void`` is fully compatible with all compression codecs (blosc, gzip,
zstd, etc.) and other codecs (sharding, transpose, etc.). The raw bytes
accessed are the *decoded* structured element bytes after all codec processing.

Existing Arrays
~~~~~~~~~~~~~~~

``open_as_void`` can be used to open any existing zarr v2 or zarr v3 array
that has a structured data type. No special array creation flags are needed.

Interoperability
~~~~~~~~~~~~~~~~

Data accessed through ``open_as_void`` reflects the exact byte representation
as stored, including:

- Endianness (as specified by the field dtypes)
- Field alignment and padding
- Field ordering

This makes it suitable for verifying compatibility with other zarr
implementations or diagnosing encoding differences in structured data.

See Also
--------

- :ref:`driver/zarr2` - Zarr v2 driver documentation
- :ref:`driver/zarr3` - Zarr v3 driver documentation
- :json:schema:`driver/zarr2.open_as_void` - Zarr v2 spec option
- :json:schema:`driver/zarr3.open_as_void` - Zarr v3 spec option
