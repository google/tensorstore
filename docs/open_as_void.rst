.. _open-as-void:

Raw Byte Access (``open_as_void``)
==================================

The ``open_as_void`` option provides raw byte-level access to zarr arrays,
bypassing the normal data type interpretation and exposing the underlying decoded
bytes. This feature is available for both the :ref:`driver/zarr2` and
:ref:`driver/zarr3` drivers.

Supported Data Types
--------------------

The scope of supported data types depends on the zarr version:

- **Zarr v2**: The ``open_as_void`` option is only valid for ``structured``
  dtype (NumPy-style structured arrays). Attempting to use it with non-structured
  data types will result in an error.
- **Zarr v3**: The ``open_as_void`` option works on **any** data type,
  including structured types (``struct`` and legacy read-only ``structured``),
  and non-structured types.

Purpose
-------

When opening an array with :json:`"open_as_void": true`, TensorStore exposes
the underlying byte representation of the array data rather than interpreting
it according to the stored data type or field structure.

Behavior
--------

Zarr v3 Behavior
~~~~~~~~~~~~~~~~

For zarr v3, ``open_as_void`` operates entirely during the resolution of the
codec pipeline. The implementation resolves the pipeline with a substituted
"raw" data type and then validates the result, rather than checking each codec
individually against an allowlist.

1. **Data type becomes byte**: The array's data type is replaced with the
   ``byte`` data type (a 1-byte, endian-invariant type) during codec pipeline
   resolution, regardless of the original data type.
2. **Additional dimension added**: A new innermost dimension is appended to
   represent the byte layout of each element. The size of this dimension equals
   the number of bytes per element in the original data type.
3. **Codec pipeline resolved with raw type**: The codec pipeline is resolved
   using the substituted ``byte`` data type. After resolution, the
   implementation verifies that:

   a. The innermost array-to-bytes encoding (after unwinding any
      ``sharding_indexed`` layers) is the ``bytes`` codec.
   b. The ``byte`` data type is preserved through all array-to-array codecs
      in the pipeline (i.e., no codec has changed the data type).

   This approach means that array-to-array codecs that preserve the raw data
   type (such as ``transpose``) are naturally supported, while codecs that
   transform element data (such as ``scale_offset`` or ``cast_value``) will
   fail validation because they alter the data type.
4. **Endianness is preserved natively**: The ``bytes`` codec, which normally
   decodes to the stored data type and handles endian conversion, sees the
   ``byte`` data type as endian-invariant and performs no byte swapping. It
   simply passes the decoded bytes through.
5. **Downstream transparency**: Because this is resolved at the codec pipeline
   level, downstream components (such as the chunk cache and grid specification)
   see the resulting ``byte`` array and extended shape without needing any
   special awareness of the ``open_as_void`` option.

Zarr v2 Behavior
~~~~~~~~~~~~~~~~

For zarr v2, ``open_as_void`` is implemented via void field synthesis at the
interface level:

1. **Data type becomes byte**: The resulting TensorStore has dtype
   :json:schema:`~dtype.byte`.
2. **Additional dimension added**: A new innermost dimension is appended, with
   size equal to the number of bytes per element in the original structured type.
3. **Codecs are preserved**: All encoding/decoding (including compression) is
   still applied based on the original structured data type. The raw bytes
   exposed are the *decoded* element bytes, not the raw compressed chunk data.

Dimension Transformation
------------------------

For an array with shape ``[D0, D1, ..., Dn]`` and a data type of size ``B``
bytes per element, opening with ``open_as_void`` produces a TensorStore with:

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

.. admonition:: Example: Zarr v3 float32 array
   :class: example

   A zarr v3 array with ``float32`` dtype (4 bytes per element) and shape
   ``[100, 200]`` becomes a ``byte`` array with shape ``[100, 200, 4]`` when
   opened with ``open_as_void``.

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

Zarr v3 Codec Restrictions
~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``open_as_void`` is enabled for zarr v3, the codec pipeline is resolved
with a substituted ``byte`` data type and validated (see
:ref:`Zarr v3 Behavior <open-as-void>` above). This means the effective codec
restrictions are a consequence of the validation rules:

- **``array -> bytes`` codecs**: The innermost array-to-bytes codec (after
  unwinding ``sharding_indexed`` layers) must be the ``bytes`` codec. Only the
  ``bytes`` and ``sharding_indexed`` (possibly nested) codecs are supported.
  Any other array-to-bytes codec will result in a validation error.
- **``array -> array`` codecs**: Any codec that preserves the ``byte`` data
  type is permitted. In practice, this means codecs that shuffle elements
  without transforming them (e.g., ``transpose``, and the proposed ``reshape``)
  are supported. Codecs that transform element data, such as ``scale_offset``,
  ``cast_value``, and ``bitround``, alter the data type and will fail
  validation.
- **``bytes -> bytes`` codecs**: All bytes-to-bytes codecs (e.g., ``gzip``,
  ``blosc``, ``zstd``, ``crc32c``) are allowed and operate unchanged.

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

Interoperability
----------------

Data accessed through ``open_as_void`` reflects the exact byte representation
as stored, including:

- Endianness (as specified by the field dtypes for zarr v2, or natively
  preserved by the ``bytes`` codec for zarr v3)
- Field alignment and padding (for structured types)
- Field ordering (for structured types)

This makes it suitable for verifying compatibility with other zarr
implementations or diagnosing encoding differences.

See Also
--------

- :ref:`driver/zarr2` - Zarr v2 driver documentation
- :ref:`driver/zarr3` - Zarr v3 driver documentation
- :json:schema:`driver/zarr2.open_as_void` - Zarr v2 spec option
- :json:schema:`driver/zarr3.open_as_void` - Zarr v3 spec option
