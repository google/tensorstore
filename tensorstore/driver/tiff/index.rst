.. _tiff-driver:

``tiff`` Driver
=============

The ``tiff`` driver provides **read-only** access to `TIFF (Tagged Image File Format) <https://libtiff.gitlab.io/libtiff/specification/index.html>`_
files backed by any supported :ref:`key_value_store`. It supports reading single Image File Directories (IFDs) or interpreting sequences of IFDs as 
additional dimensions (e.g., Z-stacks, time series).

.. important::
   This driver is **read-only**. It cannot be used to create new TIFF files or modify existing ones. Operations like writing or resizing will fail.

.. json:schema:: driver/tiff

TIFF Interpretation Options (`tiff` Object)
-------------------------------------------

The optional ``tiff`` object in the TensorStore specification controls how the TIFF file is interpreted. You can select one of two modes:

1.  **Single IFD Mode (Default):** Reads data from a single Image File Directory.
    * Use the :json:schema:`~driver/tiff.tiff.ifd` property to specify the 0-based index of the directory to read. If omitted, it defaults to `0`.

2.  **Multi-IFD Stacking Mode:** Interprets a sequence of IFDs as additional dimensions.
    * Use the :json:schema:`~driver/tiff.tiff.ifd_stacking` object to configure this mode. This is mutually exclusive with specifying a non-zero `ifd`.

        * :json:schema:`~driver/tiff.tiff.ifd_stacking.dimensions` (Required): An array of unique string labels for the dimensions represented by the IFD sequence (e.g., `["z"]`, `["time", "channel"]`). The order typically defines the default placement before the implicit 'y' and 'x' dimensions.
        * **Stacked Shape Definition** (One of the following is required):
            * :json:schema:`~driver/tiff.tiff.ifd_stacking.ifd_count`: (Positive integer) Required if `dimensions` has only one entry *and* `dimension_sizes` is absent. Defines the size of that single dimension.
            * :json:schema:`~driver/tiff.tiff.ifd_stacking.dimension_sizes`: (Array of positive integers) Required if `dimensions` has more than one entry. Must have the same length as `dimensions`. Defines the size of each corresponding stacked dimension.
        * :json:schema:`~driver/tiff.tiff.ifd_stacking.ifd_count` (Optional Verification): If specified alongside `dimension_sizes`, the product of `dimension_sizes` *must* equal `ifd_count`.
        * :json:schema:`~driver/tiff.tiff.ifd_sequence_order` (Optional): An array of string labels (must be a permutation of `dimensions`) specifying the iteration order of the stacked dimensions within the flat IFD sequence in the file. If omitted, the order defaults to the order in `dimensions`, with the *last* dimension varying fastest. For example, for `dimensions: ["t", "c"]`, the default sequence is `t0c0, t0c1, t0c2, ..., t1c0, t1c1, ...`.

* :json:schema:`~driver/tiff.tiff.sample_dimension_label` (Optional): A string label for the dimension derived from the `SamplesPerPixel` tag (if > 1). Defaults internally to `c`. This label must not conflict with labels in `ifd_stacking.dimensions`.

**Rules:**

* All IFDs involved in stacking must have uniform properties (Width, Height, DataType, Compression, etc.).
* The driver determines the final TensorStore dimension order based on the conceptual structure (Stacked dims..., Y, X, Sample dim) unless overridden by `schema.domain.labels`.

Compression
-----------

.. json:schema:: driver/tiff/Compression

The driver automatically detects and decodes the compression method specified in the TIFF file's `Compression` tag. The following compression types are currently supported for decoding:

.. json:schema:: driver/tiff/Compression/raw
.. json:schema:: driver/tiff/Compression/zlib
.. json:schema:: driver/tiff/Compression/zstd

*(Support for other compression types like LZW or PackBits may be added in the future).*

Mapping to TensorStore Schema
-----------------------------

The TensorStore schema is derived from the TIFF tags and the interpretation options specified.

Data Type
~~~~~~~~~

TensorStore infers the :json:schema:`~Schema.dtype` from the TIFF `BitsPerSample` and `SampleFormat` tags. Supported mappings include:

* Unsigned Integers (`SampleFormat=1`): `uint8`, `uint16`, `uint32`, `uint64`
* Signed Integers (`SampleFormat=2`): `int8`, `int16`, `int32`, `int64`
* Floating Point (`SampleFormat=3`): `float32`, `float64`

Mixed data types across samples (channels) are generally not supported. The driver handles byte order conversion (TIFF Little Endian or Big Endian) automatically based on the file header.

Domain
~~~~~~

The :json:schema:`~Schema.domain` is determined as follows:

* **Shape:**
    * The base shape comes from `ImageLength` (Y) and `ImageWidth` (X).
    * An additional dimension is added if `SamplesPerPixel` > 1.
    * Dimensions from `ifd_stacking.dimensions` are added.
    * The **default dimension order** is conceptually: `(stacked_dims..., 'y', 'x', sample_dim)`. For example, `("z", "y", "x")` or `("t", "c", "y", "x", "rgb")`. The actual final order can be permuted using `schema.domain.labels`.
* **Labels:**
    * Default conceptual labels are `y`, `x`, the labels from `ifd_stacking.dimensions`, and the `sample_dimension_label` (or default `c`) if SamplesPerPixel > 1.
    * These defaults become the final labels unless overridden by `schema.domain.labels`.
* **Origin:** The domain origin (:json:schema:`~IndexDomain.inclusive_min`) is always zero for all dimensions.
* **Resizing:** The domain is **fixed** and cannot be resized.

Chunk Layout
~~~~~~~~~~~~

The :json:schema:`~Schema.chunk_layout` is derived from the TIFF tiling or strip information:

* **Grid Shape:** Determined by `TileWidth`/`TileLength` or `ImageWidth`/`RowsPerStrip`, potentially including a size of 1 for stacked dimensions or the sample dimension (if planar).
* **Grid Origin:** Always zero for all dimensions.
* **Inner Order:** Defaults to C order relative to the final TensorStore dimension order (soft constraint). Can be overridden by `schema.chunk_layout.inner_order`.

Codec
~~~~~

The :json:schema:`~Schema.codec` indicates the use of the TIFF driver and specifies the detected :json:schema:`~driver/tiff.codec.compression`.

.. json:schema:: driver/tiff/Codec

Fill Value
~~~~~~~~~~

TIFF files do not store a fill value. Reads of missing chunks (unlikely in valid TIFFs) will be filled with zero if :json:schema:`~DriverSpec.fill_missing_data_reads` is true (default).

Dimension Units
~~~~~~~~~~~~~~~

Standard TIFF does not reliably store dimension units. Specify units using :json:schema:`Schema.dimension_units` during open.

Examples
--------

.. admonition:: Example: Opening the first IFD (Default)
   :class: example

   Opens the first image directory (IFD 0) in the specified TIFF file.

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/image.tif"}
      }

.. admonition:: Example: Opening a specific IFD
   :class: example

   Opens the 6th image directory (index 5) in the file.

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/image.tif"},
        "tiff": {
          "ifd": 5
        }
      }

.. admonition:: Example: Simple Z-Stack (50 planes)
   :class: example

   Interprets the first 50 IFDs as a Z-stack. Assumes `SamplesPerPixel=1`.

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/z_stack.tif"},
        "tiff": {
          "ifd_stacking": {
            "dimensions": ["z"],
            "ifd_count": 50
          }
        }
      }
   // Default TensorStore Dimensions: (z, y, x)
   // Resulting Shape (example): [50, height, width]

.. admonition:: Example: Z-Stack (50 planes) with RGB channels
   :class: example

   Interprets 50 IFDs as a Z-stack, where each IFD has `SamplesPerPixel=3`.
   Labels the sample dimension 'channel'.

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/z_stack_rgb.tif"},
        "tiff": {
          "ifd_stacking": {
            "dimensions": ["z"],
            "ifd_count": 50
          },
          "sample_dimension_label": "channel"
        }
      }
   // Default TensorStore Dimensions: (z, y, x, channel)
   // Resulting Shape (example): [50, height, width, 3]

.. admonition:: Example: Time (10) x Channel (3) Stack, SPP=1
   :class: example

   Interprets 30 IFDs (10 time points * 3 channels) as a T/C stack.
   Assumes default IFD sequence order (channel varies fastest: t0c0, t0c1, t0c2, t1c0, ...).

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/tc_stack.tif"},
        "tiff": {
          "ifd_stacking": {
            "dimensions": ["time", "channel"],
            "dimension_sizes": [10, 3]
          }
        }
      }
   // Default TensorStore Dimensions: (time, channel, y, x)
   // Resulting Shape (example): [10, 3, height, width]

.. admonition:: Example: Time (10) x Channel (3) Stack, SPP=1, Time Fastest in File
   :class: example

   Same data shape as above, but specifies that the IFDs in the file are ordered with time varying fastest (c0t0, c0t1, ..., c1t0, c1t1, ...).

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/tc_stack_t_fast.tif"},
        "tiff": {
          "ifd_stacking": {
            "dimensions": ["time", "channel"],
            "dimension_sizes": [10, 3],
            "ifd_sequence_order": ["channel", "time"] // channel slowest, time fastest
          }
        }
      }
   // Default TensorStore Dimensions: (time, channel, y, x) - Order is unaffected by ifd_sequence_order
   // Resulting Shape (example): [10, 3, height, width]

.. admonition:: Example: Overriding Dimension Order with Schema
   :class: example

   Opens the T/C stack from the previous example, but forces the final TensorStore dimension order to be Channel, Time, Y, X.

   .. code-block:: json

      {
        "driver": "tiff",
        "kvstore": {"driver": "file", "path": "/path/to/tc_stack_t_fast.tif"},
        "tiff": {
          "ifd_stacking": {
            "dimensions": ["time", "channel"], // Conceptual dimensions
            "dimension_sizes": [10, 3],
            "ifd_sequence_order": ["channel", "time"]
          }
        },
        "schema": {
          "domain": {
            "labels": ["channel", "time", "y", "x"] // Desired final order
          }
        }
      }
   // Final TensorStore Dimensions: (channel, time, y, x)
   // Resulting Shape (example): [3, 10, height, width]

Limitations
-----------

* **Read-Only:** The driver cannot create, write to, or resize TIFF files.
* **IFD Uniformity:** When using `ifd_stacking`, all involved IFDs must have consistent Width, Height, SamplesPerPixel, BitsPerSample, SampleFormat, PlanarConfiguration, Compression, and tiling/stripping configuration.
* **Unsupported TIFF Features:** Some TIFF features may not be supported, including:
    * Certain compression types (e.g., JPEG, LZW, PackBits - check supported list).
    * Planar configuration (`PlanarConfiguration=2`) combined with `ifd_stacking`.
    * Mixed data types or bits-per-sample across channels (samples).
    * Sub-byte data types (e.g., 1-bit, 4-bit).
    * Non-standard SampleFormat values.
* **Metadata Parsing:** Does not currently parse extensive metadata from ImageDescription or OME-XML (though basic properties are read from standard tags).