.. _ocdbt-kvstore-driver:

``ocdbt`` Key-Value Store driver
================================

.. warning::

   This driver is experimental and the format is not yet stable.

The ``ocdbt`` driver implements an Optionally-Cooperative Distributed B+Tree on
top of a base key-value store.

.. json:schema:: kvstore/ocdbt

.. json:schema:: kvstore/ocdbt/Compression/zstd

.. json:schema:: Context.ocdbt_coordinator

Storage format
--------------

The database is represented by the following entries within a prefix of an
underlying key-value store:

- :file:`manifest.ocdbt`

  Stores the encoded *manifest*, which specifies the database configuration and
  the tree of versions.

- :file:`d/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

  Log-structured data files that store:

  - Out-of-line raw values too large to store inline in a B+tree node;
  - Encoded version tree nodes;
  - Encoded B+Tree nodes.

  The :file:`xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` portion of the filename is the
  lowercase hex representation of a 128-bit random identifier.

To read a key from the database, a client first reads the manifest file, then
traverses the version tree to locate the root B+tree node of the desired
version, then traverses the B+tree to locate the leaf node entry for the desired
key.  If the value is small and is stored inline in the leaf B+tree node, it is
immediately available from the leaf node.  Otherwise, the leaf node contains a
pointer to the value and it must be read separately from a data file.

.. _ocdbt-manifest-format:

Manifest format
^^^^^^^^^^^^^^^

.. |varint| replace:: ``varint``

.. |header_magic_format| replace:: ``uint32be``

.. |header_version_format| replace:: |varint|

.. |header_compression_format| replace:: |varint|

.. |header_length_format| replace:: ``uint64le``

.. |crc32c_format| replace:: ``uint32le``

An encoded manifest consists of:

- :ref:`ocdbt-manifest-header`
- Body compressed according to the specified :ref:`ocdbt-manifest-compression-format`:

  - :ref:`ocdbt-manifest-config`
  - :ref:`ocdbt-manifest-version-tree`
- :ref:`ocdbt-manifest-footer`

.. _ocdbt-manifest-header:

Manifest header
~~~~~~~~~~~~~~~

+----------------------------------------+---------------------------+
|Field                                   |Binary format              |
+========================================+===========================+
|:ref:`ocdbt-manifest-magic-value`       ||header_magic_format|      |
+----------------------------------------+---------------------------+
|:ref:`ocdbt-manifest-length`            ||header_length_format|     |
+----------------------------------------+---------------------------+
|:ref:`ocdbt-manifest-version`           ||header_version_format|    |
+----------------------------------------+---------------------------+
|:ref:`ocdbt-manifest-compression-format`||header_compression_format||
+----------------------------------------+---------------------------+

.. _ocdbt-manifest-magic-value:

``magic_value``
  Must equal ``0x0cdb3a2a``

.. _ocdbt-manifest-length:

``length``
  Length in bytes of entire manifest, including this header.

.. _ocdbt-manifest-version:

``version``
  Must equal ``0``.

.. _ocdbt-manifest-crc32c-checksum:

``crc32c_checksum``
  CRC32C checksum of entire manifest, Length in bytes of entire manifest, including `this
  header<ocdbt-manifest-header>`.

.. _ocdbt-manifest-compression-format:

``compression_format``
  ``0`` for uncompressed, ``1`` for zstd.

.. _ocdbt-manifest-config:

Manifest configuration
~~~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------+--------------+
|Field                                        |Binary format |
+=============================================+==============+
|:ref:`ocdbt-config-uuid`                     |``ubyte[16]`` |
+---------------------------------------------+--------------+
|:ref:`ocdbt-config-max-inline-value-bytes`   ||varint|      |
+---------------------------------------------+--------------+
|:ref:`ocdbt-config-max-decoded-node-bytes`   ||varint|      |
+---------------------------------------------+--------------+
|:ref:`ocdbt-config-version-tree-arity-log2`  |``uint8``     |
+---------------------------------------------+--------------+
|:ref:`ocdbt-config-compression-method`       ||varint|      |
+---------------------------------------------+--------------+
|:ref:`ocdbt-config-compression-configuration`|              |
+---------------------------------------------+--------------+

.. _ocdbt-config-uuid:

``uuid``
  Unique 128-bit identifier for the database.  If not specified explicitly, is
  randomly generated when the database is first created.

.. _ocdbt-config-max-inline-value-bytes:

``max_inline_value_bytes``
  Maximum size of a value to store inline within a B+Tree node.

.. _ocdbt-config-max-decoded-node-bytes:

``max_decoded_note_bytes``
  Maximum (uncompressed) size of a B+Tree node.

.. _ocdbt-config-version-tree-arity-log2:

``version_tree_arity_log2``
  Base-2 logarithm of the arity of the version tree.

.. _ocdbt-config-compression-method:

``compression_method``
  ``0`` for compressed, ``1`` for Zstandard.

.. _ocdbt-config-compression-configuration:

Compression configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

If the :ref:`ocdbt-config-compression-method` is not ``0``, it is followed by
the method-specific configuration.

Zstd compression configuration
""""""""""""""""""""""""""""""

+-------------------------------+--------------+
|Field                          |Binary format |
+===============================+==============+
|:ref:`ocdbt-config-zstd-level` |``int32le``   |
+-------------------------------+--------------+

.. _ocdbt-config-zstd-level:

``level``
  Compresion level to use when writing.

.. _ocdbt-manifest-version-tree:

Manifest version tree
~~~~~~~~~~~~~~~~~~~~~

Following the compression configuration, the manifest specifies references to
B+tree roots and version tree nodes.

.. |generation_number_format| replace:: |varint|

+------------------------------------------------------+---------------------------------------------------+
|Field                                                 |Binary format                                      |
+======================================================+===================================================+
|:ref:`ocdbt-manifest-version-tree-inline-versions`    |:ref:`ocdbt-version-tree-leaf-node-entry-array`    |
+------------------------------------------------------+---------------------------------------------------+
|:ref:`ocdbt-manifest-version-tree-version-nodes`      |:ref:`ocdbt-version-tree-interior-node-entry-array`|
+------------------------------------------------------+---------------------------------------------------+

.. _ocdbt-manifest-version-tree-inline-versions:

``inline_versions``
  References to the most recent versions.

  Older versions are referenced indirectly via
  :ref:`ocdbt-manifest-version-tree-version-nodes`.

.. _ocdbt-manifest-version-tree-version-nodes:

``version_nodes``
  References to version tree interior nodes for versions older than those
  referenced from `ocdbt-manifest-version-tree-inline-versions`.

.. _ocdbt-manifest-footer:

Manifest footer
~~~~~~~~~~~~~~~

+-------------------------------------+---------------+
|Field                                |Binary format  |
+=====================================+===============+
|:ref:`ocdbt-manifest-crc32c-checksum`||crc32c_format||
+-------------------------------------+---------------+

.. _ocdbt-manifest-crc32-checksum:

``crc32c_checksum``
  CRC-32C checksum of the entire manifest, excluding the checksum itself.

.. _ocdbt-version-tree:

Version tree node format
^^^^^^^^^^^^^^^^^^^^^^^^

An encoded version tree node consists of:

- :ref:`ocdbt-version-tree-outer-header`
- Body compressed according to the specified :ref:`ocdbt-version-tree-compression-format`:

  - :ref:`ocdbt-version-tree-inner-header`
  - :ref:`Leaf node entries<ocdbt-version-tree-leaf-node-entry-array>` or
    :ref:`Interior node entries<ocdbt-version-tree-interior-node-entry-array>`,
    depending on the :ref:`ocdbt-version-tree-height`.
- :ref:`ocdbt-version-tree-footer`

.. _ocdbt-version-tree-outer-header:

Version tree node outer header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------------+---------------------------+
|Field                                       |Binary format              |
+============================================+===========================+
|:ref:`ocdbt-version-tree-magic-value`       ||header_magic_format|      |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-version-tree-length`            ||header_length_format|     |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-version-tree-version`           ||header_version_format|    |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-version-tree-compression-format`||header_compression_format||
+--------------------------------------------+---------------------------+

.. _ocdbt-version-tree-magic-value:

``magic_value``
  Must equal ``0x0cdb1234``.

.. _ocdbt-version-tree-length:

``length``
  Length in bytes of entire version tree node, including this header.

.. _ocdbt-version-tree-version:

``version``
  Must equal ``0``.

.. _ocdbt-version-tree-compression-format:

``compression_format``
  ``0`` for uncompressed, ``1`` for zstd.

The remaining data is encoded according to the specified
:ref:`ocdbt-version-tree-compression-format`.

.. _ocdbt-version-tree-inner-header:

Version tree node inner header
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------------------+--------------------------+
|Field                                            |Binary format             |
+=================================================+==========================+
|:ref:`ocdbt-version-tree-version-tree-arity-log2`|``uint8``                 |
+-------------------------------------------------+--------------------------+
|:ref:`ocdbt-version-tree-height`                 |``uint8``                 |
+-------------------------------------------------+--------------------------+

.. _ocdbt-version-tree-version-tree-arity-log2:

``version_tree_arity_log2``
  Base-2 logarithm of the version tree node arity.  Must match the arity
  specified in the manifest from which this node was reached.

.. _ocdbt-version-tree-height:

``height``
  Height of this version tree node.  Leaf nodes have a height of 0.

  It is required that

  .. code-block:: cpp

     height * version_tree_arity_log2 < 64

The format of the remaining data depends on the value of ``height``.

.. _ocdbt-version-tree-leaf-node-entry-array:

Version tree leaf node entries format (``height = 0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same encoded representation is used for both the entries of a leaf
:ref:`version tree node<ocdbt-version-tree>` and for the
:ref:`ocdbt-manifest-version-tree-inline-versions` specified in the
:ref:`manfiest<ocdbt-manifest-version-tree>`.

.. |data_file_id_format| replace:: ``ubyte[16]``

.. |data_file_offset_format| replace:: |varint|

.. |data_file_length_format| replace:: |varint|

.. |num_keys_statistic_format| replace:: |varint|

.. |num_tree_bytes_statistic_format| replace:: |varint|

.. |num_indirect_value_bytes_statistic_format| replace:: |varint|

+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|Field                                                  |Binary format                              |Count                                      |
+=======================================================+===========================================+===========================================+
|:ref:`ocdbt-version-tree-leaf-num-versions`            ||varint|                                   |1                                          |
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-generation-number`       ||generation_number_format|                 |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-root-height`             |``uint8``                                  |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-data-file-id`            ||data_file_id_format|                      |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-data-file-offset`        ||data_file_offset_format|                  |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-data-file-length`        ||data_file_length_format|                  |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-num-keys`                ||num_keys_statistic_format|                |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-num-tree-bytes`          ||num_tree_bytes_statistic_format|          |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-num-indirect-value-bytes`||num_indirect_value_bytes_statistic_format||:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+
|:ref:`ocdbt-version-tree-leaf-commit-time`             |``uint64le``                               |:ref:`ocdbt-version-tree-leaf-num-versions`|
+-------------------------------------------------------+-------------------------------------------+-------------------------------------------+

.. _ocdbt-version-tree-leaf-num-versions:

``num_versions``
  Number of B+tree roots that are referenced.  The value is constrained based on
  the value of ``generation_number[num_versions-1]``, the latest generation
  number referenced from the version tree node, and
  :ref:`ocdbt-version-tree-version-tree-arity-log2`:

  .. code-block:: cpp

     1 <= num_versions <= (generation_number[num_versions-1] - 1)
                        % (1 << version_tree_arity_log2)
                        + 1

  .. note::

     The same computation of ``num_versions`` applies to both leaf node entries
     included in a :ref:`version tree node<ocdbt-version-tree>`, and
     :ref:`ocdbt-manifest-version-tree-inline-versions` included in the
     :ref:`manfiest<ocdbt-manifest-version-tree>`.  In the former case, the and
     :ref:`ocdbt-version-tree-version-tree-arity-log2` value is obtained from
     the version node.  In the latter case, the
     :ref:`ocdbt-config-version-tree-arity-log2` value is taken from the
     manifest.

.. _ocdbt-version-tree-leaf-generation-number:

``generation_number[i]``
  Generation number of the referenced B+tree root.  Must not be 0.  The
  generation numbers must be strictly increasing, i.e. if ``i < j``, then
  ``generation_number[i] < generation_number[j]``.

.. _ocdbt-version-tree-leaf-root-height:

``root_height[i]``
  Height of the referenced B+tree root.  Must be 0 if there is no root node.

.. _ocdbt-version-tree-leaf-commit-time:

``commit_time[i]``
  Time at which the generation was created, in milliseconds since the Unix
  epoch (excluding leap seconds).

.. _ocdbt-version-tree-leaf-data-file-id:

``data_file_id[i]``
  Specifies the data file containing the encoded root B+tree node.

.. _ocdbt-version-tree-leaf-data-file-offset:

``data_file_offset[i]``
  Specifies the starting byte offset within
  :ref:`ocdbt-version-tree-leaf-data-file-id` of the encoded root B+tree node.

.. _ocdbt-version-tree-leaf-data-file-length:

``data_file_length[i]``
  Specifies the byte length within :ref:`ocdbt-version-tree-leaf-data-file-id`
  of the encoded root B+tree node.

.. _ocdbt-version-tree-leaf-num-keys:

``num_keys[i]``
  Specifies the total number of (leaf-node) keys within the B+tree.  Note
  that if there is more than one path from the root to a given leaf node,
  the leaf node's keys are counted more than once.

.. _ocdbt-version-tree-leaf-num-tree-bytes:

``num_tree_bytes[i]``
  Specifies the total encoded size in bytes of all B+tree nodes reachable
  from the root, including the root itself.  A given node is counted once
  for each unique path within the tree to it; if there is more than one
  path to a node, its size is counted multiple times.

.. _ocdbt-version-tree-leaf-num-indirect-value-bytes:

``num_indirect_value_bytes[i]``
  Specifies the total size in bytes of all indirectly-stored values in the
  B+tree.  If the same stored value is referenced from multiple keys, its
  size is counted multiple times.

.. _ocdbt-version-tree-interior-node-entry-array:

Interior version tree node entries (``height > 0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same encoded representation is used for both the entries of an interior
:ref:`version tree node<ocdbt-version-tree>` and for the
:ref:`ocdbt-manifest-version-tree-version-nodes` version tree nodes specified in
the :ref:`manfiest<ocdbt-manifest-version-tree>`, but the interpretation
differs, as described below.

+----------------------------------------------------+--------------------------+-----------------------------------------------+
|Field                                               |Binary format             |Count                                          |
+====================================================+==========================+===============================================+
|:ref:`ocdbt-version-tree-interior-num-children`     ||varint|                  |1                                              |
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-generation-number`||generation_number_format||:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-data-file-id`     ||data_file_id_format|     |:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-data-file-offset` ||data_file_offset_format| |:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-data-file-length` ||data_file_length_format| |:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-num-generations`  ||varint|                  |:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+
|:ref:`ocdbt-version-tree-interior-commit-time`      |``uint64le``              |:ref:`ocdbt-version-tree-interior-num-children`|
+----------------------------------------------------+--------------------------+-----------------------------------------------+

When the encoded representation is used to specify
:ref:`ocdbt-manifest-version-tree-version-nodes` in the manifest, there is one
additional field:

+-----------------------------------------------+-------------+-----------------------------------------------+
|Field                                          |Binary format|Count                                          |
+===============================================+=============+===============================================+
|:ref:`ocdbt-version-tree-interior-child-height`|``uint8``    |:ref:`ocdbt-version-tree-interior-num-children`|
+-----------------------------------------------+-------------+-----------------------------------------------+

.. _ocdbt-version-tree-interior-num-children:

``num_children``
  Number of version tree nodes that are referenced.

  - When this encoded representation is used to specify the entries of an interior
    :ref:`version tree node<ocdbt-version-tree>`, ``num_children`` is constrained by:

    .. code-block:: cpp

       1 <= num_children
         <= ((generation_number[num_children-1]
              >> (version_tree_arity_log2 * height))
             - 1)
             % (1 << version_tree_arity_log2)
            + 1

    :ref:`ocdbt-version-tree-version-tree-arity-log2` and
    :ref:`ocdbt-version-tree-height` are obtained from the version tree node.

.. _ocdbt-version-tree-interior-generation-number:

``generation_number[i]``
  Latest B+tree root generation number referenced within this subtree. Must not
  be 0.  The generation numbers must be strictly increasing, i.e. if ``i < j``,
  then ``generation_number[i] < generation_number[j]``.

.. _ocdbt-version-tree-interior-data-file-id:

``data_file_id[i]``
  Specifies the data file containing the encoded version tree node.

.. _ocdbt-version-tree-interior-data-file-offset:

``data_file_offset[i]``
  Specifies the starting byte offset within ``data_file_id[i]`` of the encoded
  version tree node.

.. _ocdbt-version-tree-interior-data-file-length:

``data_file_length[i]``
  Specifies the byte length within ``data_file_id[i]`` of the encoded version
  tree node.

.. _ocdbt-version-tree-interior-num-generations:

``num_generations[i]``
  Total number of B+tree roots referenced within this subtree.

.. _ocdbt-version-tree-interior-commit-time:

``commit_time[i]``
  Commit time, in milliseconds since the Unix epoch (excluding leap
  seconds), of the earlier B+tree root referenced within this subtree.

  .. note::

     This is the *earliest* commit time referenced within this subtree, in
     contrast with ``generation_number[i]``, which specifies the *latest*
     generation number referenced within this subtree.

.. _ocdbt-version-tree-interior-child-height:

``entry_height[i]``
  Specifies the height of the referenced version tree node.

  This field is only present when the encoded representation is used to specify
  :ref:`ocdbt-manifest-version-tree-version-nodes` in the manifest.  The heights
  must be decreasing, i.e. ``entry_height[i] > entry_height[j]`` if ``i < j``.

  When the encoded representaiton is used to specify the entries of an interior
  version tree node, this field is not present and instead, for the purpose of
  this specification, ``entry_height[i]`` is implicitly equal to ``height - 1``,
  where :ref:`ocdbt-version-tree-height` is obtained from the version tree node.

.. _ocdbt-version-tree-footer:

Version tree node footer
~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------------+---------------+
|Field                                    |Binary format  |
+=========================================+===============+
|:ref:`ocdbt-version-tree-crc32c-checksum`||crc32c_format||
+-----------------------------------------+---------------+

.. _ocdbt-version-tree-crc32c-checksum:

``crc32c_checksum``
  CRC-32C checksum of the entire version tree node, excluding the checksum itself.

B+tree node format
^^^^^^^^^^^^^^^^^^

An encoded B+tree node consists of:

- :ref:`ocdbt-btree-outer-header`
- Body compressed according to the specified :ref:`ocdbt-btree-compression-format`:

  - :ref:`ocdbt-btree-inner-header`
  - :ref:`Leaf node entries<ocdbt-btree-leaf-node-entry-array>` or
    :ref:`Interior node entries<ocdbt-btree-interior-node-entry-array>`,
    depending on the :ref:`ocdbt-btree-node-height`.
- :ref:`ocdbt-btree-footer`

.. _ocdbt-btree-outer-header:

B+tree node outer header
~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------------+---------------------------+
|Field                                       |Binary format              |
+============================================+===========================+
|:ref:`ocdbt-btree-magic-value`              ||header_magic_format|      |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-btree-length`                   ||header_length_format|     |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-btree-version`                  ||header_version_format|    |
+--------------------------------------------+---------------------------+
|:ref:`ocdbt-btree-compression-format`       ||header_compression_format||
+--------------------------------------------+---------------------------+

.. _ocdbt-btree-magic-value:

``magic_value``
  Must equal ``0x0cdb20de``.

.. _ocdbt-btree-length:

``length``
  Length in bytes of entire B+tree node, including this header.

.. _ocdbt-btree-version:

``version``
  Must equal ``0``.

.. _ocdbt-btree-compression-format:

``compression_format``
  ``0`` for uncompressed, ``1`` for zstd.

The remaining data is encoded according to the specified
:ref:`ocdbt-btree-compression-format`.

.. _ocdbt-btree-inner-header:

B+tree node inner header
~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------+-------------+
|Field                              |Binary format|
+===================================+=============+
|:ref:`ocdbt-btree-node-height`     |``uint8``    |
+-----------------------------------+-------------+
|:ref:`ocdbt-btree-node-num-entries`||varint|     |
+-----------------------------------+-------------+

.. _ocdbt-btree-node-height:

``height``
  Height of this B+tree node.  Leaf nodes have a height of 0.

.. _ocdbt-btree-node-num-entries:

``num_entries``
  Number of children (if ``height > 0``) or key/value pairs (if ``height == 0``)
  referenced from this node.

The format of the remaining data depends on the value of ``height``.

.. _ocdbt-btree-leaf-node-entry-array:

Leaf B+tree node format (``height = 0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   B+tree leaf nodes do not directly store the full key for each key/value
   entry.  Instead, only the ``relative_key`` for each entry is stored; the
   prefix that must be prepended to this ``relative_key`` to obtain the full
   ``key`` is defined by the path from the root node to the leaf node.

+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|Field                                            |Binary format                   |Count                                            |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-key-prefix-length`   ||varint|                        |:ref:`ocdbt-btree-node-num-entries` - 1          |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-key-suffix-length`   ||varint|                        |:ref:`ocdbt-btree-node-num-entries`              |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-key-suffix`          |``byte[key_suffix_length[i]]``  |:ref:`ocdbt-btree-node-num-entries`              |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-encoded-value-length`||varint|                        |:ref:`ocdbt-btree-node-num-entries`              |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-data-file-id`        ||data_file_id_format|           |:ref:`ocdbt-btree-leaf-node-num-indirect-entries`|
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-data-file-offset`    ||data_file_offset_format|       |:ref:`ocdbt-btree-leaf-node-num-indirect-entries`|
+-------------------------------------------------+--------------------------------+-------------------------------------------------+
|:ref:`ocdbt-btree-leaf-node-value`               |``byte[direct_value_length[j]]``|:ref:`ocdbt-btree-leaf-node-num-direct-entries`  |
+-------------------------------------------------+--------------------------------+-------------------------------------------------+


.. _ocdbt-btree-leaf-node-key-prefix-length:

``key_prefix_length[i]``
  Length in bytes of common prefix of ``relative_key[i]`` and
  ``relative_key[i+1]``.  For the first key, no common prefix is stored.

.. _ocdbt-btree-leaf-node-key-suffix-length:

``key_suffix_length[i]``
  Length in bytes of each relative key, excluding the length of the common
  prefix with the previous key.  For the first key, the total length is
  stored, since there is no previous key.

.. _ocdbt-btree-leaf-node-key-suffix:

``key_suffix[i]``
  Relative key for the entry, excluding the common prefix with the previous
  key.  For the first key, the entire key is stored.  Note that the
  suffixes for all keys are concatenated in the encoded representation.

.. _ocdbt-btree-leaf-node-encoded-value-length:

``encoded_value_length[i]``
  Encoded length in bytes of the value for this key/value entry.  The
  actual length is given by:

  .. code-block:: cpp

     value_length[i] = encoded_value_length[i] >> 1

  The least significant bit indicates how the value is stored:

  - ``0`` if the value is stored inline in this leaf node,
  - ``1`` if the value is stored out-of-line.

  Based on this column, the following derived values are defined:

  .. _ocdbt-btree-leaf-node-num-direct-entries:

  ``num_direct_entries``
    The number of entries for which
    ``(encode_value_length[i] & 1) == 0``.

  ``direct_entries``
    The array of indices of direct values.

  ``direct_value_length[j]``
    Equal to ``value_length[direct_values[j]]``.

  .. _ocdbt-btree-leaf-node-num-indirect-entries:

  ``num_indirect_entries``
    Equal to ``num_entries - num_direct_entries``.

  ``indirect_entries``
    The array of indices of indirect values.

.. _ocdbt-btree-leaf-node-data-file-id:

``data_file_id[j]``
  Specifies the dstarting byte offset within ``data_file_id[j]`` of the
  value for entry ``indirect_values[j]``.  Only stored for entries with
  out-of-line values.

.. _ocdbt-btree-leaf-node-data-file-offset:

``data_file_offset[j]``
  Specifies the starting byte offset within ``data_file_id[j]`` of the
  value for entry ``indirect_values[j]``.  Only stored for entries with
  out-of-line values.

.. _ocdbt-btree-leaf-node-value:

``value[j]``
  Specifies the value for entry ``direct_values[j]``.  Only stored for
  entries with inline values.  Note that all direct values are concatenated
  in the encoded representation.

.. _ocdbt-btree-interior-node-entry-array:

Interior B+tree node format (``height > 0``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   B+tree interior nodes do not directly store the full starting key for each
   child node.  Instead, only the ``relative_key`` for each child node is
   stored; the prefix that must be prepended to this ``relative_key`` to obtain
   the full ``key`` is defined by the path from the root node.

+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|Field                                                        |Binary format                              |Count                                  |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-subtree-common-prefix-length`||varint|                                   |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-key-prefix-length`           ||varint|                                   |:ref:`ocdbt-btree-node-num-entries` - 1|
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-key-suffix-length`           ||varint|                                   |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-key-suffix`                  |``byte[key_suffix_length[i]]``             |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-data-file-id`                ||data_file_id_format|                      |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-data-file-offset`            ||data_file_offset_format|                  |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-data-file-length`            ||data_file_length_format|                  |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-num-keys`                    ||num_keys_statistic_format|                |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-num-tree-bytes`              ||num_tree_bytes_statistic_format|          |:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+
|:ref:`ocdbt-btree-interior-node-num-indirect-value-bytes`    ||num_indirect_value_bytes_statistic_format||:ref:`ocdbt-btree-node-num-entries`    |
+-------------------------------------------------------------+-------------------------------------------+---------------------------------------+

.. _ocdbt-btree-interior-node-subtree-common-prefix-length:

``subtree_common_prefix_length[i]``
  Length in bytes of the prefix of ``relative_key[i]`` that is common to
  all keys within the subtree rooted at this child node.  This prefix
  serves as an implicit prefix of all keys within the subtree rooted at the
  child.

.. _ocdbt-btree-interior-node-key-prefix-length:

``key_prefix_length[i]``
  Length in bytes of common prefix of ``relative_key[i]`` and
  ``relative_key[i+1]``.  For the first key, no common prefix is stored.

.. _ocdbt-btree-interior-node-key-suffix-length:

``key_suffix_length[i]``
  Length in bytes of each relative key, excluding the length of the common
  prefix with the previous key.  For the first key, the total length is
  stored, since there is no previous key.

.. _ocdbt-btree-interior-node-key-suffix:

``key_suffix[i]``
  Relative key for the entry, excluding the common prefix with the previous
  key.  For the first key, the entire key is stored.  Note that the
  suffixes for all keys are concatenated in the encoded representation.

.. _ocdbt-btree-interior-node-data-file-id:

``data_file_id[i]``
  Specifies the data file containing the encoded child B+tree node.

.. _ocdbt-btree-interior-node-data-file-offset:

``data_file_offset[i]``
  Specifies the starting byte offset within ``data_file_id[i]`` of the
  encoded child B+tree node.

.. _ocdbt-btree-interior-node-data-file-length:

``data_file_length[i]``
  Specifies the byte length within ``data_file_id[i]`` of the encoded child
  B+tree node.

.. _ocdbt-btree-interior-node-num-keys:

``num_keys[i]``
  Specifies the total number of (leaf-node) keys within the subtree rooted
  at the child node.  Note that if there is more than one path from the
  child node to a given leaf node, the leaf node's keys are counted more
  than once.

.. _ocdbt-btree-interior-node-num-tree-bytes:

``num_tree_bytes[i]``
  Specifies the total encoded size in bytes of all B+tree nodes reachable
  from the child, including the child itself.  A given node is counted once
  for each unique path within the tree to it; if there is more than one
  path to a node, its size is counted multiple times.

.. _ocdbt-btree-interior-node-num-indirect-value-bytes:

``num_indirect_value_bytes[i]``
  Specifies the total size in bytes of all indirectly-stored values in the
  subtree rooted at the child node.  If the same stored value is referenced
  from multiple keys, its size is counted multiple times.

.. _ocdbt-btree-footer:

B+tree node footer
~~~~~~~~~~~~~~~~~~

+----------------------------------+---------------+
|Field                             |Binary format  |
+==================================+===============+
|:ref:`ocdbt-btree-crc32c-checksum`||crc32c_format||
+----------------------------------+---------------+

.. _ocdbt-btree-crc32c-checksum:

``crc32c_checksum``
  CRC-32C checksum of the entire B+tree node, excluding the checksum itself.
