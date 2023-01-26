.. _index-space:

Index space
===========

TensorStore defines an *index space* as an :math:`n`-dimensional
space, where :math:`n` is the *rank* of the space and each
index is an integer in the closed interval :math:`[-(2^{62}-2),
+(2^{62}-2)]`.  The special values :math:`\pm (2^{62}-1)` are not valid
indices but represent bounds of :math:`\pm \infty`.

.. note::

   The limits of :math:`\pm (2^{62}-2)` permit :math:`\pm (2^{62}-1)` to be
   reserved to represent :math:`\pm \infty` and still allow the
   difference between any two bounds to be represented as a signed
   64-bit integer.

TensorStore currently supports ranks :math:`n` less than or equal to 32, which
is the same constraint imposed by NumPy.

.. _dimension-labels:

Each of the :math:`n` dimensions may optionally be identified by a unique
string *label*, such as ``"x"``, ``"y"``, or ``"z"``.  Unlabeled
dimensions are indicated by an empty string as the label.

.. _implicit-bounds:
.. _index-domain:

Index domain
------------

An *index domain* is a rectangular region of an index space.

- The inclusive lower bound of each dimensions is specified as an
  integer in :math:`[-(2^{62}-1), 2^{62}-2]`, where the special value of
  :math:`-2^{62}-1` indicates the dimension is unbounded below.
- The inclusive upper bound of each dimension is specified as an
  integer in :math:`[-(2^{62}-2), 2^{62}-1]`, where the special value of
  :math:`2^{62}-1` indicates the dimension is unbounded above.
- The lower and upper bound of each dimension is marked either
  *explicit* or *implicit* (represented as two additional bits per
  dimension).  Explicit bounds indicate hard constraints on the valid
  indices for that dimension.  Implicit bounds may be used for
  resizable dimensions of arrays: the implicit bound indicates the
  bound as of a certain time, but the bound may change if a resizable
  operation is performed, and indexing operations are not constrained
  by implicit bounds.  Implicit bounds of :math:`\pm \infty` are also
  used to indicate unknown or unspecified bounds.

.. note::

   Specifying infinite bounds :math:`(-\infty, +\infty)` for a
   dimension is similar to specifying the finite bounds
   :math:`[-(2^{62}-2, +(2^{62}-2)]`, in that in both cases the domain
   contains the full range of possible indices.  The difference is
   that translating a dimension with infinite bounds has no effect,
   while translating a dimension with bounds :math:`[-(2^{62}-2,
   +(2^{62}-2)]` by any non-zero offset results in an out-of-bounds
   error.

.. json:schema:: IndexDomain

.. _index-transform:

Index transform
---------------

An *index transform* from rank :math:`m` to rank :math:`n` maps every
:math:`m`-dimensional index vector in a rank-:math:`m` :ref:`index
domain<index-domain>` to an :math:`n`-dimensional index vector in the
output space.

It is defined by its input index domain and :math:`n` *output index
maps*, one for each dimension :math:`j` of the output space, each of
one of the following three forms:

.. table:: Output index methods
   :name: output-index-methods

   .. list-table::
      :widths: auto

      * - .. _index-transform-constant-map:

          constant
        - .. math::

             \mathtt{output}[j] = \mathtt{offset},

          where :math:`\mathtt{offset}` is an arbitrary 64-bit integer.
      * - .. _index-transform-single-input-dimension-map:

          single input dimension
        - .. math::

             \mathtt{output}[j] = \mathtt{offset} + \mathtt{stride} \cdot \mathtt{input}[\mathtt{input\_dimension}],

          where :math:`\mathtt{offset}` and :math:`\mathtt{stride}` are arbitrary
          64-bit integers and :math:`\mathtt{input\_dimension}` is in the range
          :math:`[0, m)`.
      * - .. _index-transform-array-map:

          index array
        - .. math::

             \mathtt{output}[j] = \mathtt{offset} + \mathtt{stride} \cdot \mathtt{index\_array}[\mathtt{input}],

          where :math:`\mathtt{offset}` and :math:`\mathtt{stride}` are
          arbitrary 64-bit integers and :math:`\mathtt{index\_array}` is
          an :math:`n`-dimensional array of 64-bit integers indexed by a
          subset of the dimensions of the input index domain with
          explicit lower and upper bounds, stored as a strided array in
          memory.  (The dimensions by which it is indexed are indicated
          by non-zero strides.)

TensorStore uses this normalized index transform representation to
represent any composition of indexing operations.  This representation
is complete, since *any* mapping can be represented as an index array,
but in many cases just the more efficient constant and single input
dimension maps suffice.

In most cases, two index transforms can be composed with low cost that
is independent of the bounds specified in the domains: single input
dimension maps may be composed with any other map without needing to
copying any index arrays.  Only when composing two index array maps is
it necessary to write a new index array, and in some cases this can
result in index array with a larger representation size.

The following table describes the relationship between the input and output
spaces of a transform, and the source and target arrays/TensorStores for read
and write operations:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Operation
     - Input domain
     - Output range
   * - Read
     - Corresponds to the target array.
     - Indicates the positions of the underlying source TensorStore that are
       accessed.
   * - Write
     - Corresponds to the source array.
     - Indicates the positions of the underlying target TensorStore that are
       modified.

.. json:schema:: IndexTransform

.. json:schema:: OutputIndexMap

.. json:schema:: IndexInterval

.. _index-domain-alignment:

Alignment and broadcasting
--------------------------

Many operations in TensorStore involving two :ref:`index domains<index-domain>`,
such as read, write, and copy operations, automatically *align* the ``source``
domain to the ``target`` domain.

The following alignment methods are supported (by default, all alignment methods
are used):

permute
    Source dimensions are permuted based on their labels in order to align the
    source domain to the target domain.

translate
    Source dimensions are translated in order to align the source domain to the
    target.

broadcast
    Source dimensions of size 1 do not have to match a target dimension, and not
    all target dimensions must match a source dimension.



Alignment is performed based on the following rules:

First, a subset of the ``source`` dimensions are matched to a subset of the
``target`` dimensions, according to one of two cases:

.. list-table::
   :widths: auto

   * - M1
     - At least one of ``source`` or ``target`` is entirely unlabeled (all
       dimension labels are empty).  In this case, the last
       :math:`\mathtt{match\_rank} = \min(\mathtt{source\_rank},
       \mathtt{target\_rank})` dimensions of ``source`` match in order to the
       last :math:`\mathtt{match\_rank}` dimensions of ``target``,
       i.e. dimension :math:`\mathtt{source\_rank} - \mathtt{match\_rank} + i`
       of ``source`` matches to dimension :math:`\mathtt{target\_rank} -
       \mathtt{match\_rank} + i` of ``target``, for :math:`0 \leq i <
       \mathtt{match\_rank}`.  This case also applies if the **permute**
       alignment method is not permitted.

   * - M2
     - Both ``source`` and ``target`` have at least one labeled dimension.  In
       this case, dimensions of ``source`` and ``target`` with matching labels
       are matched.  Any remaining labeled dimensions remain unmatched.  The
       unlabeled dimensions of ``source`` are matched to the unlabeled
       dimensions of ``target`` using the same method as in case M1 (right to
       left).

The matching is then validated as follows:

.. list-table::
   :widths: auto

   * - V1
     - For each match between a dimension :math:`i` of ``source`` and a
       dimension :math:`j` of ``target``, if :math:`\mathtt{source\_shape}[i]
       \neq \mathtt{target\_shape}[j]`, the match is dropped.  Note that if
       :math:`\mathtt{source\_shape}[i] \neq 1`, this leads to an error in step
       V3.

   * - V2
     - If the **broadcast** alignment method is not permitted, it is an error
       for any source or target dimension to be unmatched.  (In this case, any
       matches dropped in step V1 result in an error.)

   * - V3
     - For every unmatched dimension :math:`i` of ``source``,
       :math:`\mathtt{source\_shape}[i]` must equal :math:`1`.

   * - V4
     - If the **translate** alignment method is not permitted, for each match
       between a dimension :math:`i` of ``source`` and a dimension :math:`j` of
       ``target``, it is an error if :math:`\mathtt{source\_origin}[i] \neq
       \mathtt{target\_origin}[j]`.

If matching succeeds, a new ``alignment`` transform with an (input) domain equal
to ``target`` and an output rank equal to :math:`\mathtt{source\_rank}` is
computed as follows:

.. list-table::
   :widths: auto

   * - A1
     - For each dimension :math:`j` of ``target`` with a matching dimension
       :math:`i` of ``source``, output dimension :math:`i` of ``alignment`` has
       a *single_input_dimension* map to input dimension :math:`j` with a stride
       of :math:`1` and offset of :math:`\mathtt{source\_origin}[i] -
       \mathtt{target\_origin}[j]`.

   * - A2
     - For every unmatched dimension :math:`i` of ``source``, output dimension
       :math:`i` of ``alignment`` is a *constant* map with an offset of
       :math:`\mathtt{source\_origin}[i]`.  (It must be the case that
       :math:`\mathtt{source\_shape}[i] = 1`.)

The ``alignment`` transform maps ``target`` positions to corresponding
``source`` positions; for example, when copying, each position of the ``target``
domain is assigned the value at the corresponding position of the ``source``
domain.  If the ``broadcast`` alignment method is used, the transform may map
the same ``source`` position to multiple ``target`` positions.

Examples:

* All unlabeled dimensions

  - source: ``[3, 7), [5, 6), [4, 10)``
  - target: ``[2, 6), [0, 4), [6, 12)``
  - alignment: rank :math:`3 \rightarrow 3`, with:

    .. math::

      \mathrm{source}[0] &= \mathrm{target}[0] + 1 \\
      \mathrm{source}[1] &= 5 \\
      \mathrm{source}[2] &= \mathrm{target}[2] - 2

* All labeled dimensions

  - source: ``"x": [3, 7), "y": [5, 6), "z": [4, 10)``
  - target: ``"z": [6, 12), "x": [4, 8), "y": [0, 4)``
  - alignment: rank :math:`3 \rightarrow 3`, with:

    .. math::

      \mathrm{source}[0] &= \mathrm{target}[1] - 1 \\
      \mathrm{source}[1] &= 5 \\
      \mathrm{source}[2] &= \mathrm{target}[0] - 2

* Partially labeled dimensions

  - source: ``"x": [3, 7), "y": [5, 6), "": [4, 10)``
  - target: ``"": [0, 10) "": [6, 12), "x": [4, 8), "y": [0, 4)``
  - alignment: rank :math:`4 \rightarrow 3`, with:

    .. math::

      \mathrm{source}[0] &= \mathrm{target}[2] - 1 \\
      \mathrm{source}[1] &= 5 \\
      \mathrm{source}[2] &= \mathrm{target}[1] - 2

* Mismatched labeled dimensions

  - source: ``"x": [3, 7), "y": [5, 6), "z": [4, 10)``
  - target: ``"z": [6, 12), "w": [4, 8), "y": [0, 4)``
  - ERROR: Unmatched source dimension 0 ``{"x": [3, 7)}``
    does not have a size of 1

.. note::

   The alignment behavior supported by TensorStore is fully compatible with
   `NumPy broadcasting<numpy:basics.broadcasting>` but additionally is extended
   to support non-zero origins and labeled dimensions.
