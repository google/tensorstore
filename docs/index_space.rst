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

.. list-table::
   :widths: auto

   * - constant
     - .. math::

          \mathrm{output}[j] = \mathrm{offset},

       where :math:`\mathrm{offset}` is an arbitrary 64-bit integer.
   * - single input dimension
     - .. math::

          \mathrm{output}[j] = \mathrm{offset} + \mathrm{stride} \cdot \mathrm{input}[\mathrm{input\_dimension}],

       where :math:`\mathrm{offset}` and :math:`\mathrm{stride}` are arbitrary 64-bit integers and
       :math:`\mathrm{input\_dimension}` is in the range :math:`[0, m)`.
   * - index array
     - .. math::

          \mathrm{output}[j] = \mathrm{offset} + \mathrm{stride} \cdot \mathrm{index\_array}[\mathrm{input}],

       where :math:`\mathrm{offset}` and :math:`\mathrm{stride}` are
       arbitrary 64-bit integers and :math:`\mathrm{index\_array}` is
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
