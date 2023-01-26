.. _python-indexing:

Indexing
========

`tensorstore.TensorStore` (and objects of other
`tensorstore.Indexable` types) support a common set of *indexing
operations* for read/write access to individual positions and subsets
of positions.  In addition to full support for `NumPy-style basic and
advanced indexing<python-numpy-style-indexing>`, `dimension expressions
<python-dim-expressions>` provide additional indexing capabilities
integrated with TensorStore's support for :ref:`labeled/named
dimensions<dimension-labels>` and non-zero origins.

.. note::

   In TensorStore, all indexing operations result in a (read/write)
   *view* of the original object, represented as a new object of the
   same type with a different `tensorstore.IndexDomain`.  Indexing
   operations never implicitly perform I/O or copy data.  This differs
   from `NumPy indexing<numpy:basics.indexing>`, where basic
   indexing results in a view of the original data, but advanced
   indexing always results in a copy.


Index transforms
----------------

Indexing operations are composed into a normalized representation via the
:py:obj:`tensorstore.IndexTransform` class, which represents an :ref:`index
transform<index-transform>` from an input space to an output space. The
examples below may include the :ref:`index transform<index-transform>`
representation.

.. _python-numpy-style-indexing:

NumPy-style indexing
--------------------

NumPy-style indexing is performed using the syntax
:python:`obj[expr]`, where :python:`obj` is any `tensorstore.Indexable` object
and the indexing expression :python:`expr` is one of:

.. list-table::
   :widths: auto

   * - an integer;
     - `python-indexing-integer`
   * - a `slice` object :python:`start:stop:step`, e.g. :python:`obj[:]` or
       :python:`obj[3:5]` or :python:`obj[1:7:2]`, where the :python:`start`,
       :python:`stop`, or :python:`step` values are each `None`, integers or
       sequences of integer or `None` values;
     - `python-indexing-interval`
   * - :py:obj:`tensorstore.newaxis` or :py:obj:`None`;
     - `python-indexing-newaxis`
   * - :python:`...` or :py:obj:`Ellipsis`;
     - `python-indexing-ellipsis`
   * - `array_like` with integer data type;
     - `python-indexing-integer-array`
   * - `array_like` with `bool` data type;
     - `python-indexing-boolean-array`
   * - :py:obj:`tuple` of any of the above, e.g. :python:`obj[1, 2, :, 3]` or
       :python:`obj[1, ..., :, [0, 2, 3]]`.
     -

This form of indexing always operates on a prefix of the dimensions,
consuming dimensions from the existing domain and adding dimensions to
the resultant domain in order; if the indexing expression consumes
fewer than :python:`obj.rank` dimensions, the remaining dimensions are
retained unchanged as if indexed by :python:`:`.

.. _python-indexing-integer:

Integer indexing
^^^^^^^^^^^^^^^^

Indexing with an integer selects a single position within the corresponding
dimension:

.. doctest::

   >>> a = ts.array([[0, 1, 2], [3, 4, 5]], dtype=ts.int32)
   >>> a[1]
   TensorStore({
     'array': [3, 4, 5],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })
   >>> a[1, 2]
   TensorStore({
     'array': 5,
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_rank': 0},
   })

Each integer index consumes a single dimension from the original
domain and adds no dimensions to the result domain.

Because TensorStore supports index domains defined over negative
indices, negative values have no special meaning; they simply refer to
negative positions:

.. doctest::

   >>> a = await ts.open({
   ...     "dtype": "int32",
   ...     "driver": "array",
   ...     "array": [1, 2, 3],
   ...     "transform": {
   ...         "input_shape": [3],
   ...         "input_inclusive_min": [-10],
   ...         "output": [{
   ...             "input_dimension": 0,
   ...             "offset": 10
   ...         }],
   ...     },
   ... })
   >>> a[-10]
   TensorStore({
     'array': 1,
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_rank': 0},
   })

.. warning::

   This differs from the behavior of the built-in sequence types and
   `numpy.ndarray`, where a negative index specifies a position
   relative to the end (upper bound).

Specifying an index outside the explicit bounds of a dimension results in an
immediate error:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3], dtype=ts.int32)
   >>> a[4]
   Traceback (most recent call last):
       ...
   IndexError: Checking bounds of constant output index map for dimension 0: Index 4 is outside valid range [0, 4)

Specifying an index outside the `implicit bounds<implicit-bounds>` of
a dimension is permitted:

.. doctest::

   >>> a = ts.IndexTransform(input_shape=[4], implicit_lower_bounds=[True])
   >>> a[-1]
   Rank 0 -> 1 index space transform:
     Input domain:
     Output index maps:
       out[0] = -1
   >>> a[4]
   Traceback (most recent call last):
       ...
   IndexError: Checking bounds of constant output index map for dimension 0: Index 4 is outside valid range (-inf, 4)

While implicit bounds do not constrain indexing operations, the bounds
will still be checked by any subsequent read or write operation, which
will fail if any index is actually out of bounds.

.. note::

   In addition to the `int` type, integer indices may be specified
   using any object that supports the :python:`__index__` protocol
   (:pep:`357`), including :ref:`NumPy integer scalar types
   <numpy:arrays.scalars>`.

.. _python-indexing-interval:

Interval indexing
^^^^^^^^^^^^^^^^^

Indexing with a `slice` object :python:`start:stop:step` selects an
interval or strided interval within the corresponding dimension:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ts.int32)
   >>> a[1:5]
   TensorStore({
     'array': [1, 2, 3, 4],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [5],
       'input_inclusive_min': [1],
       'output': [{'input_dimension': 0, 'offset': -1}],
     },
   })

As for the built-in sequence types, the :python:`start` value is
inclusive while the :python:`stop` value is exclusive.

Each of :python:`start`, :python:`stop`, and :python:`step` may be an
integer, `None`, or omitted (equivalent to specifying `None`).
Specifying `None` for :python:`start` or :python:`stop` retains the
existing lower or upper bound, respectively, for the dimension.
Specifying `None` for :python:`step` is equivalent to specifying
:python:`1`.

When the :python:`step` is :python:`1`, the domain of the resulting
sliced dimension is *not* translated to have an origin of zero;
instead, it has an origin equal to the start position of the interval
(or the existing origin of the start position is unspecified):

.. doctest::

   >>> a = ts.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ts.int32)
   >>> a[1:5][2]
   TensorStore({
     'array': 2,
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_rank': 0},
   })

If the :python:`step` is not :python:`1`, the origin of the resulting
sliced dimension is equal to the :python:`start` position divided by
the :python:`step` value, rounded towards zero:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ts.int32)
   >>> a[3:8:2]
   TensorStore({
     'array': [3, 5, 7],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [4],
       'input_inclusive_min': [1],
       'output': [{'input_dimension': 0, 'offset': -1}],
     },
   })
   >>> a[7:3:-2]
   TensorStore({
     'array': [7, 5],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [-1],
       'input_inclusive_min': [-3],
       'output': [{'input_dimension': 0, 'offset': 3}],
     },
   })

It is an error to specify an interval outside the explicit bounds of a
dimension:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=ts.int32)
   >>> a[3:12]
   Traceback (most recent call last):
       ...
   IndexError: Computing interval slice for dimension 0: Slice interval [3, 12) is not contained within domain [0, 10)

.. warning::

   This behavior differs from that of the built-in sequence types and
   `numpy.ndarray`, where any out-of-bounds indices within the
   interval are silently skipped.

Specifying an interval outside the `implicit bounds<implicit-bounds>`
of a dimension is permitted:

.. doctest::

   >>> a = ts.IndexTransform(input_shape=[4], implicit_lower_bounds=[True])
   >>> a[-1:2]
   Rank 1 -> 1 index space transform:
     Input domain:
       0: [-1, 2)
     Output index maps:
       out[0] = 0 + 1 * in[0]

If a non-`None` value is specified for :python:`start` or
:python:`stop`, the lower or upper bound, respectively, of the
resultant dimension will be marked explicit.  If `None` is specified
for :python:`start` or :python:`stop`, the lower or upper bound,
respectively, of the resultant dimension will be marked explicit if
the corresponding original bound is marked explicit.

As with integer indexing, negative :python:`start` or :python:`stop`
values have no special meaning, and simply indicate negative positions.

Any of the :python:`start`, :python:`stop`, or :python:`stop` values
may be specified as a sequence of integer or `None` values (e.g. a
`list`, `tuple` or 1-d `numpy.ndarray`), rather than a single integer:

.. doctest::

   >>> a = ts.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
   ...              dtype=ts.int32)
   >>> a[(1, 1):(3, 4)]
   TensorStore({
     'array': [[6, 7, 8], [10, 11, 12]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [3, 4],
       'input_inclusive_min': [1, 1],
       'output': [
         {'input_dimension': 0, 'offset': -1},
         {'input_dimension': 1, 'offset': -1},
       ],
     },
   })

This is equivalent to specifying a sequence of `slice` objects:

   >>> a = ts.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
   ...              dtype=ts.int32)
   >>> a[1:3, 1:4]
   TensorStore({
     'array': [[6, 7, 8], [10, 11, 12]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [3, 4],
       'input_inclusive_min': [1, 1],
       'output': [
         {'input_dimension': 0, 'offset': -1},
         {'input_dimension': 1, 'offset': -1},
       ],
     },
   })

It is an error to specify a :py:obj:`slice` with sequences of unequal
lengths, but a sequence may be combined with a scalar value:

   >>> a = ts.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
   ...              dtype=ts.int32)
   >>> a[1:(3, 4)]
   TensorStore({
     'array': [[6, 7, 8], [10, 11, 12]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [3, 4],
       'input_inclusive_min': [1, 1],
       'output': [
         {'input_dimension': 0, 'offset': -1},
         {'input_dimension': 1, 'offset': -1},
       ],
     },
   })

.. _python-indexing-newaxis:

Adding singleton dimensions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specifying a value of :py:obj:`tensorstore.newaxis` (equal to `None`) adds a
new dummy/singleton dimension with `implicit bounds<implicit-bounds>`
:math:`[0, 1)`:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=2)
   >>> a[ts.newaxis]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: [0*, 1*)
       1: (-inf*, +inf*)
       2: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * in[2]

This indexing term consumes no dimensions from the original domain and
adds a single dimension after any dimensions added by prior indexing
operations:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=2)
   >>> a[:, ts.newaxis, ts.newaxis]
   Rank 4 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: [0*, 1*)
       2: [0*, 1*)
       3: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[3]

Because the added dimension has implicit bounds, it may be given
arbitrary bounds by a subsequent interval indexing term:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=2)
   >>> a[ts.newaxis][3:10]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: [3, 10)
       1: (-inf*, +inf*)
       2: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * in[2]

.. _python-indexing-ellipsis:

Ellipsis
^^^^^^^^

Specifying the special `Ellipsis` value (:python:`...`) is equivalent
to specifying as many full slices :python:`:` as needed to consume the
remaining dimensions of the original domin not consumed by other
indexing terms:

.. doctest::

   >>> a = ts.array([[[1, 2, 3], [4, 5, 6]]], dtype=ts.int32)
   >>> a[..., 1]
   TensorStore({
     'array': [2, 5],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [1, 2],
       'input_inclusive_min': [0, 0],
       'output': [{'input_dimension': 1}],
     },
   })

At most one `Ellipsis` may be specified within a single NumPy-style
indexing expression:

.. doctest::

   >>> a = ts.array([[[1, 2, 3], [4, 5, 6]]], dtype=ts.int32)
   >>> a[..., 1, ...]
   Traceback (most recent call last):
       ...
   IndexError: An index can only have a single ellipsis (`...`)

As a complete indexing expression , `Ellipsis` has no effect and is
equivalent to the empty tuple :python:`()`, but can still be useful
for the purpose of an assignment:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3], dtype=ts.int32)
   >>> a[...] = 7
   >>> a
   TensorStore({
     'array': [7, 7, 7, 7],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [4], 'input_inclusive_min': [0]},
   })

.. _python-indexing-integer-array:

Integer array indexing
^^^^^^^^^^^^^^^^^^^^^^

Specifying an `array_like` *index array* of integer values selects the
coordinates of the dimension given by the elements of the array:

.. doctest::

   >>> a = ts.array([5, 4, 3, 2], dtype=ts.int32)
   >>> a[[0, 3, 3]]
   TensorStore({
     'array': [5, 2, 2],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })
   >>> a[[[0, 1], [2, 3]]]
   TensorStore({
     'array': [[5, 4], [3, 2]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })

This indexing term consumes a single dimension from the original
domain, and when the full indexing expression involves just a single
array indexing term, adds the dimensions of the index array to the
result domain.

As with integer and interval indexing, and unlike NumPy, negative
values in an index array have no special meaning, and simply indicate
negative positions.

When a single indexing expression includes multiple index arrays,
*vectorized* array indexing semantics apply by default: the shapes of
the index arrays must all be broadcast-compatible, and the dimensions
of the single broadcasted domain are added to the result domain:

.. doctest::

   >>> a = ts.array([[1, 2], [3, 4], [5, 6]], dtype=ts.int32)
   >>> a[[0, 1, 2], [0, 1, 0]]
   TensorStore({
     'array': [1, 4, 5],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })
   >>> a[[[0, 1], [2, 2]], [[0, 1], [1, 0]]]
   TensorStore({
     'array': [[1, 4], [6, 5]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })
   >>> a[[[0, 1], [2, 2]], [0, 1]]
   TensorStore({
     'array': [[1, 4], [5, 6]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })

If all of the index arrays are applied to consecutive dimensions
without any interleaved `slice`, `Ellipsis`, or `tensorstore.newaxis`
terms (interleaved integer index terms are permitted), then by default
*legacy NumPy* semantics are used: the dimensions of the broadcasted
array domain are added *inline* to the result domain after any
dimensions added by prior indexing terms in the indexing expression:

.. doctest::

   >>> a = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
   >>> a[:, [1, 0], [1, 1]]
   TensorStore({
     'array': [[4, 2], [8, 6]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })

If there are any interleaved `slice`, `Ellipsis`, or
`tensorstore.newaxis` terms, then instead the dimensions of the
broadcasted array domain are added as the first dimensions of the
result domain:

.. doctest::

   >>> a = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
   >>> a[:, [1, 0], ts.newaxis, [1, 1]]
   TensorStore({
     'array': [[4, 8], [2, 6]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 2, [1]],
       'input_inclusive_min': [0, 0, [0]],
       'output': [{'input_dimension': 0}, {'input_dimension': 1}],
     },
   })

To ensure that the added array domain dimensions are added as the
first dimensions of the result domain regardless of whether there are
any interleaved `slice`, `Ellipsis`, or `tensorstore.newaxis` terms,
use the `vindex indexing method<python-vindex-indexing>`.

To instead perform *outer* array indexing, where each index array is
applied orthogonally, use the `oindex indexing
method<python-oindex-indexing>`.

.. note::

   The *legacy NumPy* indexing behavior, whereby array domain
   dimensions are added either *inline* or as the first dimensions
   depending on whether the index arrays are applied to consecutive
   dimensions, is the default behavior for compatibility with NumPy
   but may be confusing.  It is recommended to instead use either the
   `vindex<python-vindex-indexing>` or
   `oindex<python-oindex-indexing>` indexing method for less
   confusing behavior when using multiple index arrays.

.. _python-indexing-boolean-array:

Boolean array indexing
^^^^^^^^^^^^^^^^^^^^^^

Specifying an `array_like` of `bool` values is equivalent to
specifying a sequence of integer index arrays containing the
coordinates of `True` values (in C order), e.g. as obtained from
`numpy.nonzero`.

Specifying a 1-d `bool` array is equivalent to a single index array of the
non-zero coordinates:

.. doctest::

   >>> a = ts.array([0, 1, 2, 3, 4], dtype=ts.int32)
   >>> a[[True, False, True, True]]
   TensorStore({
     'array': [0, 2, 3],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })
   >>> # equivalent, using index array
   >>> a[[0, 2, 3]]
   TensorStore({
     'array': [0, 2, 3],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })

More generally, specifying an ``n``-dimensional `bool` array is equivalent to
specifying ``n`` index arrays, where the ``i``\ th index array specifies
the ``i``\ th coordinate of the `True` values:

.. doctest::

   >>> a = ts.array([[0, 1, 2], [3, 4, 5]], dtype=ts.int32)
   >>> a[[[True, False, False], [True, True, False]]]
   TensorStore({
     'array': [0, 3, 4],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })
   >>> # equivalent, using index arrays
   >>> a[[0, 1, 1], [0, 0, 1]]
   TensorStore({
     'array': [0, 3, 4],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })

This indexing term consumes ``n`` dimensions from the original domain,
where ``n`` is the rank of the `bool` array.

It is perfectly valid to mix boolean array indexing with other forms
of indexing, including integer array indexing, with exactly the same
result as if the boolean array were replaced by the equivalent
sequence of integer index arrays:

.. doctest::

   >>> a = ts.array([[0, 1, 2], [3, 4, 5], [7, 8, 9]], dtype=ts.int32)
   >>> a[[True, False, True], [2, 1]]
   TensorStore({
     'array': [2, 8],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2], 'input_inclusive_min': [0]},
   })
   >>> # equivalent, using index array
   >>> a[[0, 2], [2, 1]]
   TensorStore({
     'array': [2, 8],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2], 'input_inclusive_min': [0]},
   })

.. warning::

   Mixing boolean and integer index arrays in the default vectorized
   indexing mode, while supported for compatibility with NumPy, is
   likely to be confusing.  In most cases of mixed boolean and integer
   array indexing, `outer indexing mode<python-oindex-indexing>`
   provides more useful behavior.

The scalar values `True` and `False` are treated as zero-rank boolean
arrays.  Zero-rank boolean arrays are supported, but there is no
equivalent integer index array representation.  If there are no other
integer or boolean arrays, specifying a zero-rank boolean array is
equivalent to specifying `tensorstore.newaxis`, except that the added
dimension has explicit rather than implicit bounds, and in the case of
a `False` array the added dimension has the empty bounds of :math:`[0,
0)`:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=2)
   >>> a[:, True]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: [0, 1)
       2: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[2]
   >>> a[:, False]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: [0, 0)
       2: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[2]

If there are other integer or boolean arrays, specifying a zero-rank
boolean array has no effect except that:

1. the other index array shapes must be broadcast-compatible with the
   shape :python:`[0]` in the case of a `False` zero-rank array,
   meaning they are all empty arrays (in the case of a `True`
   zero-rank array, the other index array shapes must be
   broadcast-compatible with the shape :python:`[1]`, which is always
   satisfied);
2. in legacy NumPy indexing mode, if it is separated from another
   integer or boolean array term by a `slice`, `Ellipsis`, or
   `tensorstore.newaxis`, it causes the dimensions of the broadcast
   array domain to be added as the first dimensions of the result
   domain:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=2)
   >>> # Index array dimension added to result domain inline
   >>> a[:, True, [0, 1]]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: [0, 2)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{0, 1}}
   >>> a[:, False, []]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*)
       1: [0, 0)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0
   >>> # Index array dimensions added as first dimension of result domain
   >>> a[True, :, [0, 1]]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: [0, 2)
       1: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
         {{0}, {1}}
   >>> a[False, :, []]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: [0, 0)
       1: (-inf*, +inf*)
     Output index maps:
       out[0] = 0 + 1 * in[1]
       out[1] = 0

.. note::

   Zero-rank boolean arrays are supported for consistency and for
   compatibility with NumPy, but are rarely useful.

.. _python-indexing-numpy-comparison:

Differences compared to NumPy indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TensorStore indexing has near-perfect compatibility with NumPy, but
there are a few differences to be aware of:

- Negative indices have no special meaning in TensorStore, and simply
  refer to negative positions.  TensorStore does not support an
  equivalent shortcut syntax to specify a position ``n`` relative to
  the upper bound of a dimension; instead, it must be specified
  explicitly, e.g. :python:`x[x.domain[0].exclusive_max - n]`.

- In TensorStore, out-of-bounds intervals specified by a `slice`
  result in an error.  In NumPy, out-of-bounds indices specified by a
  `slice` are silently truncated.

- To specify a sequence of indexing terms when using the syntax
  :python:`obj[expr]` in TensorStore, :python:`expr` must be a `tuple`. In
  NumPy, for compatibility with its predecessor library *Numeric*, if
  :python:`expr` is a `list` or other non-`numpy.ndarray` sequence type
  containing at least one `slice`, `Ellipsis`, or `None` value, it is
  interpreted the same as a `tuple` :ref:`(this behavior is deprecated
  in NumPy since version 1.15.0)<numpy:arrays.indexing>`.  TensorStore, in
  contrast, will attempt to convert any non-`tuple` sequence to an integer
  or boolean array, which results in an error if the sequence contains a
  `slice`, `Ellipsis`, or `None` value.

.. _python-vindex-indexing:

Vectorized indexing mode (:python:`vindex`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The expression :python:`obj.vindex[expr]`, where :python:`obj` is any
`tensorstore.Indexable` object and :python:`expr` is a valid
`NumPy-style indexing expression<python-numpy-style-indexing>`, has a
similar effect to :python:`obj[expr]` except that if :python:`expr`
specifies any array indexing terms, the broadcasted array dimensions
are unconditionally added as the first dimensions of the result
domain:

.. doctest::

   >>> a = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
   >>> a.vindex[:, [1, 0], [1, 1]]
   TensorStore({
     'array': [[4, 8], [2, 6]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })

This avoids the potentially-confusing behavior of the default legacy
NumPy semantics, under which the broadcasted array dimensions are
added inline to the result domain if none of the array indexing terms
are separated by a `slice`, `Ellipsis`, or `tensorstore.newaxis` term.

.. note::

   If :python:`expr` does not include any array indexing terms,
   :python:`obj.vindex[expr]` is exactly equivalent to
   :python:`obj[expr]`.

This indexing method is similar to the behavior of:

- `dask:dask.array.Array.vindex`,
- `zarr.core.Array.vindex<zarr:zarr.core.Array>`, and
- the proposed :python:`vindex` in `NumPy Enhancement Proposal 21
  <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_.

.. _python-oindex-indexing:

Outer indexing mode (:python:`oindex`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The expression :python:`obj.oindex[expr]`, where :python:`obj` is any
`tensorstore.Indexable` object and :python:`expr` is a valid
`NumPy-style indexing expression<python-numpy-style-indexing>`,
performs *outer*/*orthogonal* indexing.  The effect is similar to
:python:`obj[expr]`, but differs in that any integer or boolean array
indexing terms are applied orthogonally:

.. doctest::

   >>> a = ts.array([[0, 1, 2], [3, 4, 5]], dtype=ts.int32)
   >>> a.oindex[[0, 0, 1], [1, 2]]
   TensorStore({
     'array': [[1, 2], [1, 2], [4, 5]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3, 2], 'input_inclusive_min': [0, 0]},
   })
   >>> # equivalent, using boolean array
   >>> a.oindex[[0, 0, 1], [False, True, True]]
   TensorStore({
     'array': [[1, 2], [1, 2], [4, 5]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3, 2], 'input_inclusive_min': [0, 0]},
   })

Unlike in the `default<python-indexing-integer-array>` or
the `vindex<python-vindex-indexing>` indexing modes, the index array
shapes need not be broadcast-compatible; instead, the dimensions of
each index array (or the 1-d index array equivalent of a boolean
array) are added to the result domain immediately after any dimensions
added by the previous indexing terms:

.. doctest::

   >>> a = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
   >>> a.oindex[[1, 0], :, [0, 0, 1]]
   TensorStore({
     'array': [[[5, 5, 6], [7, 7, 8]], [[1, 1, 2], [3, 3, 4]]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 2, 3],
       'input_inclusive_min': [0, 0, 0],
     },
   })

Each boolean array indexing term adds a single dimension to the result
domain:

.. doctest::

   >>> a = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
   >>> a.oindex[[[True, False], [False, True]], [1, 0]]
   TensorStore({
     'array': [[2, 1], [8, 7]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 2], 'input_inclusive_min': [0, 0]},
   })

.. note::

   If :python:`expr` does not include any array indexing terms,
   :python:`obj.oindex[expr]` is exactly equivalent to
   :python:`obj[expr]`.

This indexing method is similar to the behavior of:

- `zarr.core.Array.oindex<zarr:zarr.core.Array>`, and
- the proposed :python:`oindex` in `NumPy Enhancement Proposal 21
  <https://numpy.org/neps/nep-0021-advanced-indexing.html>`_.

.. _python-dim-expressions:

Dimension expressions
---------------------

*Dimension expressions* provide an alternative indexing mechanism to
`python-numpy-style-indexing` that is more powerful and expressive and
supports `dimension labels<dimension-labels>` (but can be more
verbose):

The usual syntax for applying a dimension expression is:
:python:`obj[ts.d[sel] op1 ... opN]`, where :python:`obj` is any
`tensorstore.Indexable` object, :python:`sel` specifies the initial
`dimension selection <python-dim-selections>` and :python:`op1
... opN` specifies a chain of one or more
`operations<python-dim-expression-construction>` supported by
`tensorstore.DimExpression` (the :python:`...` in :python:`op1
... opN` is not a literal Python `Ellipsis` (:python:`...`), but
simply denotes a sequence of operation invocations).

The `tensorstore.DimExpression` object itself, constructed using the
syntax :python:`ts.d[sel] op1 ... opN` is simply a lightweight,
immutable representation of the sequence of operations and their
arguments, and performs only minimal validation upon construction;
full validation is deferred until it is actually applied to an
`tensorstore.Indexable` object, using the syntax
:python:`obj[ts.d[sel] op1 ... opN]`.

.. doctest::

   >>> a = ts.array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]],
   ...              dtype=ts.int32)
   >>> # Label the dimensions "x", "y", "z"
   >>> a = a[ts.d[:].label["x", "y", "z"]]
   >>> a
   TensorStore({
     'array': [[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 3, 2],
       'input_inclusive_min': [0, 0, 0],
       'input_labels': ['x', 'y', 'z'],
     },
   })
   >>> # Select the y=1, x=0 slice
   >>> a[ts.d["y", "x"][1, 0]]
   TensorStore({
     'array': [2, 3],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2],
       'input_inclusive_min': [0],
       'input_labels': ['z'],
     },
   })


Operations
^^^^^^^^^^

Dimension expressions provide the following advanced operations:

:py:obj:`~tensorstore.DimExpression.label`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sets (or changes) the labels of the selected dimensions.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a = a[ts.d[:].label["x", "y"]]
   >>> a
   TensorStore({
     'array': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [3, 4],
       'input_inclusive_min': [0, 0],
       'input_labels': ['x', 'y'],
     },
   })
   >>> # Select the x=1 slice
   >>> a[ts.d["x"][1]]
   TensorStore({
     'array': [4, 5, 6, 7],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [4],
       'input_inclusive_min': [0],
       'input_labels': ['y'],
     },
   })

:py:obj:`~tensorstore.DimExpression.diagonal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts the diagonal of the selected dimensions.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[:].diagonal]
   TensorStore({
     'array': [0, 5, 10],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })

:py:obj:`~tensorstore.DimExpression.translate_to`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates the domains of the selected input dimensions to the specified
origins without affecting the output range.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a.origin
   (0, 0)
   >>> a[ts.d[:].translate_to[1]].origin
   (1, 1)
   >>> a[ts.d[:].translate_to[1, 2]].origin
   (1, 2)

:py:obj:`~tensorstore.DimExpression.translate_by`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates (shifts) the domains of the selected input dimensions by the
specified offsets, without affecting the output range.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[:].translate_by[-1, 1]].origin
   (-1, 1)

:py:obj:`~tensorstore.DimExpression.translate_backward_by`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Translates (shifts) the domains of the selected input dimensions backward by
the specified offsets, without affecting the output range.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[:].translate_backward_by[-1, 1]].origin
   (1, -1)

:py:obj:`~tensorstore.DimExpression.stride`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strides the domains of the selected input dimensions by the specified amounts.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[1].stride[2]]
   TensorStore({
     'array': [[0, 2], [4, 6], [8, 10]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3, 2], 'input_inclusive_min': [0, 0]},
   })

:py:obj:`~tensorstore.DimExpression.transpose`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transposes the selected dimensions to the specified target indices.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a = a[ts.d[:].label["x", "y"]]
   >>> a[ts.d[1].transpose[0]]
   TensorStore({
     'array': [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [4, 3],
       'input_inclusive_min': [0, 0],
       'input_labels': ['y', 'x'],
     },
   })
   >>> a[ts.d[:].transpose[::-1]]
   TensorStore({
     'array': [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [4, 3],
       'input_inclusive_min': [0, 0],
       'input_labels': ['y', 'x'],
     },
   })

:py:obj:`~tensorstore.DimExpression.oindex`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a NumPy-style indexing operation with outer indexing semantics.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[:].oindex[(2, 2), (0, 1, 3)]]
   TensorStore({
     'array': [[8, 9, 11], [8, 9, 11]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [2, 3], 'input_inclusive_min': [0, 0]},
   })

:py:obj:`~tensorstore.DimExpression.vindex`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies a NumPy-style indexing operation with vectorized indexing semantics.

.. doctest::

   >>> a = ts.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
   ...              dtype=ts.int32)
   >>> a[ts.d[:].vindex[(1, 0, 2), (0, 1, 3)]]
   TensorStore({
     'array': [4, 1, 11],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
   })


Composed examples
^^^^^^^^^^^^^^^^^

Composing dimension expressions enables constructing more complex indexing
operations than are easily done with native syntax.

.. doctest::

   >>> a = ts.array([[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]],
   ...              dtype=ts.int32)[ts.d[:].label["x", "y", "z"]]
   >>> # Transpose "x" and "z"
   >>> a[ts.d["x", "z"].transpose[2, 0]]
   TensorStore({
     'array': [[[0, 6], [2, 8], [4, 10]], [[1, 7], [3, 9], [5, 11]]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 3, 2],
       'input_inclusive_min': [0, 0, 0],
       'input_labels': ['z', 'y', 'x'],
     },
   })
   >>> # Select the x=d, y=d diagonal, and transpose "d" to end
   >>> a[ts.d["x", "y"].diagonal.label["d"].transpose[-1]]
   TensorStore({
     'array': [[0, 8], [1, 9]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 2],
       'input_inclusive_min': [0, 0],
       'input_labels': ['z', 'd'],
     },
   })
   >>> # Slice z=0, apply outer indexing to "x" and "y", label as "a", "b"
   >>> a[ts.d["z", "x", "y"].oindex[0, [0, 1], [2, 1]].label["a", "b"]]
   TensorStore({
     'array': [[4, 2], [10, 8]],
     'context': {'data_copy_concurrency': {}},
     'driver': 'array',
     'dtype': 'int32',
     'transform': {
       'input_exclusive_max': [2, 2],
       'input_inclusive_min': [0, 0],
       'input_labels': ['a', 'b'],
     },
   })


.. _python-dim-selections:

Dimension selections
^^^^^^^^^^^^^^^^^^^^

A dimension selection is specified using the syntax :python:`ts.d[sel]`, where
:python:`sel` is one of:

- an integer, specifying an existing or new dimension by index (as with
  built-in sequence types, negative numbers specify a dimension index relative
  to the end);

- a non-empty `str`, specifying an existing dimension by label;

- a `slice` object, :python:`start:stop:step`, where :python:`start`,
  :python:`stop`, and :python:`step` are either integers or `None`,
  specifying a range of existing or new dimensions by index (as for built-in
  sequence types, negative numbers specify a dimension index relative to the
  end);

- any sequence (including a `tuple`, `list`, or another `tensorstore.d` object)
  of any of the above.

The result is a `tensorstore.d` object, which is simply a lightweight, immutable
container representing the flattened sequence of `int`, `str`, or `slice`
objects:

.. doctest::

   >>> ts.d[0, 1, 2]
   d[0,1,2]
   >>> ts.d[0:1, 2, "x"]
   d[0:1,2,'x']
   >>> ts.d[[0, 1], [2]]
   d[0,1,2]
   >>> ts.d[[0, 1], ts.d[2, 3]]
   d[0,1,2,3]

A `str` label always identifies an existing dimension, and is only
compatible with operations/terms that expect an existing dimension:

.. doctest::

   >>> a = ts.IndexTransform(input_labels=['x'])
   >>> a[ts.d["x"][2:3]]
   Rank 1 -> 1 index space transform:
     Input domain:
       0: [2, 3) "x"
     Output index maps:
       out[0] = 0 + 1 * in[0]

An integer may identify either an existing or new dimension depending
on whether it is used with a `tensorstore.newaxis` term:

.. doctest::

   >>> a = ts.IndexTransform(input_labels=['x', 'y'])
   >>> # `1` refers to existing dimension "y"
   >>> a[ts.d[1][2:3]]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: [2, 3) "y"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
   >>> # `1` refers to new singleton dimension
   >>> a[ts.d[1][ts.newaxis]]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: [0*, 1*)
       2: (-inf*, +inf*) "y"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[2]

A negative dimension index :python:`-i` is equivalent to :python:`n -
i`, where ``n`` is the *sum* of the rank of the original domain *plus*
the number of `tensorstore.newaxis` terms:

.. doctest::

   >>> a = ts.IndexTransform(input_labels=['x', 'y'])
   >>> # `-1` is equivalent to 1, refers to existing dimension "y"
   >>> a[ts.d[-1][2:3]]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: [2, 3) "y"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
   >>> # `-1` is equivalent to 2, refers to new singleton dimension
   >>> a[ts.d[-1][ts.newaxis]]
   Rank 3 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: (-inf*, +inf*) "y"
       2: [0*, 1*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]

Likewise, a `slice` may identify either existing or new dimensions:

.. doctest::

   >>> a = ts.IndexTransform(input_labels=['x', 'y', 'z'])
   >>> # `:2` refers to existing dimensions "x", "y"
   >>> a[ts.d[:2][1:2, 3:4]]
   Rank 3 -> 3 index space transform:
     Input domain:
       0: [1, 2) "x"
       1: [3, 4) "y"
       2: (-inf*, +inf*) "z"
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0 + 1 * in[1]
       out[2] = 0 + 1 * in[2]
   >>> # `:2` refers to two new singleton dimensions
   >>> a[ts.d[:2][ts.newaxis, ts.newaxis]]
   Rank 5 -> 3 index space transform:
     Input domain:
       0: [0*, 1*)
       1: [0*, 1*)
       2: (-inf*, +inf*) "x"
       3: (-inf*, +inf*) "y"
       4: (-inf*, +inf*) "z"
     Output index maps:
       out[0] = 0 + 1 * in[2]
       out[1] = 0 + 1 * in[3]
       out[2] = 0 + 1 * in[4]

If a `tensorstore.newaxis` term is mixed with a term that consumes an
existing dimension, any dimension indices specified in the dimension
selection (either directly or via `slice` objects) are with respect to
an *intermediate* domain with any new singleton dimensions inserted
but no existing dimensions consumed:

   >>> a = ts.IndexTransform(input_labels=['x', 'y'])
   >>> # `1` refers to new singleton dimension, `2` refers to "y"
   >>> # intermediate domain is: {0: "x", 1: "", 2: "y"}
   >>> a[ts.d[1, 2][ts.newaxis, 0]]
   Rank 2 -> 2 index space transform:
     Input domain:
       0: (-inf*, +inf*) "x"
       1: [0*, 1*)
     Output index maps:
       out[0] = 0 + 1 * in[0]
       out[1] = 0

.. _python-dim-expression-construction:

Dimension expression construction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A `tensorstore.DimExpression` that applies a given operation to an
initial dimension selection :python:`dexpr = ts.d[sel]` is constructed using:

- subscript syntax :python:`dexpr[iexpr]` (for `NumPy-style indexing<python-dim-expression-numpy-indexing>`);
- attribute syntax :python:`dexpr.diagonal` for operations that take no arguments; or
- attribute subscript syntax :python:`dexpr.label[arg]`.

The same syntax may also be used to chain additional operations onto
an existing `tensorstore.DimExpression`:

.. doctest::

   >>> a = ts.IndexTransform(input_rank=0)
   >>> a[ts.d[0][ts.newaxis][1:10].label['z']]
   Rank 1 -> 0 index space transform:
     Input domain:
       0: [1, 10) "z"
     Output index maps:

When a `tensorstore.DimExpression` :python:`dexpr` is applied to a
`tensorstore.Indexable` object :python:`obj`, using the syntax
:python:`obj[dexpr]`, the following steps occur:

1. The initial dimension selection specified in :python:`dexpr` is
   resolved based on the domain of :python:`obj` and the first
   operation of :python:`dexpr`.
2. The first operation specified in :python:`dexpr` is applied to
   :python:`obj` using the resolved initial dimension selection.  This results
   in a new `tensorstore.Indexable` object of the same type as
   :python:`obj` and a new dimension selection consisting of the dimensions
   retained from the prior dimension selection or added by the operation.
3. Each subsequent operation, is applied, in order, to the new
   `tensorstore.Indexable` object and new dimension selection produced
   by each prior operation.

.. _python-dim-expression-numpy-indexing:

NumPy-style dimension expression indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The syntax :python:`dexpr[iexpr]`, :python:`dexpr.vindex[iexpr]`, and
:python:`dexpr.oindex[iexpr]` chains a NumPy-style indexing operation to an
existing `tensorstore.d` or `tensorstore.DimExpression`.

The behavior is similar to that of regular `NumPy-style
indexing<python-numpy-style-indexing>` applied directly to a
`tensorstore.Indexable` object, with the following differences:

- The terms of the indexing expression :python:`iexpr` consume
  dimensions in order from the dimension selection rather than
  starting from the first dimension of the domain, and unless an
  `Ellipsis` (:python:`...`) term is specified, :python:`iexpr` must
  include a sufficient number of indexing terms to consume the entire
  dimension selection.

- `tensorstore.newaxis` terms are only permitted in the first
  operation of a dimension expression, since in subsequent operations
  all dimensions of the dimension selection necessarily refer to
  existing dimensions.  Additionally, the dimension selection must
  specify the index of the new dimension for each
  `tensorstore.newaxis` term.

- If :python:`iexpr` is a *scalar* indexing expression that consists of a:

  - single integer,
  - `slice` :python:`start:stop:step` where :python:`start`, :python:`stop`,
    and :python:`step` are integers or `None`, or
  - `tensorstore.newaxis` term,

  it may be used with a dimension selection of more than one
  dimension, in which case :python:`iexpr` is implicitly duplicated to
  match the number of dimensions in the dimension selection:

  .. doctest::

     >>> a = ts.IndexTransform(input_labels=["x", "y"])
     >>> # add singleton dimension to beginning and end
     >>> a[ts.d[0, -1][ts.newaxis]]
     Rank 4 -> 2 index space transform:
       Input domain:
         0: [0*, 1*)
         1: (-inf*, +inf*) "x"
         2: (-inf*, +inf*) "y"
         3: [0*, 1*)
       Output index maps:
         out[0] = 0 + 1 * in[1]
         out[1] = 0 + 1 * in[2]
     >>> # slice out square region
     >>> a[ts.d[:][0:10]]
     Rank 2 -> 2 index space transform:
       Input domain:
         0: [0, 10) "x"
         1: [0, 10) "y"
       Output index maps:
         out[0] = 0 + 1 * in[0]
         out[1] = 0 + 1 * in[1]

- When using the default indexing mode, i.e. :python:`dexpr[iexpr]`, if more
  than one array indexing term is specified (even if they are
  consecutive), the array dimensions are always added as the first
  dimensions of the result domain (as if :python:`dexpr.vindex[iexpr]`
  were specified).

- When using outer indexing mode, i.e. :python:`dexpr.oindex[iexpr]`,
  zero-rank boolean arrays are not permitted.

