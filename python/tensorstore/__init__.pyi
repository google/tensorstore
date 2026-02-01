"""TensorStore is a library for reading and writing multi-dimensional arrays."""

from __future__ import annotations
import asyncio
import abc
import builtins
import collections.abc
import types
import typing
import numbers
import numpy
import numpy.typing
from . import ocdbt

T_contra = typing.TypeVar("T_contra", contravariant=True)
newaxis = None
"Alias for `None` used in :ref:`indexing expressions<python-indexing>` to specify a new singleton dimension.\n\nExample:\n\n    >>> transform = ts.IndexTransform(input_rank=3)\n    >>> transform[ts.newaxis, 5]\n    Rank 3 -> 3 index space transform:\n      Input domain:\n        0: [0*, 1*)\n        1: (-inf*, +inf*)\n        2: (-inf*, +inf*)\n      Output index maps:\n        out[0] = 5\n        out[1] = 0 + 1 * in[1]\n        out[2] = 0 + 1 * in[2]\n\nGroup:\n  Indexing\n"
inf: int
"Special constant equal to :math:`2^{62}-1` that indicates an unbounded :ref:`index domain<index-domain>`.\n\nExample:\n\n    >>> d = ts.Dim()\n    >>> d.inclusive_min\n    -4611686018427387903\n    >>> d.inclusive_max\n    4611686018427387903\n    >>> assert d.inclusive_min == -ts.inf\n    >>> assert d.inclusive_max == +ts.inf\n\nGroup:\n  Indexing\n"


class Indexable(metaclass=abc.ABCMeta):
    """Abstract base class for types that support :ref:`TensorStore indexing operations<python-indexing>`.

    Supported types are:

    - :py:class:`tensorstore.TensorStore`
    - :py:class:`tensorstore.Spec`
    - :py:class:`tensorstore.IndexTransform`

    Group:
      Indexing
    """


T = typing.TypeVar("T", covariant=True)
FutureLike = Future[T] | collections.abc.Awaitable[T] | T
"Generic type representing a possibly-asynchronous result.\n\nThe following types may be used where a :py:obj:`FutureLike[T]<.FutureLike>`\nvalue is expected:\n\n- an immediate value of type :python:`T`;\n- :py:class:`tensorstore.Future` that resolves to a value of type :python:`T`;\n- :ref:`coroutine<async>` that resolves to a value of type :python:`T`.\n\nGroup:\n  Asynchronous support\n"
Future.__type_params__ = (T,)
Promise.__type_params__ = (typing.TypeVar("T", contravariant=True),)
bool: dtype
"Boolean data type (0 or 1).  Corresponds to the :py:obj:`python:bool` type and ``numpy.bool_``.\n\nGroup:\n  Data types\n"
char: dtype
"Single byte, interpreted as an ASCII character.\n\nGroup:\n  Data types\n"
byte: dtype
"Single byte.\n\nGroup:\n  Data types\n"
int2: dtype
"2-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type, internally stored as its 8-bit signed integer equivalent (i.e. sign-extended). Corresponds to ``jax.numpy.int2``.\n\nGroup:\n  Data types\n"
int4: dtype
"4-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type, internally stored as its 8-bit signed integer equivalent (i.e. sign-extended). Corresponds to ``jax.numpy.int4``.\n\nGroup:\n  Data types\n"
int8: dtype
"8-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int8``.\n\nGroup:\n  Data types\n"
uint8: dtype
"8-bit unsigned integer.  Corresponds to ``numpy.uint8``.\n\nGroup:\n  Data types\n"
int16: dtype
"16-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int16``.\n\nGroup:\n  Data types\n"
uint16: dtype
"16-bit unsigned integer.  Corresponds to ``numpy.uint16``.\n\nGroup:\n  Data types\n"
int32: dtype
"32-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int32``.\n\nGroup:\n  Data types\n"
uint32: dtype
"32-bit unsigned integer.  Corresponds to ``numpy.uint32``.\n\nGroup:\n  Data types\n"
int64: dtype
"32-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int64``.\n\nGroup:\n  Data types\n"
uint64: dtype
"64-bit unsigned integer data type.  Corresponds to ``numpy.uint64``.\n\nGroup:\n  Data types\n"
float8_e3m4: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e3m4\n\nGroup:\n  Data types\n"
float8_e4m3fn: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e4m3fn\n\nGroup:\n  Data types\n"
float8_e4m3fnuz: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e4m3fnuz\n\nGroup:\n  Data types\n"
float8_e4m3b11fnuz: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e4m3b11fnuz\n\nGroup:\n  Data types\n"
float8_e5m2: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e5m2\n\nGroup:\n  Data types\n"
float8_e5m2fnuz: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float8_e5m2fnuz\n\nGroup:\n  Data types\n"
float4_e2m1fn: dtype
"8-bit floating-point data type.\n\nDetails in https://github.com/jax-ml/ml_dtypes#float4_e2m1fn\n\nGroup:\n  Data types\n"
float16: dtype
":wikipedia:`IEEE 754 binary16 <Half-precision_floating-point_format>` half-precision floating-point data type.  Correspond to ``numpy.float16``.\n\nGroup:\n  Data types\n"
bfloat16: dtype
':wikipedia:`bfloat16 floating-point <Bfloat16_floating-point_format>` data type.\n\nNumPy does not have built-in support for bfloat16.  As an extension, TensorStore\ndefines the :python:`tensorstore.bfloat16.dtype` NumPy data type (also available\nas :python:`numpy.dtype("bfloat16")`, as well as the corresponding\n:python:`tensorstore.bfloat16.type` :ref:`array scalar\ntype<numpy:arrays.scalars>`, and these types are guaranteed to interoperate with\n`TensorFlow <tensorflow.org>`_ and `JAX <https://github.com/google/jax>`_.\n\nGroup:\n  Data types\n'
float32: dtype
":wikipedia:`IEEE 754 binary32 <Single-precision_floating-point_format>` single-precision floating-point data type.  Corresponds to ``numpy.float32``.\n\nGroup:\n  Data types\n"
float64: dtype
":wikipedia:`IEEE 754 binary64 <Double-precision_floating-point_format>` double-precision floating-point data type.  Corresponds to ``numpy.float64``.\n\nGroup:\n  Data types\n"
complex64: dtype
"Complex number based on :py:obj:`.float32`.  Corresponds to ``numpy.complex64``.\n\nGroup:\n  Data types\n"
complex128: dtype
"Complex number based on :py:obj:`.float64`.  Corresponds to ``numpy.complex128``.\n\nGroup:\n  Data types\n"
string: dtype
"Variable-length byte string data type.  Corresponds to the Python :py:obj:`python:bytes` type.\n\nThere is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.\n\n.. note::\n\n   The :ref:`NumPy string types<numpy:string-dtype-note>`, while related, differ\n   in that they are fixed-length and null-terminated.\n\nGroup:\n  Data types\n"
ustring: dtype
"Variable-length Unicode string data type.  Corresponds to the Python :py:obj:`python:str` type.\n\nThere is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.\n\n.. note::\n\n   The :ref:`NumPy string types<numpy:string-dtype-note>`, while related, differ\n   in that they are fixed-length and null-terminated.\n\nGroup:\n  Data types\n"
json: dtype
"JSON data type.  Corresponds to an arbitrary Python JSON value.\n\nThere is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.\n\nGroup:\n  Data types\n"
RecheckCacheOption = builtins.bool | typing.Literal["open"] | builtins.float
'Determines under what circumstances cached data is revalidated.\n\n``True``\n  Revalidate cached data at every option.\n\n``False``\n  Assume cached data is always fresh and never revalidate.\n\n``"open"``\n  Revalidate cached data older than the time at which the TensorStore was\n  opened.\n\n:py:obj:`float`\n  Revalidate cached data older than the specified time in seconds since\n  the unix epoch.\n\nGroup:\n  Spec\n'
DownsampleMethod = typing.Literal["stride", "median", "mode", "mean", "min", "max"]
"Downsampling method for use by the :ref:`downsample driver<driver/downsample>`.\n\nRefer to the :json:schema:`JSON<DownsampleMethod>` documentation for\ndetails on each method.\n\nGroup:\n  Views\n"
NumpyIndexTerm = typing.Union[
    typing.SupportsIndex, "slice", None, types.EllipsisType, "numpy.typing.ArrayLike"
]
"Individual term in a `NumpyIndexingSpec`.\n\nRefer to the :ref:`NumPy-style indexing<python-numpy-style-indexing>`\ndocumentation for details.\n\nGroup:\n  Indexing\n"
NumpyIndexingSpec = typing.Union["NumpyIndexTerm", tuple["NumpyIndexTerm", ...]]
"NumPy-style indexing expression.\n\nRefer to the :ref:`NumPy-style indexing<python-numpy-style-indexing>`\ndocumentation for details.\n\nGroup:\n  Indexing\n\n"
DimSelectionLike = typing.Union[
    DimSelection,
    typing.SupportsIndex,
    str,
    builtins.bytes,
    "slice",
    collections.abc.Sequence["DimSelectionLike"],
]
"Subsequence of dimensions to select in a `DimSelection` object.\n\nRefer to the :ref:`dimension selections<python-dim-selections>`\ndocumentation for details.\n\nGroup:\n  Indexing\n"
DTypeLike = typing.Union["dtype", str, type, "numpy.dtype"]
'Value that may be converted to a TensorStore `dtype`.\n\nMay specify a data type by name, its corresponding scalar class type,\nor by its corresponding :py:obj:`NumPy dtype<numpy.dtype>`.\n\nExample:\n\n    >>> ts.dtype("int32")\n    dtype("int32")\n    >>> ts.dtype(np.int32)\n    dtype("int32")\n    >>> ts.dtype(bool)\n    dtype("bool")\n\nGroup:\n  Data types\n\n'
d: _DimSelection
"Subscriptable object for constructing a `DimSelection`."
__all__ = [
    "Batch",
    "ChunkLayout",
    "CodecSpec",
    "Context",
    "DTypeLike",
    "Dim",
    "DimExpression",
    "DimSelection",
    "DimSelectionLike",
    "DownsampleMethod",
    "Future",
    "FutureLike",
    "IndexDomain",
    "IndexTransform",
    "Indexable",
    "KvStore",
    "NumpyIndexTerm",
    "NumpyIndexingSpec",
    "OpenMode",
    "OutputIndexMap",
    "OutputIndexMaps",
    "OutputIndexMethod",
    "Promise",
    "RecheckCacheOption",
    "Schema",
    "Spec",
    "T",
    "TensorStore",
    "Transaction",
    "Unit",
    "VirtualChunkedReadParameters",
    "VirtualChunkedWriteParameters",
    "WriteFutures",
    "array",
    "bfloat16",
    "bool",
    "byte",
    "cast",
    "char",
    "complex128",
    "complex64",
    "concat",
    "d",
    "downsample",
    "dtype",
    "experimental_collect_matching_metrics",
    "experimental_collect_prometheus_format_metrics",
    "experimental_push_metrics_to_prometheus",
    "experimental_update_verbose_logging",
    "float16",
    "float32",
    "float4_e2m1fn",
    "float64",
    "float8_e3m4",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "inf",
    "int16",
    "int2",
    "int32",
    "int4",
    "int64",
    "int8",
    "json",
    "newaxis",
    "open",
    "overlay",
    "parse_tensorstore_flags",
    "stack",
    "string",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "ustring",
    "virtual_chunked",
]


class Batch:
    """


    Batches are used to group together read operations for potentially improved
    efficiency.

    Operations associated with a batch will potentially be deferred until all
    references to the batch are released.

    The batch behavior of any particular operation ultimately depends on the
    underlying driver implementation, but in many cases batching operations can
    reduce the number of separate I/O requests performed.

    Example usage as a context manager (recommended):

        >>> store = ts.open(
        ...     {
        ...         'driver': 'zarr3',
        ...         'kvstore': {
        ...             'driver': 'file',
        ...             'path': 'tmp/dataset/'
        ...         },
        ...     },
        ...     shape=[5, 6],
        ...     chunk_layout=ts.ChunkLayout(read_chunk_shape=[2, 3],
        ...                                 write_chunk_shape=[6, 6]),
        ...     dtype=ts.uint16,
        ...     create=True,
        ...     delete_existing=True).result()
        >>> store[...] = np.arange(5 * 6, dtype=np.uint16).reshape([5, 6])
        >>> with ts.Batch() as batch:
        ...     read_future1 = store[:3].read(batch=batch)
        ...     read_future2 = store[3:].read(batch=batch)
        >>> await read_future1
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17]], dtype=uint16)
        >>> await read_future2
        array([[18, 19, 20, 21, 22, 23],
               [24, 25, 26, 27, 28, 29]], dtype=uint16)

    .. warning::

       Any operation performed as part of a batch may be deferred until the batch is
       submitted.  Blocking on (or awaiting) the completion of such an operation
       while retaining a reference to the batch will likely lead to deadlock.

    Equivalent example using explicit call to :py:meth:`.submit`:

        >>> batch = ts.Batch()
        >>> read_future1 = store[:3].read(batch=batch)
        >>> read_future2 = store[3:].read(batch=batch)
        >>> batch.submit()
        >>> await read_future1
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17]], dtype=uint16)
        >>> await read_future2
        array([[18, 19, 20, 21, 22, 23],
               [24, 25, 26, 27, 28, 29]], dtype=uint16)

    Equivalent example relying on implicit submit by the destructor when the last reference is released:

        >>> batch = ts.Batch()
        >>> read_future1 = store[:3].read(batch=batch)
        >>> read_future2 = store[3:].read(batch=batch)
        >>> del batch
        >>> await read_future1
        array([[ 0,  1,  2,  3,  4,  5],
               [ 6,  7,  8,  9, 10, 11],
               [12, 13, 14, 15, 16, 17]], dtype=uint16)
        >>> await read_future2
        array([[18, 19, 20, 21, 22, 23],
               [24, 25, 26, 27, 28, 29]], dtype=uint16)

    .. warning::

       Relying on this implicit submit behavior is not recommended and may result in
       the submit being delayed indefinitely, due to Python implicitly retaining a
       reference to the object, or due to a cyclic reference.

    Group:
      Core

    Constructors
    ============

    Operations
    ==========

    """

    def __enter__(self) -> Batch: ...

    def __exit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> None: ...

    def __init__(self) -> None:
        """
        Creates a new batch.
        """

    def submit(self) -> None:
        """
        Submits the batch.

        After calling this method, attempting to start any new operation will this batch
        will result in an error.

        Raises:
          ValueError: If :py:meth:`.submit` has already been called.

        Group:
          Operations
        """


class ChunkLayout:
    """

    Describes the storage layout of a :py:obj:`tensorstore.TensorStore`.

    Group:
      Spec

    Constructors
    ------------

    Classes
    -------

    Accessors
    ---------

    Setters
    -------

    Chunk templates
    ---------------

    Comparison operators
    --------------------

    """

    class Grid:
        """

        Describes a regular grid layout for write/read/codec chunks.
        """

        __hash__: typing.ClassVar[None] = None

        def __eq__(self, other: ChunkLayout.Grid) -> bool:
            """
            Compares two chunk grids for equality.
            """

        @typing.overload
        def __init__(
            self,
            *,
            rank: int | None = None,
            shape: collections.abc.Iterable[int | None] | None = None,
            shape_soft_constraint: collections.abc.Iterable[int | None] | None = None,
            aspect_ratio: collections.abc.Iterable[float | None] | None = None,
            aspect_ratio_soft_constraint: (
                collections.abc.Iterable[float | None] | None
            ) = None,
            elements: int | None = None,
            elements_soft_constraint: int | None = None,
            grid: ChunkLayout.Grid | None = None,
            grid_soft_constraint: ChunkLayout.Grid | None = None,
        ) -> None:
            """
            Constructs a chunk grid.

            Args:
              rank: Specifies the number of dimensions.
              shape: Hard constraints on the chunk size for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.shape`.
              shape_soft_constraint: Soft constraints on the chunk size for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.shape_soft_constraint`.
              aspect_ratio: Aspect ratio for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.aspect_ratio`.
              aspect_ratio_soft_constraint: Soft constraints on the aspect ratio for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.aspect_ratio_soft_constraint`.
              elements: Target number of elements per chunk.  Corresponds to
                :json:schema:`ChunkLayout/Grid.elements`.
              elements_soft_constraint: Soft constraint on the target number of elements per chunk.  Corresponds to
                :json:schema:`ChunkLayout/Grid.elements_soft_constraint`.
              grid: Other grid constraints to merge in.  Hard and soft constraints in
                :py:param:`.grid` are retained as hard and soft constraints, respectively.
              grid_soft_constraint: Other grid constraints to merge in as soft constraints.


            Overload:
              components
            """

        @typing.overload
        def __init__(self, json: typing.Any) -> None:
            """
            Constructs from the :json:schema:`JSON representation<ChunkLayout/Grid>`.

            Overload:
              json
            """

        def to_json(self, include_defaults: bool = False) -> typing.Any:
            """
            Converts to the :json:schema:`JSON representation<ChunkLayout/Grid>`.
            """

        def update(
            self,
            *,
            rank: int | None = None,
            shape: collections.abc.Iterable[int | None] | None = None,
            shape_soft_constraint: collections.abc.Iterable[int | None] | None = None,
            aspect_ratio: collections.abc.Iterable[float | None] | None = None,
            aspect_ratio_soft_constraint: (
                collections.abc.Iterable[float | None] | None
            ) = None,
            elements: int | None = None,
            elements_soft_constraint: int | None = None,
            grid: ChunkLayout.Grid | None = None,
            grid_soft_constraint: ChunkLayout.Grid | None = None,
        ) -> None:
            """
            Adds additional constraints.

            Args:
              rank: Specifies the number of dimensions.
              shape: Hard constraints on the chunk size for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.shape`.
              shape_soft_constraint: Soft constraints on the chunk size for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.shape_soft_constraint`.
              aspect_ratio: Aspect ratio for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.aspect_ratio`.
              aspect_ratio_soft_constraint: Soft constraints on the aspect ratio for each dimension.  Corresponds to
                :json:schema:`ChunkLayout/Grid.aspect_ratio_soft_constraint`.
              elements: Target number of elements per chunk.  Corresponds to
                :json:schema:`ChunkLayout/Grid.elements`.
              elements_soft_constraint: Soft constraint on the target number of elements per chunk.  Corresponds to
                :json:schema:`ChunkLayout/Grid.elements_soft_constraint`.
              grid: Other grid constraints to merge in.  Hard and soft constraints in
                :py:param:`.grid` are retained as hard and soft constraints, respectively.
              grid_soft_constraint: Other grid constraints to merge in as soft constraints.
            """

        @property
        def aspect_ratio(self) -> tuple[float | None, ...] | None:
            """
            Chunk shape aspect ratio.
            """

        @property
        def aspect_ratio_soft_constraint(self) -> tuple[float | None, ...] | None:
            """
            Soft constraints on chunk shape aspect ratio.
            """

        @property
        def elements(self) -> int | None:
            """
            Target number of elements per chunk.
            """

        @property
        def elements_soft_constraint(self) -> int | None:
            """
            Soft constraint on target number of elements per chunk.
            """

        @property
        def ndim(self) -> int | None:
            """
            Alias for :py:obj:`.rank`.
            """

        @property
        def rank(self) -> int | None:
            """
            Number of dimensions, or :py:obj:`None` if unspecified.
            """

        @property
        def shape(self) -> tuple[int | None, ...] | None:
            """
            Hard constraints on chunk shape.
            """

        @property
        def shape_soft_constraint(self) -> tuple[int | None, ...] | None:
            """
            Soft constraints on chunk shape.
            """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, other: ChunkLayout) -> bool:
        """
        Compares two chunk layouts for equality.
        """

    @typing.overload
    def __init__(self, json: typing.Any) -> None:
        """
        Constructs from the :json:schema:`JSON representation<ChunkLayout>`.

        Overload:
          json
        """

    @typing.overload
    def __init__(
        self,
        *,
        rank: int | None = None,
        inner_order: collections.abc.Iterable[int] | None = None,
        inner_order_soft_constraint: collections.abc.Iterable[int] | None = None,
        grid_origin: collections.abc.Iterable[int | None] | None = None,
        grid_origin_soft_constraint: collections.abc.Iterable[int | None] | None = None,
        chunk: ChunkLayout.Grid | None = None,
        write_chunk: ChunkLayout.Grid | None = None,
        read_chunk: ChunkLayout.Grid | None = None,
        codec_chunk: ChunkLayout.Grid | None = None,
        chunk_shape: collections.abc.Iterable[int | None] | None = None,
        chunk_shape_soft_constraint: collections.abc.Iterable[int | None] | None = None,
        write_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        write_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        read_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        read_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        codec_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        codec_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        write_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        write_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        read_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        read_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        codec_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        codec_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        chunk_elements: int | None = None,
        chunk_elements_soft_constraint: int | None = None,
        write_chunk_elements: int | None = None,
        write_chunk_elements_soft_constraint: int | None = None,
        read_chunk_elements: int | None = None,
        read_chunk_elements_soft_constraint: int | None = None,
        codec_chunk_elements: int | None = None,
        codec_chunk_elements_soft_constraint: int | None = None,
        finalize: bool | None = None,
    ) -> None:
        """
        Constructs from component parts.

        Args:
          rank: Specifies the number of dimensions.
          inner_order: Permutation specifying the element storage order within the innermost chunks.
            Corresponds to the JSON :json:schema:`ChunkLayout.inner_order` member.  This
            must be a permutation of ``[0, 1, ..., rank-1]``.  Lexicographic order (i.e. C
            order/row-major order) is specified as ``[0, 1, ..., rank-1]``, while
            colexicographic order (i.e. Fortran order/column-major order) is specified as
            ``[rank-1, ..., 1, 0]``.
          inner_order_soft_constraint: Specifies a preferred value for :py:obj:`~ChunkLayout.inner_order` rather than a
            hard constraint.  Corresponds to the JSON
            :json:schema:`ChunkLayout.inner_order_soft_constraint` member.  If
            :py:obj:`~ChunkLayout.inner_order` is also specified, it takes precedence.
          grid_origin: Hard constraints on the origin of the chunk grid.
            Corresponds to the JSON :json:schema:`ChunkLayout.grid_origin` member.
          grid_origin_soft_constraint: Soft constraints on the origin of the chunk grid.  Corresponds to the JSON
            :json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
          chunk: Common constraints on write, read, and codec chunks.  Corresponds to the JSON
            :json:schema:`ChunkLayout.chunk` member.  The :py:obj:`~ChunkLayout.Grid.shape`
            and :py:obj:`~ChunkLayout.Grid.elements` constraints apply only to write and
            read chunks, while the :py:obj:`~ChunkLayout.Grid.aspect_ratio` constraints
            apply to write, read, and codec chunks.
          write_chunk: Constraints on write chunks.  Corresponds to the JSON
            :json:schema:`ChunkLayout.write_chunk` member.
          read_chunk: Constraints on read chunks.  Corresponds to
            the JSON :json:schema:`ChunkLayout.read_chunk` member.
          codec_chunk: Constraints on codec chunks.  Corresponds to
            the JSON :json:schema:`ChunkLayout.codec_chunk` member.
          chunk_shape: Hard constraints on both the write and read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_shape` and :py:param:`.read_chunk_shape`.
          chunk_shape_soft_constraint: Soft constraints on both the write and read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_shape_soft_constraint` and
            :py:param:`.read_chunk_shape_soft_constraint`.
          write_chunk_shape: Hard constraints on the write chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_shape_soft_constraint: Soft constraints on the write chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_shape: Hard constraints on the read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_shape_soft_constraint: Soft constraints on the read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_shape: Soft constraints on the codec chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_shape_soft_constraint: Soft constraints on the codec chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          chunk_aspect_ratio: Hard constraints on the write, read, and codec chunk aspect ratio.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
            :py:param:`.write_chunk_aspect_ratio`, :py:param:`.read_chunk_aspect_ratio`, and
            :py:param:`.codec_chunk_aspect_ratio`.
          chunk_aspect_ratio_soft_constraint: Soft constraints on the write, read, and codec chunk aspect ratio.  Corresponds
            to the :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
            :py:param:`.write_chunk_aspect_ratio_soft_constraint`,
            :py:param:`.read_chunk_aspect_ratio_soft_constraint`, and
            :py:param:`.codec_chunk_aspect_ratio_soft_constraint`.
          write_chunk_aspect_ratio: Hard constraints on the write chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_aspect_ratio_soft_constraint: Soft constraints on the write chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_aspect_ratio: Hard constraints on the read chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_aspect_ratio_soft_constraint: Soft constraints on the read chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_aspect_ratio: Soft constraints on the codec chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_aspect_ratio_soft_constraint: Soft constraints on the codec chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          chunk_elements: Hard constraints on the target number of elements for write and read chunks.
            Corresponds to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_elements` and :py:param:`.read_chunk_elements`.
          chunk_elements_soft_constraint: Soft constraints on the target number of elements for write and read chunks.
            Corresponds to the JSON
            :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_elements_soft_constraint` and
            :py:param:`.read_chunk_elements_soft_constraint`.
          write_chunk_elements: Hard constraints on the target number of elements for write chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_elements_soft_constraint: Soft constraints on the target number of elements for write chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_elements: Hard constraints on the target number of elements for read chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_elements_soft_constraint: Soft constraints on the target number of elements for read chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_elements: Hard constraints on the target number of elements for codec chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_elements_soft_constraint: Soft constraints on the target number of elements for codec chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.codec_chunk`.
          finalize: Validates and converts the layout into a *precise* chunk
            layout.

            - All dimensions of :py:obj:`~ChunkLayout.grid_origin` must be specified as hard
              constraints.

            - Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.shape` soft constraints
              are cleared.

            - Any unspecified dimensions of the read chunk shape are set from the
              write chunk shape.

            - Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.aspect_ratio` or
              :py:obj:`~ChunkLayout.Grid.elements` constraints are cleared.


        Overload:
          components
        """

    def __repr__(self) -> str: ...

    def to_json(self) -> typing.Any:
        """
        Converts to the :json:schema:`JSON representation<ChunkLayout>`.

        Example:

            >>> layout = ts.ChunkLayout(
            ...     inner_order=[0, 2, 1],
            ...     write_chunk_shape_soft_constraint=[100, None, 200],
            ...     read_chunk_elements=1000000)
            >>> layout.to_json()
            {'inner_order': [0, 2, 1],
             'read_chunk': {'elements': 1000000},
             'write_chunk': {'shape_soft_constraint': [100, None, 200]}}

        Group:
          Accessors
        """

    def update(
        self,
        *,
        rank: int | None = None,
        inner_order: collections.abc.Iterable[int] | None = None,
        inner_order_soft_constraint: collections.abc.Iterable[int] | None = None,
        grid_origin: collections.abc.Iterable[int | None] | None = None,
        grid_origin_soft_constraint: collections.abc.Iterable[int | None] | None = None,
        chunk: ChunkLayout.Grid | None = None,
        write_chunk: ChunkLayout.Grid | None = None,
        read_chunk: ChunkLayout.Grid | None = None,
        codec_chunk: ChunkLayout.Grid | None = None,
        chunk_shape: collections.abc.Iterable[int | None] | None = None,
        chunk_shape_soft_constraint: collections.abc.Iterable[int | None] | None = None,
        write_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        write_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        read_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        read_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        codec_chunk_shape: collections.abc.Iterable[int | None] | None = None,
        codec_chunk_shape_soft_constraint: (
            collections.abc.Iterable[int | None] | None
        ) = None,
        chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        write_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        write_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        read_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        read_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        codec_chunk_aspect_ratio: collections.abc.Iterable[float | None] | None = None,
        codec_chunk_aspect_ratio_soft_constraint: (
            collections.abc.Iterable[float | None] | None
        ) = None,
        chunk_elements: int | None = None,
        chunk_elements_soft_constraint: int | None = None,
        write_chunk_elements: int | None = None,
        write_chunk_elements_soft_constraint: int | None = None,
        read_chunk_elements: int | None = None,
        read_chunk_elements_soft_constraint: int | None = None,
        codec_chunk_elements: int | None = None,
        codec_chunk_elements_soft_constraint: int | None = None,
        finalize: bool | None = None,
    ) -> None:
        """
        Adds additional constraints.

        Args:
          rank: Specifies the number of dimensions.
          inner_order: Permutation specifying the element storage order within the innermost chunks.
            Corresponds to the JSON :json:schema:`ChunkLayout.inner_order` member.  This
            must be a permutation of ``[0, 1, ..., rank-1]``.  Lexicographic order (i.e. C
            order/row-major order) is specified as ``[0, 1, ..., rank-1]``, while
            colexicographic order (i.e. Fortran order/column-major order) is specified as
            ``[rank-1, ..., 1, 0]``.
          inner_order_soft_constraint: Specifies a preferred value for :py:obj:`~ChunkLayout.inner_order` rather than a
            hard constraint.  Corresponds to the JSON
            :json:schema:`ChunkLayout.inner_order_soft_constraint` member.  If
            :py:obj:`~ChunkLayout.inner_order` is also specified, it takes precedence.
          grid_origin: Hard constraints on the origin of the chunk grid.
            Corresponds to the JSON :json:schema:`ChunkLayout.grid_origin` member.
          grid_origin_soft_constraint: Soft constraints on the origin of the chunk grid.  Corresponds to the JSON
            :json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
          chunk: Common constraints on write, read, and codec chunks.  Corresponds to the JSON
            :json:schema:`ChunkLayout.chunk` member.  The :py:obj:`~ChunkLayout.Grid.shape`
            and :py:obj:`~ChunkLayout.Grid.elements` constraints apply only to write and
            read chunks, while the :py:obj:`~ChunkLayout.Grid.aspect_ratio` constraints
            apply to write, read, and codec chunks.
          write_chunk: Constraints on write chunks.  Corresponds to the JSON
            :json:schema:`ChunkLayout.write_chunk` member.
          read_chunk: Constraints on read chunks.  Corresponds to
            the JSON :json:schema:`ChunkLayout.read_chunk` member.
          codec_chunk: Constraints on codec chunks.  Corresponds to
            the JSON :json:schema:`ChunkLayout.codec_chunk` member.
          chunk_shape: Hard constraints on both the write and read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_shape` and :py:param:`.read_chunk_shape`.
          chunk_shape_soft_constraint: Soft constraints on both the write and read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_shape_soft_constraint` and
            :py:param:`.read_chunk_shape_soft_constraint`.
          write_chunk_shape: Hard constraints on the write chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_shape_soft_constraint: Soft constraints on the write chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_shape: Hard constraints on the read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_shape_soft_constraint: Soft constraints on the read chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_shape: Soft constraints on the codec chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_shape_soft_constraint: Soft constraints on the codec chunk shape.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.shape_soft_constraint` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          chunk_aspect_ratio: Hard constraints on the write, read, and codec chunk aspect ratio.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
            :py:param:`.write_chunk_aspect_ratio`, :py:param:`.read_chunk_aspect_ratio`, and
            :py:param:`.codec_chunk_aspect_ratio`.
          chunk_aspect_ratio_soft_constraint: Soft constraints on the write, read, and codec chunk aspect ratio.  Corresponds
            to the :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying
            :py:param:`.write_chunk_aspect_ratio_soft_constraint`,
            :py:param:`.read_chunk_aspect_ratio_soft_constraint`, and
            :py:param:`.codec_chunk_aspect_ratio_soft_constraint`.
          write_chunk_aspect_ratio: Hard constraints on the write chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_aspect_ratio_soft_constraint: Soft constraints on the write chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_aspect_ratio: Hard constraints on the read chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_aspect_ratio_soft_constraint: Soft constraints on the read chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_aspect_ratio: Soft constraints on the codec chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_aspect_ratio_soft_constraint: Soft constraints on the codec chunk aspect ratio.  Corresponds to the
            JSON :json:schema:`~ChunkLayout/Grid.aspect_ratio_soft_constraint` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          chunk_elements: Hard constraints on the target number of elements for write and read chunks.
            Corresponds to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_elements` and :py:param:`.read_chunk_elements`.
          chunk_elements_soft_constraint: Soft constraints on the target number of elements for write and read chunks.
            Corresponds to the JSON
            :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member of
            :json:schema:`ChunkLayout.chunk`.  Equivalent to specifying both
            :py:param:`.write_chunk_elements_soft_constraint` and
            :py:param:`.read_chunk_elements_soft_constraint`.
          write_chunk_elements: Hard constraints on the target number of elements for write chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.write_chunk`.
          write_chunk_elements_soft_constraint: Soft constraints on the target number of elements for write chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.write_chunk`.
          read_chunk_elements: Hard constraints on the target number of elements for read chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.read_chunk`.
          read_chunk_elements_soft_constraint: Soft constraints on the target number of elements for read chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.read_chunk`.
          codec_chunk_elements: Hard constraints on the target number of elements for codec chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements` member of
            :json:schema:`ChunkLayout.codec_chunk`.
          codec_chunk_elements_soft_constraint: Soft constraints on the target number of elements for codec chunks.  Corresponds
            to the JSON :json:schema:`~ChunkLayout/Grid.elements_soft_constraint` member
            of :json:schema:`ChunkLayout.codec_chunk`.
          finalize: Validates and converts the layout into a *precise* chunk
            layout.

            - All dimensions of :py:obj:`~ChunkLayout.grid_origin` must be specified as hard
              constraints.

            - Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.shape` soft constraints
              are cleared.

            - Any unspecified dimensions of the read chunk shape are set from the
              write chunk shape.

            - Any write/read/codec chunk :py:obj:`~ChunkLayout.Grid.aspect_ratio` or
              :py:obj:`~ChunkLayout.Grid.elements` constraints are cleared.


        Group:
          Setters
        """

    @property
    def codec_chunk(self) -> ChunkLayout.Grid:
        """
        Chunk grid used by the codec.

        See also:
          - JSON :json:schema:`ChunkLayout.codec_chunk` member.

        Group:
          Accessors
        """

    @property
    def grid_origin(self) -> tuple[int | None, ...] | None:
        """
        Hard constraints on the grid origin.

        See also:
          - JSON :json:schema:`ChunkLayout.grid_origin` member.

        Group:
          Accessors
        """

    @property
    def grid_origin_soft_constraint(self) -> tuple[int | None, ...] | None:
        """
        Soft constraints on the grid origin.

        See also:
          - JSON :json:schema:`ChunkLayout.grid_origin_soft_constraint` member.
          - :py:obj:`.grid_origin`

        Group:
          Accessors
        """

    @property
    def inner_order(self) -> tuple[int, ...] | None:
        """
        Permutation specifying the element storage order within the innermost chunks.

        If the inner order is specified as a soft constraint rather than a hard
        constraint, :py:obj:`.inner_order` is equal to `None` and the soft constraint is
        accessed via :py:obj:`.inner_order_soft_constraint`.

        Lexicographic order (i.e. C order/row-major order) is specified as ``[0, 1, ...,
        rank-1]``, while colexicographic order (i.e. Fortran order/column-major order)
        is specified as ``[rank-1, ..., 1, 0]``.

        See also:
          - :py:obj:`.inner_order_soft_constraint`
          - JSON :json:schema:`ChunkLayout.inner_order` member

        Group:
          Accessors
        """

    @property
    def inner_order_soft_constraint(self) -> tuple[int, ...] | None:
        """
        Permutation specifying soft constraint on the element storage order.

        If the inner order is specified as a hard constraint rather than a soft
        constraint, :py:obj:`.inner_order_soft_constraint` is equal to `None` and the
        hard constraint is accessed via :py:obj:`.inner_order`.

        See also:
          - :py:obj:`.inner_order`
          - JSON :json:schema:`ChunkLayout.inner_order_soft_constraint` member

        Group:
          Accessors
        """

    @property
    def ndim(self) -> int:
        """
        Alias for :py:obj:`.rank`.

        Example:

            >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
            >>> layout.ndim
            3

        Group:
          Accessors
        """

    @property
    def rank(self) -> int:
        """
        Number of dimensions in the index space.

        Example:

            >>> layout = ts.ChunkLayout(inner_order=[0, 2, 1])
            >>> layout.rank
            3

        Group:
          Accessors
        """

    @property
    def read_chunk(self) -> ChunkLayout.Grid:
        """
        Chunk grid for efficient reads.

        See also:
          - JSON :json:schema:`ChunkLayout.read_chunk` member.

        Group:
          Accessors
        """

    @property
    def read_chunk_template(self) -> IndexDomain:
        """
        Chunk offset and shape for efficient reads.

        Example:

            >>> layout = ts.ChunkLayout(grid_origin=[5, 6, 7],
            ...                         read_chunk_shape=[100, 200, 300])
            >>> layout.read_chunk_template
            { [5, 105), [6, 206), [7, 307) }

        Note:

          Only the hard constraints :py:obj:`.grid_origin` and
          :py:obj:`~ChunkLayout.Grid.shape` of :py:obj:`.read_chunk` are taken into
          account.  The soft constraints :py:obj:`.grid_origin_soft_constraint` and all
          other constraints specified on :py:obj:`.read_chunk` are **ignored**.

        For any dimension ``i`` for which :python:`self.grid_origin[i] is None` or
        :python:`self.read_chunk_shape[i] is None`,
        :python:`self.read_chunk_template[i]` is an unbounded interval
        :python:`ts.Dim()`:

            >>> layout = ts.ChunkLayout(grid_origin=[None, 6, 7],
            ...                         read_chunk_shape=[100, None, 200])
            >>> layout.read_chunk_template
            { (-inf, +inf), (-inf, +inf), [7, 207) }

        Raises:

          ValueError: If :py:obj:`.rank` is unspecified or :py:obj:`.grid_origin` and
            :python:`self.read_chunk.shape` are incompatible.

        See also:
          - :py:meth:`ChunkLayout.write_chunk_template`

        Group:
          Chunk templates
        """

    @property
    def write_chunk(self) -> ChunkLayout.Grid:
        """
        Chunk grid for efficient writes.

        See also:
          - JSON :json:schema:`ChunkLayout.write_chunk` member.

        Group:
          Accessors
        """

    @property
    def write_chunk_template(self) -> IndexDomain:
        """
        Chunk offset and shape for efficient writes.

        Example:

            >>> layout = ts.ChunkLayout(grid_origin=[5, 6, 7],
            ...                         write_chunk_shape=[100, 200, 300])
            >>> layout.write_chunk_template
            { [5, 105), [6, 206), [7, 307) }

        Note:

          Only the hard constraints :py:obj:`.grid_origin` and
          :py:obj:`~ChunkLayout.Grid.shape` of :py:obj:`.write_chunk` are taken into
          account.  The soft constraints :py:obj:`.grid_origin_soft_constraint` and all
          other constraints specified on :py:obj:`.write_chunk` are **ignored**.

        For any dimension ``i`` for which :python:`self.grid_origin[i] is None` or
        :python:`self.write_chunk.shape[i] is None`,
        :python:`self.write_chunk_template[i]` is an unbounded interval
        :python:`ts.Dim()`:

            >>> layout = ts.ChunkLayout(grid_origin=[None, 6, 7],
            ...                         write_chunk_shape=[100, None, 200])
            >>> layout.write_chunk_template
            { (-inf, +inf), (-inf, +inf), [7, 207) }

        Raises:

          ValueError: If :py:obj:`.rank` is unspecified or :py:obj:`.grid_origin` and
            :python:`self.write_chunk.shape` are incompatible.

        See also:
          - :py:meth:`ChunkLayout.read_chunk_template`

        Group:
          Chunk templates
        """


class CodecSpec:
    """

    Specifies driver-specific encoding/decoding parameters.

    Group:
      Spec
    """

    def __init__(self, json: typing.Any) -> None:
        """
        Constructs from the :json:schema:`JSON representation<Codec>`.
        """

    def __repr__(self) -> str: ...

    def to_json(self, include_defaults: bool = False) -> typing.Any:
        """
        Converts to the :json:schema:`JSON representation<Codec>`.
        """


class Context:
    """

    Manages shared TensorStore :ref:`context resources<context>`, such as caches and credentials.

    Group:
      Core

    See also:
      :ref:`context`

    """

    class Resource:
        """

        Handle to a context resource.
        """

        def __repr__(self) -> str: ...

        def to_json(self, include_defaults: bool = False) -> typing.Any:
            """
            Returns the :json:schema:`JSON representation<ContextResource>` of the context resource.

            Example:

                >>> context = ts.Context(
                ...     {'cache_pool#a': {
                ...         'total_bytes_limit': 10000000
                ...     }})
                >>> context['cache_pool#a'].to_json()
                {'total_bytes_limit': 10000000}

            Group:
              Accessors
            """

    class Spec:
        """

        Parsed representation of a :json:schema:`JSON Context<Context>` specification.
        """

        def __init__(self, json: typing.Any) -> None:
            """
            Creates a context specification from its :json:schema:`JSON representation<Context>`.
            """

        def __repr__(self) -> str: ...

        def to_json(self, include_defaults: bool = False) -> typing.Any:
            """
            Returns the :json:schema:`JSON representation<Context>`.

            Args:
              include_defaults: Indicates whether to include members even if they are equal to the default value.

            Group:
              Accessors
            """

    def __getitem__(self, key: str) -> Context.Resource:
        """
        Creates or retrieves the context resource for the given key.

        This is primarily useful for introspection of a context.

        Example:

            >>> context = ts.Context(
            ...     {'cache_pool#a': {
            ...         'total_bytes_limit': 10000000
            ...     }})
            >>> context['cache_pool#a']
            Context.Resource({'total_bytes_limit': 10000000})
            >>> context['cache_pool']
            Context.Resource({})

        Args:
          key: Resource key, of the form :python:`'<resource-type>'` or
            :python:`<resource-type>#<id>`.

        Returns:
          The resource handle.

        Group:
          Accessors
        """

    @typing.overload
    def __init__(self) -> None:
        """
        Constructs a default context.

        Example:

            >>> context = ts.Context()
            >>> context.spec is None
            True

        .. note::

           Each call to this constructor returns a unique default context instance, that
           does *not* share resources with other default context instances.  To share
           resources, you must use the same :py:obj:`Context` instance.

        Overload:
          default
        """

    @typing.overload
    def __init__(self, spec: Context.Spec, parent: Context | None = None) -> None:
        """
        Constructs a context from a parsed spec.

        Args:
          spec: Parsed context spec.
          parent: Parent context from which to inherit.  Defaults to a new default
            context as returned by :python:`tensorstore.Context()`.

        Overload:
          spec
        """

    @typing.overload
    def __init__(self, json: typing.Any, parent: Context | None = None) -> None:
        """
        Constructs a context from its :json:schema:`JSON representation<Context>`.

        Example:

            >>> context = ts.Context({'cache_pool': {'total_bytes_limit': 5000000}})
            >>> context.spec
            Context.Spec({'cache_pool': {'total_bytes_limit': 5000000}})

        Args:
          json: :json:schema:`JSON representation<Context>` of the context.
          parent: Parent context from which to inherit.  Defaults to a new default
            context as returned by :python:`tensorstore.Context()`.

        Overload:
          json
        """

    @property
    def parent(self) -> Context:
        """
        Parent context from which this context inherits.

        Example:

            >>> parent = ts.Context({
            ...     'cache_pool': {
            ...         'total_bytes_limit': 5000000
            ...     },
            ...     'file_io_concurrency': {
            ...         'limit': 10
            ...     }
            ... })
            >>> child = ts.Context({'cache_pool': {
            ...     'total_bytes_limit': 10000000
            ... }},
            ...                    parent=parent)
            >>> assert child.parent is parent
            >>> parent['cache_pool'].to_json()
            {'total_bytes_limit': 5000000}
            >>> child['cache_pool'].to_json()
            {'total_bytes_limit': 10000000}
            >>> child['file_io_concurrency'].to_json()
            {'limit': 10}

        Group:
          Accessors
        """

    @property
    def spec(self) -> Context.Spec:
        """
        Spec from which this context was constructed.

        Example:

            >>> parent = ts.Context({
            ...     'cache_pool': {
            ...         'total_bytes_limit': 5000000
            ...     },
            ...     'file_io_concurrency': {
            ...         'limit': 10
            ...     }
            ... })
            >>> child = ts.Context({'cache_pool': {
            ...     'total_bytes_limit': 10000000
            ... }},
            ...                    parent=parent)
            >>> child.spec
            Context.Spec({'cache_pool': {'total_bytes_limit': 10000000}})
            >>> child.parent.spec
            Context.Spec({
              'cache_pool': {'total_bytes_limit': 5000000},
              'file_io_concurrency': {'limit': 10},
            })

        Group:
          Accessors
        """


class Dim:
    """

    1-d index interval with optionally-implicit bounds and dimension label.

    Represents a contiguous range of integer :ref:`index values<index-space>`.  The
    inclusive lower and upper bounds may either be finite values in the closed
    interval :math:`[-(2^{62}-2), +(2^{62}-2)]`, or infinite, as indicated by
    -/+ :py:obj:`.inf` for the lower and upper bounds, respectively.

    The lower and upper bounds may additionally be marked as either
    :ref:`explicit or implicit<implicit-bounds>`.

    The interval may also have an associated
    :ref:`dimension label<dimension-labels>`, which is primarily useful for
    specifying the dimensions of an :py:obj:`.IndexDomain`.

    Examples:

        >>> ts.Dim('x')
        Dim(label="x")
        >>> ts.Dim(inclusive_min=3, exclusive_max=10, label='x')
        Dim(inclusive_min=3, exclusive_max=10, label="x")

    See also:
      :py:obj:`IndexDomain`

    Group:
      Indexing
    """

    __hash__: typing.ClassVar[None] = None

    @typing.overload
    def __contains__(self, other: int) -> bool:
        """
        Checks if the interval contains a given index.

        Examples:

            >>> 5 in ts.Dim(inclusive_min=1, exclusive_max=10)
            True
            >>> 5 in ts.Dim()
            True
            >>> 5 in ts.Dim(inclusive_min=6)
            False

        Overload:
          index

        Group:
          Operations
        """

    @typing.overload
    def __contains__(self, inner: Dim) -> bool:
        """
        Checks if the interval contains another interval.

        Examples:

            >>> ts.Dim(inclusive_min=1, exclusive_max=5) in ts.Dim(10)
            True
            >>> ts.Dim(inclusive_min=1, exclusive_max=5) in ts.Dim(4)
            False

        Overload:
          dim

        Group:
          Operations
        """

    def __copy__(self) -> Dim: ...

    def __deepcopy__(self, memo: dict) -> Dim: ...

    def __eq__(self, other: Dim) -> bool:
        """
        Compares for equality with another interval.

        In addition to the bounds, the values of :py:obj:`.label`,
        :py:obj:`.implicit_lower`, and :py:obj:`.implicit_upper` are also taken into
        account.

            >>> a = ts.Dim(inclusive_min=5, exclusive_max=10)
            >>> b = ts.Dim(inclusive_min=5, inclusive_max=9)
            >>> a == b
            True

        Group:
          Operations
        """

    @typing.overload
    def __init__(
        self,
        label: str | None = None,
        *,
        implicit_lower: bool = True,
        implicit_upper: bool = True,
    ) -> None:
        """
        Constructs an unbounded interval ``(-inf, +inf)``.

        Args:
          label: :ref:`Dimension label<dimension-labels>`.
          implicit_lower: Indicates whether the lower bound is
            :ref:`implicit<implicit-bounds>`.
          implicit_upper: Indicates whether the upper bound is
            :ref:`implicit<implicit-bounds>`.

        Examples:

            >>> x = ts.Dim()
            >>> print(x)
            (-inf*, +inf*)
            >>> x.finite
            False

            >>> x = ts.Dim("x", implicit_upper=False)
            >>> print(x)
            "x": (-inf*, +inf)
            >>> x.finite
            False

        Overload:
          unbounded
        """

    @typing.overload
    def __init__(
        self,
        size: int | None,
        label: str | None = None,
        *,
        inclusive_min: int | None = None,
        implicit_lower: bool = False,
        implicit_upper: bool | None = None,
    ) -> None:
        """
        Constructs a sized interval ``[inclusive_min, inclusive_min+size)``.

        Args:
          size: Size of the interval.
          label: :ref:`Dimension label<dimension-labels>`.
          inclusive_min: Inclusive lower bound.  Defaults to :python:`0`.
          implicit_lower: Indicates whether the lower bound is
            :ref:`implicit<implicit-bounds>`.
          implicit_upper: Indicates whether the upper bound is
            :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
            :python:`size` is specified, otherwise :python:`True`.

        Examples:

            >>> x = ts.Dim(10)
            >>> print(x)
            [0, 10)
            >>> print(ts.Dim(inclusive_min=5, size=10))
            [5, 15)

        Overload:
          size
        """

    @typing.overload
    def __init__(
        self,
        inclusive_min: int | None = ...,
        exclusive_max: int | None = ...,
        *,
        label: str | None = None,
        implicit_lower: bool | None = None,
        implicit_upper: bool | None = None,
    ) -> None:
        """
        Constructs a half-open interval ``[inclusive_min, exclusive_max)``.

        Args:
          inclusive_min: Inclusive lower bound.
          exclusive_max: Exclusive upper bound.
          label: :ref:`Dimension label<dimension-labels>`.
          implicit_lower: Indicates whether the lower bound is
            :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
            ``inclusive_min`` is specified, otherwise :python:`True`.
          implicit_upper: Indicates whether the upper bound is
            :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
            ``exclusive_max`` is specified, otherwise :python:`True`.

        Examples:

            >>> x = ts.Dim(5, 10)
            >>> x
            Dim(inclusive_min=5, exclusive_max=10)
            >>> print(x)
            [5, 10)

        Overload:
          exclusive_max
        """

    @typing.overload
    def __init__(
        self,
        *,
        inclusive_min: int | None = ...,
        inclusive_max: int | None = ...,
        label: str | None = None,
        implicit_lower: bool | None = None,
        implicit_upper: bool | None = None,
    ) -> None:
        """
        Constructs a closed interval ``[inclusive_min, inclusive_max]``.

        Args:
          inclusive_min: Inclusive lower bound.
          inclusive_max: Inclusive upper bound.
          label: :ref:`Dimension label<dimension-labels>`.
          implicit_lower: Indicates whether the lower bound is
            :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
            ``inclusive_min`` is specified, otherwise :python:`True`.
          implicit_upper: Indicates whether the upper bound is
            :ref:`implicit<implicit-bounds>`.  Defaults to :python:`False` if
            ``exclusive_max`` is specified, otherwise :python:`True`.

        Examples:

            >>> x = ts.Dim(inclusive_min=5, inclusive_max=10)
            >>> x
            Dim(inclusive_min=5, exclusive_max=11)
            >>> print(x)
            [5, 11)

        Overload:
          inclusive_max
        """

    def __iter__(self) -> typing.Iterator:
        """
        Enables iteration over the indices contained in the interval.

        Raises:
            ValueError: If not :py:obj:`.finite`.

        Examples:

            >>> list(ts.Dim(inclusive_min=1, exclusive_max=6))
            [1, 2, 3, 4, 5]

        Group:
          Operations
        """

    def __len__(self) -> int:
        """
        Size of the interval, equivalent to :py:obj:`.size`.

        Group:
          Accessors
        """

    def __repr__(self) -> str:
        """
        Returns the string representation as a Python expression.

            >>> ts.Dim(size=5, label='x', implicit_upper=True)
            Dim(inclusive_min=0, exclusive_max=5, implicit_upper=True, label="x")
        """

    def __str__(self) -> str:
        """
        Returns the string representation of the interval.

            >>> print(ts.Dim(inclusive_min=5, exclusive_max=10))
            [5, 10)
            >>> print(ts.Dim(exclusive_max=10))
            (-inf*, 10)
            >>> print(ts.Dim(exclusive_max=10, label="x"))
            "x": (-inf*, 10)
        """

    def hull(self, other: Dim) -> Dim:
        """
        Hull with another Dim.

        The ``implicit`` flag that corresponds to the selected bound is propagated.
        The :py:obj:`.label` field, if non-empty, must match, and will be propagated.

        Args:
          other: Object to hull with.

        Example:

            >>> a = ts.Dim(inclusive_min=1, exclusive_max=5, label='x')
            >>> a.hull(ts.Dim(size=3))
            Dim(inclusive_min=0, exclusive_max=5, label="x")
        """

    def intersect(self, other: Dim) -> Dim:
        """
        Intersect with another Dim.

        The ``implicit`` flag that corresponds to the selected bound is propagated.
        The :py:obj:`.label`  field, if non-empty, must match, and will be propagated.

        Args:
          other: Object to intersect with.

        Example:

            >>> a = ts.Dim(inclusive_min=1, exclusive_max=5, label='x')
            >>> a.intersect(ts.Dim(size=3))
            Dim(inclusive_min=1, exclusive_max=3, label="x")
        """

    @property
    def empty(self) -> bool:
        """
        Returns `True` if `size` is zero.

        Group:
          Accessors
        """

    @property
    def exclusive_max(self) -> int:
        """
        Exclusive upper bound of the interval.

        Equal to :python:`self.inclusive_max + 1`.  If the interval is unbounded above,
        equal to the special value of :py:obj:`+inf+1<tensorstore.inf>`.

        Example:

            >>> ts.Dim(inclusive_min=5, inclusive_max=10).exclusive_max
            11
            >>> ts.Dim(exclusive_max=5).exclusive_max
            5
            >>> ts.Dim().exclusive_max
            4611686018427387904

        Group:
          Accessors
        """

    @property
    def exclusive_min(self) -> int:
        """
        Exclusive lower bound of the interval.

        Equal to :python:`self.inclusive_min - 1`.  If the interval is unbounded below,
        equal to the special value of :py:obj:`-inf-1<tensorstore.inf>`.

        Example:

            >>> ts.Dim(inclusive_min=5, inclusive_max=10).exclusive_min
            4
            >>> ts.Dim(5).exclusive_min
            -1
            >>> ts.Dim(exclusive_max=10).exclusive_min
            -4611686018427387904
            >>> ts.Dim().exclusive_min
            -4611686018427387904

        Group:
          Accessors
        """

    @property
    def finite(self) -> bool:
        """
        Indicates if the interval is finite.

        Example:

          >>> ts.Dim().finite
          False
          >>> ts.Dim(5).finite
          True
          >>> ts.Dim(exclusive_max=10).finite
          False
          >>> ts.Dim(inclusive_min=10).finite
          False
          >>> ts.Dim(inclusive_min=10, exclusive_max=20).finite
          True

        Group:
          Accessors
        """

    @property
    def implicit_lower(self) -> bool:
        """
        Indicates if the lower bound is :ref:`implicit/resizeable<implicit-bounds>`.

        Example:

            >>> ts.Dim().implicit_lower
            True
            >>> ts.Dim(5).implicit_lower
            False
            >>> ts.Dim(exclusive_max=5).implicit_lower
            True
            >>> ts.Dim(inclusive_min=1, exclusive_max=5).implicit_lower
            False
            >>> ts.Dim(implicit_lower=False).implicit_lower
            False
            >>> ts.Dim(inclusive_min=5, implicit_lower=True).implicit_lower
            True

        Group:
          Accessors
        """

    @implicit_lower.setter
    def implicit_lower(self, arg1: bool) -> None: ...

    @property
    def implicit_upper(self) -> bool:
        """
        Indicates if the upper bound is :ref:`implicit/resizeable<implicit-bounds>`.

        Example:

            >>> ts.Dim().implicit_upper
            True
            >>> ts.Dim(5).implicit_upper
            False
            >>> ts.Dim(inclusive_min=5).implicit_upper
            True
            >>> ts.Dim(inclusive_min=1, exclusive_max=5).implicit_upper
            False
            >>> ts.Dim(implicit_upper=False).implicit_upper
            False
            >>> ts.Dim(inclusive_max=5, implicit_upper=True).implicit_upper
            True

        Group:
          Accessors
        """

    @implicit_upper.setter
    def implicit_upper(self, arg1: bool) -> None: ...

    @property
    def inclusive_max(self) -> int:
        """
        Inclusive upper bound of the interval.

        Equal to :python:`self.exclusive_max - 1`.  If the interval is unbounded above,
        equal to the special value of :py:obj:`+inf<tensorstore.inf>`.

        Example:

            >>> ts.Dim(inclusive_min=5, inclusive_max=10).inclusive_max
            10
            >>> ts.Dim(exclusive_max=5).inclusive_max
            4
            >>> ts.Dim().inclusive_max
            4611686018427387903

        Group:
          Accessors
        """

    @property
    def inclusive_min(self) -> int:
        """
        Inclusive lower bound of the interval.

        Equal to :python:`self.exclusive_min + 1`.  If the interval is unbounded below,
        equal to the special value of :py:obj:`-inf<tensorstore.inf>`.

        Example:

            >>> ts.Dim(5).inclusive_min
            0
            >>> ts.Dim(inclusive_min=5, inclusive_max=10).inclusive_min
            5
            >>> ts.Dim().inclusive_min
            -4611686018427387903

        Group:
          Accessors
        """

    @property
    def label(self) -> str:
        """
        Dimension label, or the empty string to indicate an unlabeled dimension.

        Example:

            >>> ts.Dim().label
            ''
            >>> ts.Dim(label='x').label
            'x'

        Group:
          Accessors
        """

    @label.setter
    def label(self, arg1: str) -> None: ...

    @property
    def size(self) -> int:
        """
        Size of the interval.

        Equal to :python:`self.exclusive_max - self.inclusive_min`.

        Example:

            >>> ts.Dim(5).size
            5
            >>> ts.Dim(inclusive_min=3, inclusive_max=7).size
            5
            >>> ts.Dim().size
            9223372036854775807

        Note:

          If the interval is unbounded below or above
          (i.e. :python:`self.finite == False`), this value it not particularly
          meaningful.

        Group:
          Accessors
        """


class DimExpression:
    """

    Specifies an advanced indexing operation.

    :ref:`Dimension expressions<python-dim-expressions>` permit indexing using
    :ref:`dimension labels<dimension-labels>`, and also support additional operations
    that cannot be performed with plain :ref:`python-numpy-style-indexing`.

    Group:
      Indexing

    Operations
    ==========

    """

    class _Label:
        __iter__ = None

        def __getitem__(
            self, labels: str | collections.abc.Iterable[str]
        ) -> DimExpression:
            """
            Sets (or changes) the :ref:`labels<dimension-labels>` of the selected dimensions.

            Examples:

                >>> ts.IndexTransform(3)[ts.d[0].label['x']]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf*, +inf*) "x"
                    1: (-inf*, +inf*)
                    2: (-inf*, +inf*)
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]
                >>> ts.IndexTransform(3)[ts.d[0, 2].label['x', 'z']]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf*, +inf*) "x"
                    1: (-inf*, +inf*)
                    2: (-inf*, +inf*) "z"
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]
                >>> ts.IndexTransform(3)[ts.d[:].label['x', 'y', 'z']]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf*, +inf*) "x"
                    1: (-inf*, +inf*) "y"
                    2: (-inf*, +inf*) "z"
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]
                >>> ts.IndexTransform(3)[ts.d[0, 1].label['x', 'y'].translate_by[2]]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf*, +inf*) "x"
                    1: (-inf*, +inf*) "y"
                    2: (-inf*, +inf*)
                  Output index maps:
                    out[0] = -2 + 1 * in[0]
                    out[1] = -2 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]

            The new dimension selection is the same as the prior dimension selection.

            Args:
              labels: Dimension labels for each selected dimension.

            Returns:
              Dimension expression with the label operation added.

            Raises:
              IndexError: If the number of labels does not match the number of selected
                dimensions, or if the resultant domain would have duplicate labels.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> DimExpression:
            """
            Marks the lower/upper bounds of the selected dimensions as
            :ref:`implicit/explicit<implicit-bounds>`.

            For a `TensorStore`, implicit bounds indicate resizeable dimensions.  Marking a
            bound as explicit fixes it to its current value such that it won't be adjusted
            by subsequent `TensorStore.resolve` calls if the stored bounds change.

            Because implicit bounds do not constrain subsequent indexing/slicing operations,
            a bound may be marked implicit in order to expand the domain.

            .. warning::

               Be careful when marking bounds as implicit, since this may bypass intended
               constraints on the domain.

            Examples:

                >>> s = await ts.open({
                ...     'driver': 'zarr',
                ...     'kvstore': 'memory://'
                ... },
                ...                   shape=[100, 200],
                ...                   dtype=ts.uint32,
                ...                   create=True)
                >>> s.domain
                { [0, 100*), [0, 200*) }
                >>> await s.resize(exclusive_max=[200, 300])
                >>> (await s.resolve()).domain
                { [0, 200*), [0, 300*) }
                >>> (await s[ts.d[0].mark_bounds_implicit[False]].resolve()).domain
                { [0, 100), [0, 300*) }
                >>> s_subregion = s[20:30, 40:50]
                >>> s_subregion.domain
                { [20, 30), [40, 50) }
                >>> (await
                ...  s_subregion[ts.d[0].mark_bounds_implicit[:True]].resolve()).domain
                { [20, 200*), [40, 50) }

                >>> t = ts.IndexTransform(input_rank=3)
                >>> t = t[ts.d[0, 2].mark_bounds_implicit[False]]
                >>> t
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf, +inf)
                    1: (-inf*, +inf*)
                    2: (-inf, +inf)
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]
                >>> t = t[ts.d[0, 1].mark_bounds_implicit[:True]]
                >>> t
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf, +inf*)
                    1: (-inf*, +inf*)
                    2: (-inf, +inf)
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]
                >>> t = t[ts.d[1, 2].mark_bounds_implicit[True:False]]
                >>> t
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: (-inf, +inf*)
                    1: (-inf*, +inf)
                    2: (-inf*, +inf)
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[2]

            The new dimension selection is the same as the prior dimension selection.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.  Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Returns:
              Dimension expression with bounds marked as implicit/explicit.

            Raises:
              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _Oindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> DimExpression:
            """
            Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`DimExpression.__getitem__`, but differs in that any integer or
            boolean array indexing terms are applied orthogonally:

            Examples:

               >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
               >>> transform[ts.d['x', 'z'].oindex[[1, 2, 3], [4, 5, 6]]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [0, 3)
                   1: (-inf*, +inf*) "y"
                   2: [0, 3)
                 Output index maps:
                   out[0] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{{1}}, {{2}}, {{3}}}
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{{4, 5, 6}}}

            Returns:
              Dimension expression with the indexing operation added.

            See also:

              - :ref:`python-oindex-indexing`

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _Stride:
        __iter__ = None

        def __getitem__(
            self, strides: collections.abc.Iterable[int | None] | int | None
        ) -> DimExpression:
            """
            Strides the domains of the selected input dimensions by the specified amounts.

            For each selected dimension ``i``, the new domain is the set of indices ``x``
            such that :python:`x * strides[i]` is contained in the original domain.

            Examples:

               >>> transform = ts.IndexTransform(input_inclusive_min=[0, 2, 1],
               ...                               input_inclusive_max=[6, 5, 8],
               ...                               input_labels=["x", "y", "z"])
               >>> transform[ts.d["x", "z"].stride[-2, 3]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [-3, 1) "x"
                   1: [2, 6) "y"
                   2: [1, 3) "z"
                 Output index maps:
                   out[0] = 0 + -2 * in[0]
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 3 * in[2]
               >>> transform[ts.d["x", "z"].stride[3]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [0, 3) "x"
                   1: [2, 6) "y"
                   2: [1, 3) "z"
                 Output index maps:
                   out[0] = 0 + 3 * in[0]
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 3 * in[2]

            Note:

              :python:`expr.stride[strides]` is similar to the
              :ref:`NumPy-style slicing<python-indexing-interval>` operation
              :python:`expr[::strides]` except that the striding is always done with respect
              to an origin of 0, irrespective of the existing dimension lower bounds.

            The new dimension selection is the same as the prior dimension selection.

            Args:

              strides: Strides for each selected dimension.  May also be a scalar,
                e.g. :python:`2`, in which case the same stride value is used for all
                selected dimensions.  Specifying :python:`None` for a given dimension
                (equivalent to specifying a stride of :python:`1`) leaves that dimension
                unchanged.  Specify a stride of :python:`0` is not valid.

            Returns:
              Dimension expression with the striding operation added.

            Raises:

              IndexError:
                If the number strides does not match the number of selected dimensions.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> DimExpression:
            """
            Translates (shifts) the domains of the selected input dimensions backward by the
            specified offsets, without affecting the output range.

            Examples:

               >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4],
               ...                               input_shape=[4, 5, 6],
               ...                               input_labels=['x', 'y', 'z'])
               >>> transform[ts.d['x', 'y'].translate_backward_by[10, 20]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [-8, -4) "x"
                   1: [-17, -12) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = 10 + 1 * in[0]
                   out[1] = 20 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_backward_by[10, None]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [-8, -4) "x"
                   1: [3, 8) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = 10 + 1 * in[0]
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_backward_by[10]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [-8, -4) "x"
                   1: [-7, -2) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = 10 + 1 * in[0]
                   out[1] = 10 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]

            The new dimension selection is the same as the prior dimension selection.

            Args:

              offsets: The offsets for each of the selected dimensions.  May also be a
                scalar, e.g. :python:`5`, in which case the same offset is used for all
                selected dimensions.  Specifying :python:`None` for a given dimension
                (equivalent to specifying an offset of :python:`0`) leaves the origin of
                that dimension unchanged.

            Returns:
              Dimension expression with the translation operation added.

            Raises:

              IndexError:
                If the number origins does not match the number of selected dimensions.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> DimExpression:
            """
            Translates (shifts) the domains of the selected input dimensions by the
            specified offsets, without affecting the output range.

            Examples:

               >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4],
               ...                               input_shape=[4, 5, 6],
               ...                               input_labels=['x', 'y', 'z'])
               >>> transform[ts.d['x', 'y'].translate_by[10, 20]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [12, 16) "x"
                   1: [23, 28) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = -20 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_by[10, None]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [12, 16) "x"
                   1: [3, 8) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_by[10]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [12, 16) "x"
                   1: [13, 18) "y"
                   2: [4, 10) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = -10 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]

            The new dimension selection is the same as the prior dimension selection.

            Args:

              offsets: The offsets for each of the selected dimensions.  May also be a
                scalar, e.g. :python:`5`, in which case the same offset is used for all
                selected dimensions.  Specifying :python:`None` for a given dimension
                (equivalent to specifying an offset of :python:`0`) leaves the origin of
                that dimension unchanged.

            Returns:
              Dimension expression with the translation operation added.

            Raises:

              IndexError:
                If the number origins does not match the number of selected dimensions.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> DimExpression:
            """
            Translates the domains of the selected input dimensions to the specified
            origins without affecting the output range.

            Examples:

               >>> transform = ts.IndexTransform(input_shape=[4, 5, 6],
               ...                               input_labels=['x', 'y', 'z'])
               >>> transform[ts.d['x', 'y'].translate_to[10, 20]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [10, 14) "x"
                   1: [20, 25) "y"
                   2: [0, 6) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = -20 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_to[10, None]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [10, 14) "x"
                   1: [0, 5) "y"
                   2: [0, 6) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = 0 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]
               >>> transform[ts.d['x', 'y'].translate_to[10]]
               Rank 3 -> 3 index space transform:
                 Input domain:
                   0: [10, 14) "x"
                   1: [10, 15) "y"
                   2: [0, 6) "z"
                 Output index maps:
                   out[0] = -10 + 1 * in[0]
                   out[1] = -10 + 1 * in[1]
                   out[2] = 0 + 1 * in[2]

            The new dimension selection is the same as the prior dimension selection.

            Args:

              origins: The new origins for each of the selected dimensions.  May also be a
                scalar, e.g. :python:`5`, in which case the same origin is used for all
                selected dimensions.  If :python:`None` is specified for a given dimension,
                the origin of that dimension remains unchanged.

            Returns:
              Dimension expression with the translation operation added.

            Raises:

              IndexError:
                If the number origins does not match the number of selected dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _Transpose:
        __iter__ = None

        def __getitem__(self, target: DimSelectionLike) -> DimExpression:
            """
            Transposes the selected dimensions to the specified target indices.

            A dimension range may be specified to reverse the order of all dimensions:

                >>> transform = ts.IndexTransform(input_shape=[2, 3, 4],
                ...                               input_labels=["x", "y", "z"])
                >>> transform[ts.d[:].transpose[::-1]]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: [0, 4) "z"
                    1: [0, 3) "y"
                    2: [0, 2) "x"
                  Output index maps:
                    out[0] = 0 + 1 * in[2]
                    out[1] = 0 + 1 * in[1]
                    out[2] = 0 + 1 * in[0]

            Dimensions not in the selection retain their relative order and fill in the
            dimension indices not in :python:`target`:

                >>> transform = ts.IndexTransform(input_shape=[2, 3, 4],
                ...                               input_labels=["x", "y", "z"])
                >>> transform[ts.d['x', 'z'].transpose[0, 1]]
                Rank 3 -> 3 index space transform:
                  Input domain:
                    0: [0, 2) "x"
                    1: [0, 4) "z"
                    2: [0, 3) "y"
                  Output index maps:
                    out[0] = 0 + 1 * in[0]
                    out[1] = 0 + 1 * in[2]
                    out[2] = 0 + 1 * in[1]

            A single non-negative :python:`target` index may be specified to reorder all of
            the selected dimensions to start at the specified index:

                >>> transform = ts.IndexTransform(input_shape=[2, 3, 4, 5],
                ...                               input_labels=["a", "b", "c", "d"])
                >>> transform[ts.d['a', 'd'].transpose[1]]
                Rank 4 -> 4 index space transform:
                  Input domain:
                    0: [0, 3) "b"
                    1: [0, 2) "a"
                    2: [0, 5) "d"
                    3: [0, 4) "c"
                  Output index maps:
                    out[0] = 0 + 1 * in[1]
                    out[1] = 0 + 1 * in[0]
                    out[2] = 0 + 1 * in[3]
                    out[3] = 0 + 1 * in[2]

            A single negative :python:`target` index may be specified to order all of the
            selected dimensions to end at the specified index from end:

                >>> transform = ts.IndexTransform(input_shape=[2, 3, 4, 5],
                ...                               input_labels=["a", "b", "c", "d"])
                >>> transform[ts.d['a', 'd'].transpose[-1]]
                Rank 4 -> 4 index space transform:
                  Input domain:
                    0: [0, 3) "b"
                    1: [0, 4) "c"
                    2: [0, 2) "a"
                    3: [0, 5) "d"
                  Output index maps:
                    out[0] = 0 + 1 * in[2]
                    out[1] = 0 + 1 * in[0]
                    out[2] = 0 + 1 * in[1]
                    out[3] = 0 + 1 * in[3]

            Args:

              target: Target dimension indices for the selected dimensions.  All dimensions
                must be specified by index.  Labels are not permitted.  If the dimension
                selection has :python:`k > 1` dimensions, a single non-negative index
                :python:`i` is equivalent to :python:`i:i+k`; a single negative index
                :python:`-i` is equivalent to :python:`-i-k:-i`.

            Returns:
              Dimension expression with the transpose operation added.

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    class _Vindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> DimExpression:
            """
            Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`DimExpression.__getitem__`, but differs in that if
            :python:`indices` specifies any array indexing terms, the broadcasted array
            dimensions are unconditionally added as the first dimensions of the result
            domain:

            Examples:

               >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
               >>> transform[ts.d['y', 'z'].vindex[[1, 2, 3], [4, 5, 6]]]
               Rank 2 -> 3 index space transform:
                 Input domain:
                   0: [0, 3)
                   1: (-inf*, +inf*) "x"
                 Output index maps:
                   out[0] = 0 + 1 * in[1]
                   out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{1}, {2}, {3}}
                   out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{4}, {5}, {6}}

            Returns:
              Dimension expression with the indexing operation added.

            See also:

              - :ref:`python-vindex-indexing`

            Group:
              Operations
            """

        def __repr__(self) -> str: ...

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: DimExpression) -> bool: ...

    def __getitem__(self, indices: NumpyIndexingSpec) -> DimExpression:
        """
        Applies a :ref:`NumPy-style indexing operation<python-dim-expression-numpy-indexing>` with default index array semantics.

        When using NumPy-style indexing with a dimension expression, all selected
        dimensions must be consumed by a term of the indexing spec; there is no implicit
        addition of an `Ellipsis` term to consume any remaining dimensions.

        Returns:
          Dimension expression with the indexing operation added.

        Group:
          Operations

        Examples
        ========

        :ref:`Integer indexing<python-indexing-integer>`
        ------------------------------------------------

           >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
           >>> transform[ts.d['x'][5]]
           Rank 2 -> 3 index space transform:
             Input domain:
               0: (-inf*, +inf*) "y"
               1: (-inf*, +inf*) "z"
             Output index maps:
               out[0] = 5
               out[1] = 0 + 1 * in[0]
               out[2] = 0 + 1 * in[1]
           >>> transform[ts.d['x', 'z'][5, 6]]
           Rank 1 -> 3 index space transform:
             Input domain:
               0: (-inf*, +inf*) "y"
             Output index maps:
               out[0] = 5
               out[1] = 0 + 1 * in[0]
               out[2] = 6

        A single scalar index term applies to all selected dimensions:

           >>> transform[ts.d['x', 'y'][5]]
           Rank 1 -> 3 index space transform:
             Input domain:
               0: (-inf*, +inf*) "z"
             Output index maps:
               out[0] = 5
               out[1] = 5
               out[2] = 0 + 1 * in[0]

        .. seealso::

          :ref:`python-indexing-integer`

        :ref:`Interval indexing<python-indexing-interval>`
        --------------------------------------------------

           >>> transform = ts.IndexTransform(input_labels=['x', 'y', 'z'])
           >>> transform[ts.d['x'][5:10]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [5, 10) "x"
               1: (-inf*, +inf*) "y"
               2: (-inf*, +inf*) "z"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]
           >>> transform[ts.d['x', 'z'][5:10, 20:30]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [5, 10) "x"
               1: (-inf*, +inf*) "y"
               2: [20, 30) "z"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        As an extension, TensorStore allows the ``start``, ``stop``, and ``step``
        :py:obj:`python:slice` terms to be vectors rather than scalars:

           >>> transform[ts.d['x', 'z'][[5, 20]:[10, 30]]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [5, 10) "x"
               1: (-inf*, +inf*) "y"
               2: [20, 30) "z"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]
           >>> transform[ts.d['x', 'z'][[5, 20]:30]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [5, 30) "x"
               1: (-inf*, +inf*) "y"
               2: [20, 30) "z"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        As with integer indexing, a single scalar slice applies to all selected
        dimensions:

           >>> transform[ts.d['x', 'z'][5:30]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [5, 30) "x"
               1: (-inf*, +inf*) "y"
               2: [5, 30) "z"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        .. seealso::

          :ref:`python-indexing-interval`

        :ref:`Adding singleton dimensions<python-indexing-newaxis>`
        -----------------------------------------------------------

        Specifying a value of :py:obj:`.newaxis` (equal to `None`) adds a new
        inert/singleton dimension with :ref:`implicit bounds<implicit-bounds>`
        :math:`[0, 1)`:

           >>> transform = ts.IndexTransform(input_labels=['x', 'y'])
           >>> transform[ts.d[1][ts.newaxis]]
           Rank 3 -> 2 index space transform:
             Input domain:
               0: (-inf*, +inf*) "x"
               1: [0*, 1*)
               2: (-inf*, +inf*) "y"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[2]
           >>> transform[ts.d[0, -1][ts.newaxis, ts.newaxis]]
           Rank 4 -> 2 index space transform:
             Input domain:
               0: [0*, 1*)
               1: (-inf*, +inf*) "x"
               2: (-inf*, +inf*) "y"
               3: [0*, 1*)
             Output index maps:
               out[0] = 0 + 1 * in[1]
               out[1] = 0 + 1 * in[2]

        As with integer indexing, if only a single :python:`ts.newaxis` term is
        specified, it applies to all selected dimensions:

           >>> transform[ts.d[0, -1][ts.newaxis]]
           Rank 4 -> 2 index space transform:
             Input domain:
               0: [0*, 1*)
               1: (-inf*, +inf*) "x"
               2: (-inf*, +inf*) "y"
               3: [0*, 1*)
             Output index maps:
               out[0] = 0 + 1 * in[1]
               out[1] = 0 + 1 * in[2]

        :py:obj:`.newaxis` terms are only permitted in the first operation of a
        dimension expression, since in subsequent operations all dimensions of the
        dimension selection necessarily refer to existing dimensions:

        .. admonition:: Error
           :class: failure

           >>> transform[ts.d[0, 1].translate_by[5][ts.newaxis]]
           Traceback (most recent call last):
               ...
           IndexError: tensorstore.newaxis (`None`) not valid in chained indexing operations

        It is also an error to use :py:obj:`.newaxis` with dimensions specified by
        label:

        .. admonition:: Error
           :class: failure

           >>> transform[ts.d['x'][ts.newaxis]]
           Traceback (most recent call last):
               ...
           IndexError: New dimensions cannot be specified by label...

        .. seealso::

          :ref:`python-indexing-newaxis`

        :ref:`Ellipsis<python-indexing-ellipsis>`
        -----------------------------------------

        Specifying the special `Ellipsis` value (:python:`...`) is equivalent to
        specifying as many full slices :python:`:` as needed to consume the remaining
        selected dimensions not consumed by other indexing terms:

            >>> transform = ts.IndexTransform(input_rank=4)
            >>> transform[ts.d[:][1, ..., 5].translate_by[3]]
            Rank 2 -> 4 index space transform:
              Input domain:
                0: (-inf*, +inf*)
                1: (-inf*, +inf*)
              Output index maps:
                out[0] = 1
                out[1] = -3 + 1 * in[0]
                out[2] = -3 + 1 * in[1]
                out[3] = 5

        An indexing spec consisting solely of an `Ellipsis` term has no effect:

           >>> transform[ts.d[:][...]]
           Rank 4 -> 4 index space transform:
             Input domain:
               0: (-inf*, +inf*)
               1: (-inf*, +inf*)
               2: (-inf*, +inf*)
               3: (-inf*, +inf*)
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]
               out[3] = 0 + 1 * in[3]

        .. seealso::

          :ref:`python-indexing-ellipsis`

        :ref:`Integer array indexing<python-indexing-integer-array>`
        ------------------------------------------------------------

        Specifying an `~numpy.typing.ArrayLike` *index array* of integer values selects
        the coordinates given by the elements of the array of the selected dimension:

            >>> x = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
            >>> x = x[ts.d[:].label['x', 'y']]
            >>> x[ts.d['y'][[1, 1, 0]]]
            TensorStore({
              'array': [[2, 2, 1], [5, 5, 4]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [2, 3],
                'input_inclusive_min': [0, 0],
                'input_labels': ['x', ''],
              },
            })

        As in the example above, if only a single index array term is specified, the
        dimensions of the index array are added to the result domain in place of the
        selected dimension, consistent with
        :ref:`direct NumPy-style indexing<python-indexing-integer-array>` in the default
        index array mode.

        However, when using NumPy-style indexing with a dimension expression, if more
        than one index array term is specified, the broadcast dimensions of the index
        arrays are always added to the beginning of the result domain, i.e. exactly the
        behavior of :py:obj:`DimExpression.vindex`.  Unlike with direct NumPy-style
        indexing (not with a dimension expression), the behavior does not depend on
        whether the index array terms apply to consecutive dimensions, since consecutive
        dimensions are not well-defined for dimension expressions:

            >>> x = ts.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=ts.int32)
            >>> x = x[ts.d[:].label['x', 'y', 'z']]
            >>> x[ts.d['z', 'y'][[1, 0], [1, 1]]]
            TensorStore({
              'array': [[4, 3], [8, 7]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [2, 2],
                'input_inclusive_min': [0, 0],
                'input_labels': ['x', ''],
              },
            })

        .. seealso::

           :ref:`python-indexing-integer-array`

        :ref:`Boolean array indexing<python-indexing-boolean-array>`
        ------------------------------------------------------------

        Specifying an `~numpy.typing.ArrayLike` of `bool` values is equivalent to
        specifying a sequence of integer index arrays containing the coordinates of
        `True` values (in C order), e.g. as obtained from `numpy.nonzero`:

        Specifying a 1-d `bool` array is equivalent to a single index array of the
        non-zero coordinates:

            >>> x = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
            >>> x = x[ts.d[:].label['x', 'y']]
            >>> x[ts.d['y'][[False, True, True]]]
            TensorStore({
              'array': [[2, 3], [5, 6]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [2, 2],
                'input_inclusive_min': [0, 0],
                'input_labels': ['x', ''],
              },
            })

        Equivalently, using an index array:

            >>> x[ts.d['y'][[1, 2]]]
            TensorStore({
              'array': [[2, 3], [5, 6]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [2, 2],
                'input_inclusive_min': [0, 0],
                'input_labels': ['x', ''],
              },
            })

        More generally, specifying an ``n``-dimensional `bool` array is equivalent to
        specifying ``n`` 1-dimensional index arrays, where the ``i``\\ th index array specifies
        the ``i``\\ th coordinate of the `True` values:

            >>> x = ts.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            ...              dtype=ts.int32)
            >>> x = x[ts.d[:].label['x', 'y', 'z']]
            >>> x[ts.d['x', 'z'][[[True, False, False], [True, True, False]]]]
            TensorStore({
              'array': [[1, 4], [7, 10], [8, 11]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [3, 2],
                'input_inclusive_min': [0, 0],
                'input_labels': ['', 'y'],
              },
            })

        Equivalently, using an index array:

            >>> x[ts.d['x', 'z'][[0, 1, 1], [0, 0, 1]]]
            TensorStore({
              'array': [[1, 4], [7, 10], [8, 11]],
              'context': {'data_copy_concurrency': {}},
              'driver': 'array',
              'dtype': 'int32',
              'transform': {
                'input_exclusive_max': [3, 2],
                'input_inclusive_min': [0, 0],
                'input_labels': ['', 'y'],
              },
            })

        Note that as with integer array indexing, when using NumPy-styling indexing with
        a dimension expression, if boolean arrays are applied to more than one selected
        dimension, the added dimension corresponding to the `True` values is always
        added to the beginning of the result domain, i.e. exactly the behavior of
        :py:obj:`DimExpression.vindex`.

        .. seealso::

           :ref:`python-indexing-boolean-array`
        """

    def __repr__(self) -> str: ...

    @property
    def diagonal(self) -> DimExpression:
        """
        Extracts the diagonal of the selected dimensions.

        The selection dimensions are removed from the resultant index space, and a new
        dimension corresponding to the diagonal is added as the first dimension, with an
        input domain equal to the intersection of the input domains of the selection
        dimensions.  The new dimension selection is equal to :python:`ts.d[0]`,
        corresponding to the newly added diagonal dimension.

        The lower and upper bounds of the new diagonal dimension are
        :ref:`implicit<implicit-bounds>` if, and only if, the lower or upper bounds,
        respectively, of every selected dimension are implicit.

        Examples:

            >>> transform = ts.IndexTransform(input_shape=[2, 3],
            ...                               input_labels=["x", "y"])
            >>> transform[ts.d['x', 'y'].diagonal]
            Rank 1 -> 2 index space transform:
              Input domain:
                0: [0, 2)
              Output index maps:
                out[0] = 0 + 1 * in[0]
                out[1] = 0 + 1 * in[0]
            >>> transform = ts.IndexTransform(3)
            >>> transform[ts.d[0, 2].diagonal]
            Rank 2 -> 3 index space transform:
              Input domain:
                0: (-inf*, +inf*)
                1: (-inf*, +inf*)
              Output index maps:
                out[0] = 0 + 1 * in[0]
                out[1] = 0 + 1 * in[1]
                out[2] = 0 + 1 * in[0]

        Note:

          If zero dimensions are selected, :py:obj:`.diagonal` simply results in a new singleton
          dimension as the first dimension, equivalent to :python:`[ts.newaxis]`:

          >>> transform = ts.IndexTransform(1)
          >>> transform[ts.d[()].diagonal]
          Rank 2 -> 1 index space transform:
            Input domain:
              0: (-inf*, +inf*)
              1: (-inf*, +inf*)
            Output index maps:
              out[0] = 0 + 1 * in[1]

          If only one dimension is selected, :py:obj:`.diagonal` is equivalent to
          :python:`.label[''].transpose[0]`:

          >>> transform = ts.IndexTransform(input_labels=['x', 'y'])
          >>> transform[ts.d[1].diagonal]
          Rank 2 -> 2 index space transform:
            Input domain:
              0: (-inf*, +inf*)
              1: (-inf*, +inf*) "x"
            Output index maps:
              out[0] = 0 + 1 * in[1]
              out[1] = 0 + 1 * in[0]

        Group:
          Operations
        """

    @property
    def label(self) -> DimExpression._Label: ...

    @property
    def mark_bounds_implicit(self) -> DimExpression._MarkBoundsImplicit: ...

    @property
    def oindex(self) -> DimExpression._Oindex: ...

    @property
    def stride(self) -> DimExpression._Stride: ...

    @property
    def translate_backward_by(self) -> DimExpression._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> DimExpression._TranslateBy: ...

    @property
    def translate_to(self) -> DimExpression._TranslateTo: ...

    @property
    def transpose(self) -> DimExpression._Transpose: ...

    @property
    def vindex(self) -> DimExpression._Vindex: ...


class DimSelection(DimExpression):
    """

    Specifies a dimension selection, for starting a :ref:`dimension expression<python-dim-expressions>`.

    A dimension selection specifies a sequence of dimensions, either by index or
    :ref:`label<dimension-labels>`.

    :ref:`python-dim-selections` may be used as part of a
    :ref:`dimension expression<python-dim-expression-construction>` to specify the
    dimensions to which an indexing operation applies.

    A `DimSelection` may be constructed by subscripting `tensorstore.d`:

    Examples:

       >>> ts.d[0, 1, 2]
       d[0,1,2]
       >>> ts.d[0:1, 2, "x"]
       d[0:1,2,'x']
       >>> ts.d[[0, 1], [2]]
       d[0,1,2]
       >>> ts.d[[0, 1], ts.d[2, 3]]
       d[0,1,2,3]

    Group:
      Indexing

    Operations
    ==========

    """

    __hash__: typing.ClassVar[None] = None
    __iter__ = None

    def __eq__(self, other: DimSelection) -> bool: ...


class Future(typing.Generic[T]):
    """

    Handle for *consuming* the result of an asynchronous operation.

    This type supports several different patterns for consuming results:

    - Asynchronously with :py:mod:`asyncio`, using the :ref:`await<python:await>`
      keyword:

          >>> future = ts.open({
          ...     'driver': 'array',
          ...     'array': [1, 2, 3],
          ...     'dtype': 'uint32'
          ... })
          >>> await future
          TensorStore({
            'array': [1, 2, 3],
            'context': {'data_copy_concurrency': {}},
            'driver': 'array',
            'dtype': 'uint32',
            'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
          })

    - Synchronously blocking the current thread, by calling :py:meth:`.result()`.

          >>> future = ts.open({
          ...     'driver': 'array',
          ...     'array': [1, 2, 3],
          ...     'dtype': 'uint32'
          ... })
          >>> future.result()
          TensorStore({
            'array': [1, 2, 3],
            'context': {'data_copy_concurrency': {}},
            'driver': 'array',
            'dtype': 'uint32',
            'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
          })

    - Asynchronously, by registering a callback using :py:meth:`.add_done_callback`:

          >>> future = ts.open({
          ...     'driver': 'array',
          ...     'array': [1, 2, 3],
          ...     'dtype': 'uint32'
          ... })
          >>> future.add_done_callback(
          ...     lambda f: print(f'Callback: {f.result().domain}'))
          ... future.force()  # ensure the operation is started
          ... # wait for completion (for testing only)
          ... result = future.result()
          Callback: { [0, 3) }

    If an error occurs, instead of returning a value, :py:obj:`.result()` or
    :ref:`await<python:await>` will raise an exception.

    This type supports a subset of the interfaces of
    :py:class:`python:concurrent.futures.Future` and
    :py:class:`python:asyncio.Future`.  Unlike those types, however,
    :py:class:`Future` provides only the *consumer* interface.  The corresponding
    *producer* interface is provided by :py:class:`Promise`.

    .. warning::

       While this class is designed to interoperate with :py:mod:`asyncio`, it
       cannot be used with functions such as :py:obj:`asyncio.wait` that require an
       :py:class:`python:asyncio.Future`, because :py:obj:`.add_done_callback` does
       not guarantee that the callback is invoked from the current event loop.  To
       convert to a real :py:class:`python:asyncio.Future`, use
       :py:obj:`python:asyncio.ensure_future`:

           >>> dataset = await ts.open({
           ...     'driver': 'zarr',
           ...     'kvstore': 'memory://'
           ... },
           ...                         dtype=ts.uint32,
           ...                         shape=[70, 80],
           ...                         create=True)
           >>> await asyncio.wait([
           ...     asyncio.ensure_future(dataset[i * 5].write(i))
           ...     for i in range(10)
           ... ])

    See also:
      - :py:class:`WriteFutures`

    Type parameters:
      T:
        Result type of the asynchronous operation.

    Group:
      Asynchronous support
    """

    def __await__(self) -> collections.abc.Generator[typing.Any, None, T]: ...

    def __new__(
        self, future: FutureLike, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> Future:
        """
        Converts a :py:obj:`.FutureLike` object to a :py:obj:`.Future`.

        Example:

            >>> await ts.Future(3)
            3

            >>> x = ts.Future(3)
            >>> assert x is ts.Future(x)

            >>> async def get_value():
            ...     return 42
            >>> x = ts.Future(get_value())
            >>> x.done()
            False
            >>> await x
            >>> x.result()
            42

        Args:
          future: Specifies the immediate or asynchronous result.

          loop: Event loop on which to run :py:param:`.future` if it is a
            :ref:`coroutine<async>`.  If not specified (or :py:obj:`None` is specified),
            defaults to the loop returned by :py:obj:`asyncio.get_running_loop`.  If
            :py:param:`.loop` is not specified and there is no running event loop, it is
            an error for :py:param:`.future` to be a coroutine.

        Returns:

          - If :py:param:`.future` is a :py:obj:`.Future`, it is simply returned as is.

          - If :py:param:`.future` is a :ref:`coroutine<async>`, it is run using
            :py:param:`.loop` and the returned :py:obj:`.Future` corresponds to the
            asynchronous result.

          - Otherwise, :py:param:`.future` is treated as an immediate result, and the
            returned :py:obj:`.Future` resolves immediately to :py:param:`.future`.

        Warning:

          If :py:param:`.future` is a :ref:`coroutine<async>`, a blocking call to
          :py:obj:`Future.result` or :py:obj:`Future.exception` in the thread running
          the associated event loop may lead to deadlock.  Blocking calls should be
          avoided when using an event loop.
        """

    def add_done_callback(self, callback: typing.Callable[[Future[T]], None]) -> None:
        """
        Registers a callback to be invoked upon completion of the asynchronous operation.

        Args:
          callback: Callback to invoke with :python:`self` when this future becomes
            ready.

        .. warning::

           Unlike :py:obj:`python:asyncio.Future.add_done_callback`, but like
           :py:obj:`python:concurrent.futures.Future.add_done_callback`, the
           :py:param:`.callback` may be invoked from any thread.  If using
           :py:mod:`asyncio` and :py:param:`.callback` needs to be invoked from a
           particular event loop, wrap :py:param:`.callback` with
           :py:obj:`python:asyncio.loop.call_soon_threadsafe`.

        Group:
          Callback interface
        """

    def cancel(self) -> bool:
        """
        Requests cancellation of the asynchronous operation.

        If the operation has not already completed, it is marked as unsuccessfully
        completed with an instance of :py:obj:`asyncio.CancelledError`.

        Group:
          Operations
        """

    def cancelled(self) -> bool:
        """
        Queries whether the asynchronous operation has been cancelled.

        Example:

            >>> promise, future = ts.Promise.new()
            >>> future.cancelled()
            False
            >>> future.cancel()
            >>> future.cancelled()
            True
            >>> future.exception()
            Traceback (most recent call last):
                ...
            ...CancelledError...

        Group:
          Accessors
        """

    def done(self) -> bool:
        """
        Queries whether the asynchronous operation has completed or been cancelled.

        Group:
          Accessors
        """

    def exception(
        self, timeout: float | None = None, deadline: float | None = None
    ) -> typing.Any:
        """
        Blocks until asynchronous operation completes, and returns the error if any.

        Args:
          timeout: Maximum number of seconds to block.
          deadline: Deadline in seconds since the Unix epoch.

        Returns:

          The error that was produced by the asynchronous operation, or :py:obj:`None`
          if the operation completed successfully.

        Raises:

          TimeoutError: If the result did not become ready within the specified
            :py:param:`.timeout` or :py:param:`.deadline`.

          KeyboardInterrupt: If running on the main thread and a keyboard interrupt is
            received.

        Group:
          Blocking interface
        """

    def force(self) -> None:
        """
        Ensures the asynchronous operation begins executing.

        This is called automatically by :py:obj:`.result` and :py:obj:`.exception`, but
        must be called explicitly when using :py:obj:`.add_done_callback`.

        Group:
          Operations
        """

    def remove_done_callback(self, callback: typing.Callable[[Future[T]], None]) -> int:
        """
        Unregisters a previously-registered callback.

        Group:
          Callback interface
        """

    def result(self, timeout: float | None = None, deadline: float | None = None) -> T:
        """
        Blocks until the asynchronous operation completes, and returns the result.

        If the asynchronous operation completes unsuccessfully, raises the error that
        was produced.

        Args:
          timeout: Maximum number of seconds to block.
          deadline: Deadline in seconds since the Unix epoch.

        Returns:
          The result of the asynchronous operation, if successful.

        Raises:

          TimeoutError: If the result did not become ready within the specified
            :py:param:`.timeout` or :py:param:`.deadline`.

          KeyboardInterrupt: If running on the main thread and a keyboard interrupt is
            received.

        Group:
          Blocking interface
        """


class IndexDomain:
    """

    :ref:`Domain<index-domain>` (including bounds and optional dimension labels) of an N-dimensional :ref:`index space<index-space>`.

    Logically, an :py:class:`.IndexDomain` is the cartesian product of a sequence of
    `Dim` objects, and supports the :py:obj:`~collections.abc.Collection`
    interface.

    Note:

       Index domains are immutable, but
       :ref:`dimension expressions<python-dim-expressions>` may be applied using
       :py:obj:`.__getitem__(expr)` to obtain a modified domain.

    See also:
      - :py:obj:`IndexTransform`, which define a class of functions for index domains.
      - The :json:schema:`JSON representation<IndexDomain>`.

    Group:
      Indexing
    """

    class _Label:
        __iter__ = None

        def __getitem__(
            self, labels: str | collections.abc.Iterable[str]
        ) -> IndexDomain:
            """
            Returns a new view with the :ref:`dimension labels<dimension-labels>` changed.

            This is equivalent to :python:`self[ts.d[:].label[labels]]`.

            Args:
              labels: Dimension labels for each dimension.

            Raises:

              IndexError: If the number of labels does not match the number of dimensions,
                or if the resultant domain would have duplicate labels.

            See also:
              - `tensorstore.DimExpression.label`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> IndexDomain:
            """
            Returns a new view with the lower/upper bounds changed to
            :ref:`implicit/explicit<implicit-bounds>`.

            This is equivalent to :python:`self[ts.d[:].mark_bounds_implicit[implicit]]`.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.
                Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Raises:

              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            See also:
              - `tensorstore.DimExpression.mark_bounds_implicit`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> IndexDomain:
            """
            Returns a new view with the `.origin` translated backward by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_backward_by[offsets]]`.

            Args:

              offsets: The offset for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_backward_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> IndexDomain:
            """
            Returns a new view with the `.origin` translated by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_by[offsets]]`.

            Args:

              offsets: The offset for each dimension.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> IndexDomain:
            """
            Returns a new view with `.origin` translated to the specified origin.

            This is equivalent to :python:`self[ts.d[:].translate_to[origins]]`.

            Args:

              origins: The new origin for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same origin is used for all dimensions.
                If :python:`None` is specified for a given dimension, the origin of that
                dimension remains unchanged.

            Raises:

              IndexError:
                If the number origins does not match the number of dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            See also:
              - `tensorstore.DimExpression.translate_to`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    __hash__: typing.ClassVar[None] = None

    def __copy__(self) -> IndexDomain: ...

    def __deepcopy__(self, memo: dict) -> IndexDomain: ...

    def __eq__(self, arg0: IndexDomain) -> bool: ...

    @typing.overload
    def __getitem__(self, identifier: int | str) -> Dim:
        """
        Returns the single dimension specified by :python:`identifier`.

        Args:
          identifier: Specifies a dimension by integer index or label.  As with
              :py:obj:`python:list`, a negative index specifies a dimension starting
              from the last dimension.

        Examples:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                         exclusive_max=[4, 5, 6],
            ...                         labels=['x', 'y', 'z'])
            >>> domain[0]
            Dim(inclusive_min=1, exclusive_max=4, label="x")
            >>> domain['y']
            Dim(inclusive_min=2, exclusive_max=5, label="y")
            >>> domain[-1]
            Dim(inclusive_min=3, exclusive_max=6, label="z")

        Overload:
          identifier

        Group:
          Sequence accessors
        """

    @typing.overload
    def __getitem__(self, selection: DimSelectionLike) -> IndexDomain:
        """
        Returns a new domain with a subset of the dimensions.

        Args:

          selection: Specifies the dimensions to include, either by index or label.  May
              be any value or sequence of values convertible to a
              :ref:`dimension selection<python-dim-selections>`.

        Raises:
           ValueError: If any dimension is specified more than once.

        Examples:

            >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                    exclusive_max=[4, 5, 6],
            ...                    labels=['x', 'y', 'z'])
            >>> a[:2]
            { "x": [1, 4), "y": [2, 5) }
            >>> a[0, -1]
            { "x": [1, 4), "z": [3, 6) }
            >>> a['y', 'x']
            { "y": [2, 5), "x": [1, 4) }
            >>> a['y', 1]
            Traceback (most recent call last):
                ...
            ValueError: Input dimensions {1} specified more than once

        Overload:
          selection

        Group:
          Indexing
        """

    @typing.overload
    def __getitem__(self, other: IndexDomain) -> IndexDomain:
        """
        Slices this domain by another domain.

        The result is determined by matching dimensions of :python:`other` to dimensions
        of :python:`self` either by label or by index, according to one of the following
        three cases:

        .. list-table::
           :widths: auto

           * - :python:`other` is entirely unlabeled

             - Result is
               :python:`self[ts.d[:][other.inclusive_min:other.exclusive_max]`.
               It is an error if :python:`self.rank != other.rank`.

           * - :python:`self` is entirely unlabeled

             - Result is
               :python:`self[ts.d[:][other.inclusive_min:other.exclusive_max].labels[other.labels]`.
               It is an error if :python:`self.rank != other.rank`.

           * - Both :python:`self` and :python:`other` have at least one labeled dimension.

             - Result is
               :python:`self[ts.d[dims][other.inclusive_min:other.exclusive_max]`, where
               the sequence of :python:`other.rank` dimension identifiers :python:`dims`
               is determined as follows:

               1. If :python:`other.labels[i]` is specified (i.e. non-empty),
                  :python:`dims[i] = self.labels.index(other.labels[i])`.  It is an
                  error if no such dimension exists.

               2. Otherwise, ``i`` is the ``j``\\ th unlabeled dimension of :python:`other`
                  (left to right), and :python:`dims[i] = k`, where ``k`` is the ``j``\\ th
                  unlabeled dimension of :python:`self` (left to right).  It is an error
                  if no such dimension exists.

               If any dimensions of :python:`other` are unlabeled, then it is an error
               if :python:`self.rank != other.rank`.  This condition is not strictly
               necessary but serves to avoid a discrepancy in behavior with normal
               :ref:`domain alignment<index-domain-alignment>`.

        .. admonition:: Example with all unlabeled dimensions
           :class: example

           >>> a = ts.IndexDomain(inclusive_min=[0, 1], exclusive_max=[5, 7])
           >>> b = ts.IndexDomain(inclusive_min=[2, 3], exclusive_max=[4, 6])
           >>> a[b]
           { [2, 4), [3, 6) }

        .. admonition:: Example with fully labeled dimensions
           :class: example

           >>> a = ts.IndexDomain(inclusive_min=[0, 1, 2],
           ...                    exclusive_max=[5, 7, 8],
           ...                    labels=["x", "y", "z"])
           >>> b = ts.IndexDomain(inclusive_min=[2, 3],
           ...                    exclusive_max=[6, 4],
           ...                    labels=["y", "x"])
           >>> a[b]
           { "x": [3, 4), "y": [2, 6), "z": [2, 8) }

        .. admonition:: Example with mixed labeled and unlabeled dimensions
           :class: example

           >>> a = ts.IndexDomain(inclusive_min=[0, 0, 0, 0],
           ...                    exclusive_max=[10, 10, 10, 10],
           ...                    labels=["x", "", "", "y"])
           >>> b = ts.IndexDomain(inclusive_min=[1, 2, 3, 4],
           ...                    exclusive_max=[6, 7, 8, 9],
           ...                    labels=["y", "", "x", ""])
           >>> a[b]
           { "x": [3, 8), [2, 7), [4, 9), "y": [1, 6) }

        Note:

          On :python:`other`, :ref:`implicit bounds<implicit-bounds>` indicators have no
          effect.

        Overload:
          domain

        Group:
          Indexing
        """

    @typing.overload
    def __getitem__(self, expr: DimExpression) -> IndexDomain:
        """
        Transforms the domain by a :ref:`dimension expression<python-dim-expressions>`.

        Args:
          expr: :ref:`Dimension expression<python-dim-expressions>` to apply.

        Examples:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                         exclusive_max=[6, 7, 8],
            ...                         labels=['x', 'y', 'z'])
            >>> domain[ts.d[:].translate_by[5]]
            { "x": [6, 11), "y": [7, 12), "z": [8, 13) }
            >>> domain[ts.d['y'][3:5]]
            { "x": [1, 6), "y": [3, 5), "z": [3, 8) }
            >>> domain[ts.d['z'][5]]
            { "x": [1, 6), "y": [2, 7) }

        Note:

           For the purpose of applying a dimension expression, an
           :py:class:`IndexDomain` behaves like an :py:class:`IndexTransform` with an
           output rank of 0.  Consequently, operations that primarily affect the output
           index mappings, like
           :ref:`integer array indexing<python-indexing-integer-array>`, are not very
           useful, though they are still permitted.

               >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
               ...                         exclusive_max=[6, 7, 8],
               ...                         labels=['x', 'y', 'z'])
               >>> domain[ts.d['z'][[3, 5, 7]]]
               { "x": [1, 6), "y": [2, 7), [0, 3) }

        Overload:
          expr

        Group:
          Indexing
        """

    @typing.overload
    def __getitem__(self, transform: IndexTransform) -> IndexDomain:
        """
        Transforms the domain using an explicit :ref:`index transform<index-transform>`.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                         exclusive_max=[6, 7, 8])
            >>> transform = ts.IndexTransform(
            ...     input_rank=4,
            ...     output=[
            ...         ts.OutputIndexMap(offset=5, input_dimension=3),
            ...         ts.OutputIndexMap(offset=-7, input_dimension=0),
            ...         ts.OutputIndexMap(offset=3, input_dimension=1),
            ...     ])
            >>> domain[transform]
            { [9, 14), [0, 5), (-inf*, +inf*), [-4, 1) }

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.

        Returns:

          New domain of rank :python:`transform.input_rank`.

        Note:

           This is equivalent to composing an identity transform over :python:`self`
           with :py:param:`.transform`,
           i.e. :python:`ts.IndexTransform(self)[transform].domain`.  Consequently,
           operations that primarily affect the output index mappings, like
           :ref:`integer array indexing<python-indexing-integer-array>`, are not very
           useful, though they are still permitted.

        Overload:
          transform

        Group:
          Indexing
        """

    @typing.overload
    def __init__(
        self,
        rank: int | None = None,
        *,
        inclusive_min: collections.abc.Iterable[int] | None = None,
        implicit_lower_bounds: collections.abc.Iterable[bool] | None = None,
        exclusive_max: collections.abc.Iterable[int] | None = None,
        inclusive_max: collections.abc.Iterable[int] | None = None,
        shape: collections.abc.Iterable[int] | None = None,
        implicit_upper_bounds: collections.abc.Iterable[bool] | None = None,
        labels: collections.abc.Iterable[str | None] | None = None,
    ) -> None:
        """
        Constructs an index domain from component vectors.

        Args:
          rank: Number of dimensions.  Only required if no other parameter is specified.
          inclusive_min: Inclusive lower bounds for each dimension.  If not specified,
              defaults to all zero if ``shape`` is specified, otherwise unbounded.
          implicit_lower_bounds: Indicates whether each lower bound is
              :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
              ``inclusive_min`` or ``shape`` is specified, otherwise defaults to all
              implicit.
          exclusive_max: Exclusive upper bounds for each dimension.  At most one of
              ``exclusive_max``, ``inclusive_max``, and ``shape`` may be specified.
          inclusive_max: Inclusive upper bounds for each dimension.
          shape: Size for each dimension.
          implicit_upper_bounds: Indicates whether each upper bound is
              :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
              ``exclusive_max``, ``inclusive_max``, or ``shape`` is specified, otherwise
              defaults to all implicit.
          labels: :ref:`Dimension labels<dimension-labels>`.  Defaults to all unlabeled.

        Examples:

            >>> ts.IndexDomain(rank=5)
            { (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*), (-inf*, +inf*) }
            >>> ts.IndexDomain(shape=[2, 3])
            { [0, 2), [0, 3) }

        Overload:
          components
        """

    @typing.overload
    def __init__(self, dimensions: collections.abc.Iterable[Dim]) -> None:
        """
        Constructs an index domain from a :py:class`.Dim` sequence.

        Args:
          dimensions: :py:obj:`Sequence<python:typing.Sequence>` of :py:class`.Dim` objects.

        Examples:

            >>> ts.IndexDomain([ts.Dim(5), ts.Dim(6, label='y')])
            { [0, 5), "y": [0, 6) }

        Overload:
          dimensions
        """

    @typing.overload
    def __init__(self, *, json: typing.Any) -> None:
        """
        Constructs an index domain from its :json:schema:`JSON representation<IndexDomain>`.

        Examples:

            >>> ts.IndexDomain(
            ...     json={
            ...         "inclusive_min": ["-inf", 7, ["-inf"], [8]],
            ...         "exclusive_max": ["+inf", 10, ["+inf"], [17]],
            ...         "labels": ["x", "y", "z", ""]
            ...     })
            { "x": (-inf, +inf), "y": [7, 10), "z": (-inf*, +inf*), [8*, 17*) }

        Overload:
          json
        """

    def __len__(self) -> int:
        """
        Returns the number of dimensions (:py:obj:`.rank`).

        Example:

          >>> domain = ts.IndexDomain(shape=[100, 200, 300])
          >>> len(domain)
          3

        Group:
          Sequence accessors
        """

    def __repr__(self) -> str:
        """
        Returns the string representation.
        """

    def hull(self, other: IndexDomain) -> IndexDomain:
        """
        Computes the hull (minimum containing box) with another domain.

        The ``implicit`` flag that corresponds to the selected bound is propagated.

        Args:
          other: Object to hull with.

        Example:

            >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                    exclusive_max=[4, 5, 6],
            ...                    labels=['x', 'y', ''])
            >>> a.hull(ts.IndexDomain(shape=[2, 3, 4]))
            { "x": [0, 4), "y": [0, 5), [0, 6) }

        Group:
          Geometric operations
        """

    def intersect(self, other: IndexDomain) -> IndexDomain:
        """
        Intersects with another domain.

        The ``implicit`` flag that corresponds to the selected bound is propagated.

        Args:
          other: Object to intersect with.

        Example:

            >>> a = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                    exclusive_max=[4, 5, 6],
            ...                    labels=['x', 'y', ''])
            >>> a.intersect(ts.IndexDomain(shape=[2, 3, 4]))
            { "x": [1, 2), "y": [2, 3), [3, 4) }

        Group:
          Geometric operations
        """

    def to_json(self) -> typing.Any:
        """
        Returns the :json:schema:`JSON representation<IndexDomain>`.

        Group:
          Accessors
        """

    def transpose(self, axes: DimSelectionLike | None = None) -> IndexDomain:
        """
        Returns a view with a transposed domain.

        This is equivalent to :python:`self[ts.d[axes].transpose[:]]`.

        Args:

          axes: Specifies the existing dimension corresponding to each dimension of the
            new view.  Dimensions may be specified either by index or label.  Specifying
            `None` is equivalent to specifying :python:`[rank-1, ..., 0]`, which
            reverses the dimension order.

        Raises:

          ValueError: If :py:param:`.axes` does not specify a valid permutation.

        See also:
          - `tensorstore.DimExpression.transpose`
          - :py:obj:`.T`

        Group:
          Indexing
        """

    @property
    def T(self) -> IndexDomain:
        """
        View with the dimension order reversed (transposed).

        Example:

            >>> domain = ts.IndexDomain(labels=['x', 'y', 'z'])
            >>> domain.T
            { "z": (-inf*, +inf*), "y": (-inf*, +inf*), "x": (-inf*, +inf*) }

        See also:
          - `.transpose`
          - `tensorstore.DimExpression.transpose`

        Group:
          Indexing
        """

    @property
    def exclusive_max(self) -> tuple[int, ...]:
        """
        Exclusive upper bound of the domain.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.exclusive_max
            (4, 6, 8)

        Group:
          Accessors
        """

    @property
    def implicit_lower_bounds(self) -> tuple[bool, ...]:
        """
        Indicates whether the lower bound of each dimension is :ref:`implicit or explicit<implicit-bounds>`.

        Example:

            >>> domain = ts.IndexDomain(rank=3)
            >>> domain.implicit_lower_bounds
            (True, True, True)
            >>> domain = ts.IndexDomain(inclusive_min=[2, 3, 4])
            >>> domain.implicit_lower_bounds
            (False, False, False)
            >>> domain = ts.IndexDomain(exclusive_max=[2, 3, 4])
            >>> domain.implicit_lower_bounds
            (True, True, True)
            >>> domain = ts.IndexDomain(shape=[4, 5, 6])
            >>> domain.implicit_lower_bounds
            (False, False, False)
            >>> domain = ts.IndexDomain(inclusive_min=[4, 5, 6],
            ...                         implicit_lower_bounds=[False, True, False])
            >>> domain.implicit_lower_bounds
            (False, True, False)

        Group:
          Accessors
        """

    @property
    def implicit_upper_bounds(self) -> tuple[bool, ...]:
        """
        Indicates whether the upper bound of each dimension is :ref:`implicit or explicit<implicit-bounds>`.

        Example:

            >>> domain = ts.IndexDomain(rank=3)
            >>> domain.implicit_upper_bounds
            (True, True, True)
            >>> domain = ts.IndexDomain(shape=[2, 3, 4])
            >>> domain.implicit_upper_bounds
            (False, False, False)
            >>> domain = ts.IndexDomain(inclusive_min=[4, 5, 6])
            >>> domain.implicit_upper_bounds
            (True, True, True)
            >>> domain = ts.IndexDomain(exclusive_max=[4, 5, 6],
            ...                         implicit_upper_bounds=[False, True, False])
            >>> domain.implicit_upper_bounds
            (False, True, False)

        Group:
          Accessors
        """

    @property
    def inclusive_max(self) -> tuple[int, ...]:
        """
        Inclusive upper bound of the domain.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.inclusive_max
            (3, 5, 7)

        Group:
          Accessors
        """

    @property
    def inclusive_min(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain, alias of :py:obj:`.origin`.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.inclusive_min
            (1, 2, 3)

        Group:
          Accessors
        """

    @property
    def index_exp(self) -> tuple[slice, ...]:
        """
        Equivalent NumPy-compatible :py:obj:`index expression<numpy.s_>`.

        The index expression consists of a :py:obj:`tuple` of :py:obj:`.rank`
        :py:obj:`slice` objects that specify the lower and upper bounds for each
        dimension, where an infinite bound in the domain corresponds to a bound of
        :py:obj:`None` in the :py:obj:`slice` object.

        The index expression can be used with this library as a
        :ref:`NumPy-style indexing expression<python-numpy-style-indexing>` or to
        directly index a `NumPy array<numpy.ndarray>`.

        Example:

            >>> ts.IndexDomain(rank=2).index_exp
            (slice(None, None, None), slice(None, None, None))
            >>> ts.IndexDomain(inclusive_min=[1, 2], exclusive_max=[5, 10]).index_exp
            (slice(1, 5, None), slice(2, 10, None))
            >>> arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> domain = ts.IndexDomain(inclusive_min=[0, 2], shape=[3, 2])
            >>> arr[domain.index_exp]
            array([[3, 4],
                   [8, 9]])

        Raises:
          ValueError: If any finite bound in :py:obj:`.inclusive_min` or
            :py:obj:`.exclusive_max` is negative.  In this case the index expression
            would not actually NumPy-compatible since NumPy does not support actual
            negative indices, and instead interprets negative numbers as counting from
            the end.

        Group:
          Accessors
        """

    @property
    def label(self) -> IndexDomain._Label: ...

    @property
    def labels(self) -> tuple[str, ...]:
        """
        :ref:`Dimension labels<dimension-labels>` for each dimension.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.labels
            ('', '', '')
            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                         shape=[3, 4, 5],
            ...                         labels=['x', 'y', 'z'])
            >>> domain.labels
            ('x', 'y', 'z')

        Group:
          Accessors
        """

    @property
    def mark_bounds_implicit(self) -> IndexDomain._MarkBoundsImplicit: ...

    @property
    def ndim(self) -> int:
        """
        Alias for :py:obj:`.rank`.

        Example:

          >>> domain = ts.IndexDomain(shape=[100, 200, 300])
          >>> domain.ndim
          3

        Group:
          Accessors
        """

    @property
    def origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.origin
            (1, 2, 3)

        Group:
          Accessors
        """

    @property
    def rank(self) -> int:
        """
        Number of dimensions in the index space.

        Example:

          >>> domain = ts.IndexDomain(shape=[100, 200, 300])
          >>> domain.rank
          3

        Group:
          Accessors
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the domain.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.shape
            (3, 4, 5)

        Group:
          Accessors
        """

    @property
    def size(self) -> int:
        """
        Total number of elements in the domain.

        This is simply the product of the extents in :py:obj:`.shape`.

        Example:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3], shape=[3, 4, 5])
            >>> domain.size
            60

        Group:
          Accessors
        """

    @property
    def translate_backward_by(self) -> IndexDomain._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> IndexDomain._TranslateBy: ...

    @property
    def translate_to(self) -> IndexDomain._TranslateTo: ...

    def __iter__(self) -> collections.abc.Iterator[Dim]: ...


class IndexTransform(Indexable):
    """

    Represents a transform from an input index space to an output space.

    The :ref:`index transform abstraction<index-transform>` underlies all indexing
    operations in the TensorStore library, and enables fully-composable virtual
    views.  For many common use cases cases, however, it does not need to be used
    directly; instead, it is used indirectly through
    :ref:`indexing operations<python-indexing>` on the :py:class:`TensorStore` class
    and other :py:class:`Indexable` types.

    See also:
      - :py:obj:`IndexDomain`, which represents the domain of an index transform.
      - The :json:schema:`JSON representation<IndexTransform>`.

    Group:
      Indexing

    Constructors
    ============

    Accessors
    =========

    Indexing
    ========

    """

    class _Label:
        __iter__ = None

        def __getitem__(
            self, labels: str | collections.abc.Iterable[str]
        ) -> IndexTransform:
            """
            Returns a new view with the :ref:`dimension labels<dimension-labels>` changed.

            This is equivalent to :python:`self[ts.d[:].label[labels]]`.

            Args:
              labels: Dimension labels for each dimension.

            Raises:

              IndexError: If the number of labels does not match the number of dimensions,
                or if the resultant domain would have duplicate labels.

            See also:
              - `tensorstore.DimExpression.label`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> IndexTransform:
            """
            Returns a new view with the lower/upper bounds changed to
            :ref:`implicit/explicit<implicit-bounds>`.

            This is equivalent to :python:`self[ts.d[:].mark_bounds_implicit[implicit]]`.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.
                Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Raises:

              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            See also:
              - `tensorstore.DimExpression.mark_bounds_implicit`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Oindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> IndexTransform:
            """
            Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`IndexTransform.__getitem__(indices)`, but differs in
            that any integer or boolean array indexing terms are applied orthogonally:

            Example:

               >>> transform = ts.IndexTransform(3)
               >>> transform.oindex[2, [1, 2, 3], [6, 7, 8]]
               Rank 2 -> 3 index space transform:
                 Input domain:
                   0: [0, 3)
                   1: [0, 3)
                 Output index maps:
                   out[0] = 2
                   out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{1}, {2}, {3}}
                   out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {{6, 7, 8}}

            See also:

               - :ref:`python-numpy-style-indexing`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> IndexTransform:
            """
            Returns a new view with the `.origin` translated backward by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_backward_by[offsets]]`.

            Args:

              offsets: The offset for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_backward_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> IndexTransform:
            """
            Returns a new view with the `.origin` translated by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_by[offsets]]`.

            Args:

              offsets: The offset for each dimension.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> IndexTransform:
            """
            Returns a new view with `.origin` translated to the specified origin.

            This is equivalent to :python:`self[ts.d[:].translate_to[origins]]`.

            Args:

              origins: The new origin for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same origin is used for all dimensions.
                If :python:`None` is specified for a given dimension, the origin of that
                dimension remains unchanged.

            Raises:

              IndexError:
                If the number origins does not match the number of dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            See also:
              - `tensorstore.DimExpression.translate_to`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Vindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> IndexTransform:
            """
            Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`IndexTransform.__getitem__(indices)`, but differs in
            that if :python:`indices` specifies any array indexing terms, the broadcasted
            array dimensions are unconditionally added as the first dimensions of the result
            domain:

            Example:

               >>> transform = ts.IndexTransform(3)
               >>> transform.vindex[2, [1, 2, 3], [6, 7, 8]]
               Rank 1 -> 3 index space transform:
                 Input domain:
                   0: [0, 3)
                 Output index maps:
                   out[0] = 2
                   out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {1, 2, 3}
                   out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                     {6, 7, 8}

            See also:

               - :ref:`python-numpy-style-indexing`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    __hash__: typing.ClassVar[None] = None
    __iter__ = None

    def __call__(self, indices: collections.abc.Iterable[int]) -> tuple[int, ...]:
        """
        Maps an input index vector to an output index vector.

        Args:
          indices: Input vector of length :py:obj:`.input_rank`.

        Returns:
          Output vector of length :py:obj:`output_rank`.

        Examples:

            >>> transform = ts.IndexTransform(2)[ts.d[:].translate_by[1, 2]]
            >>> transform([0, 0])
            (-1, -2)
            >>> transform([1, 2])
            (0, 0)

        Group:
          Indexing
        """

    def __copy__(self) -> IndexTransform: ...

    def __deepcopy__(self, memo: dict) -> IndexTransform: ...

    def __eq__(self, other: IndexTransform) -> bool: ...

    @typing.overload
    def __getitem__(self, transform: IndexTransform) -> IndexTransform:
        """
        Composes this index transform with another index transform.

        The resultant transform maps :python:`x` to :python:`self(transform(x))`.

        Examples:

           >>> a = ts.IndexTransform(
           ...     input_rank=1,
           ...     output=[ts.OutputIndexMap(input_dimension=0, offset=5)])
           >>> b = ts.IndexTransform(
           ...     input_rank=1,
           ...     output=[ts.OutputIndexMap(input_dimension=0, offset=3)])
           >>> a[b]
           Rank 1 -> 1 index space transform:
             Input domain:
               0: (-inf*, +inf*)
             Output index maps:
               out[0] = 8 + 1 * in[0]

        Group:
          Indexing

        Overload:
          transform
        """

    @typing.overload
    def __getitem__(self, domain: IndexDomain) -> IndexTransform:
        """
        Slices this index transform by another domain.

        The result is determined by matching dimensions of :python:`domain` to
        dimensions of :python:`self.domain` either by label or by index, according to
        one of the following three cases:

        .. list-table::
           :widths: auto

           * - :python:`domain` is entirely unlabeled

             - Result is
               :python:`self[ts.d[:][domain.inclusive_min:domain.exclusive_max]`.  It is
               an error if :python:`self.input_rank != domain.rank`.

           * - :python:`self.domain` is entirely unlabeled

             - Result is
               :python:`self[ts.d[:][domain.inclusive_min:domain.exclusive_max].labels[domain.labels]`.
               It is an error if :python:`self.input_rank != domain.rank`.

           * - Both :python:`self.domain` and :python:`domain` have at least one labeled
               dimension.

             - Result is
               :python:`self[ts.d[dims][domain.inclusive_min:domain.exclusive_max]`,
               where the sequence of :python:`domain.rank` dimension identifiers
               :python:`dims` is determined as follows:

               1. If :python:`domain.labels[i]` is specified (i.e. non-empty),
                  :python:`dims[i] = self.input_labels.index(domain.labels[i])`.  It is
                  an error if no such dimension exists.

               2. Otherwise, ``i`` is the ``j``\\ th unlabeled dimension of
                  :python:`domain` (left to right), and :python:`dims[i] = k`, where
                  ``k`` is the ``j``\\ th unlabeled dimension of :python:`self` (left to
                  right).  It is an error if no such dimension exists.

               If any dimensions of :python:`domain` are unlabeled, then it is an error
               if :python:`self.input_rank != domain.rank`.  This condition is not
               strictly necessary but serves to avoid a discrepancy in behavior with
               normal :ref:`domain alignment<index-domain-alignment>`.

        .. admonition:: Example with all unlabeled dimensions
           :class: example

           >>> a = ts.IndexTransform(input_inclusive_min=[0, 1],
           ...                       input_exclusive_max=[5, 7])
           >>> b = ts.IndexDomain(inclusive_min=[2, 3], exclusive_max=[4, 6])
           >>> transform[domain]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [1, 4)
               1: [2, 5)
               2: [3, 6)
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        .. admonition:: Example with fully labeled dimensions
           :class: example

           >>> a = ts.IndexTransform(input_inclusive_min=[0, 1, 2],
           ...                       input_exclusive_max=[5, 7, 8],
           ...                       input_labels=["x", "y", "z"])
           >>> b = ts.IndexDomain(inclusive_min=[2, 3],
           ...                    exclusive_max=[6, 4],
           ...                    labels=["y", "x"])
           >>> transform[domain]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: [1, 4)
               1: [2, 5)
               2: [3, 6)
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        .. admonition:: Example with mixed labeled and unlabeled dimensions
           :class: example

           >>> a = ts.IndexTransform(input_inclusive_min=[0, 0, 0, 0],
           ...                       input_exclusive_max=[10, 10, 10, 10],
           ...                       input_labels=["x", "", "", "y"])
           >>> b = ts.IndexDomain(inclusive_min=[1, 2, 3, 4],
           ...                    exclusive_max=[6, 7, 8, 9],
           ...                    labels=["y", "", "x", ""])
           >>> a[b]
           Rank 4 -> 4 index space transform:
             Input domain:
               0: [3, 8) "x"
               1: [2, 7)
               2: [4, 9)
               3: [1, 6) "y"
             Output index maps:
               out[0] = 0 + 1 * in[0]
               out[1] = 0 + 1 * in[1]
               out[2] = 0 + 1 * in[2]
               out[3] = 0 + 1 * in[3]

        Note:

          On :python:`domain`, :ref:`implicit bounds<implicit-bounds>` indicators have
          no effect.

        Group:
          Indexing

        Overload:
          domain
        """

    @typing.overload
    def __getitem__(self, expr: DimExpression) -> IndexTransform:
        """
        Applies a :ref:`dimension expression<python-dim-expressions>` to this transform.

        Example:

           >>> transform = ts.IndexTransform(input_rank=3)
           >>> transform[ts.d[0, 1].label['x', 'y'].translate_by[5]]
           Rank 3 -> 3 index space transform:
             Input domain:
               0: (-inf*, +inf*) "x"
               1: (-inf*, +inf*) "y"
               2: (-inf*, +inf*)
             Output index maps:
               out[0] = -5 + 1 * in[0]
               out[1] = -5 + 1 * in[1]
               out[2] = 0 + 1 * in[2]

        Group:
          Indexing

        Overload:
          expr
        """

    @typing.overload
    def __getitem__(self, indices: NumpyIndexingSpec) -> IndexTransform:
        """
        Applies a :ref:`NumPy-style indexing operation<python-numpy-style-indexing>` with default index array semantics.

        Example:

           >>> transform = ts.IndexTransform(3)
           >>> transform[2, [1, 2, 3], [6, 7, 8]]
           Rank 1 -> 3 index space transform:
             Input domain:
               0: [0, 3)
             Output index maps:
               out[0] = 2
               out[1] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                 {1, 2, 3}
               out[2] = 0 + 1 * bounded((-inf, +inf), array(in)), where array =
                 {6, 7, 8}

        See also:

           - :ref:`python-numpy-style-indexing`
           - py:obj:`IndexTransform.oindex`
           - py:obj:`IndexTransform.vindex`

        Group:
          Indexing

        Overload:
          indices
        """

    @typing.overload
    def __init__(
        self,
        input_rank: int | None = None,
        *,
        input_inclusive_min: collections.abc.Iterable[int] | None = None,
        implicit_lower_bounds: collections.abc.Iterable[bool] | None = None,
        input_exclusive_max: collections.abc.Iterable[int] | None = None,
        input_inclusive_max: collections.abc.Iterable[int] | None = None,
        input_shape: collections.abc.Iterable[int] | None = None,
        implicit_upper_bounds: collections.abc.Iterable[bool] | None = None,
        input_labels: collections.abc.Iterable[str | None] | None = None,
        output: collections.abc.Iterable[OutputIndexMap] | None = None,
    ) -> None:
        """
        Constructs an index transform from component vectors.

        Args:
          input_rank: Number of input dimensions.  Only required if the input rank is
              not otherwise specified.
          input_inclusive_min: Inclusive lower bounds for each input dimension.  If not
              specified, defaults to all zero if ``input_shape`` is specified, otherwise
              unbounded.
          implicit_lower_bounds: Indicates whether each lower bound is
              :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
              ``input_inclusive_min`` or ``input_shape`` is specified, otherwise
              defaults to all implicit.
          input_exclusive_max: Exclusive upper bounds for each input dimension.  At most
              one of ``input_exclusive_max``, ``input_inclusive_max``, and
              ``input_shape`` may be specified.
          input_inclusive_max: Inclusive upper bounds for each input dimension.
          input_shape: Size for each input dimension.
          implicit_upper_bounds: Indicates whether each upper bound is
              :ref:`implicit or explicit<implicit-bounds>`.  Defaults to all explicit if
              ``input_exclusive_max``, ``input_inclusive_max``, or ``shape`` is
              specified, otherwise defaults to all implicit.
          input_labels: :ref:`Dimension labels<dimension-labels>` for each input
              dimension.  Defaults to all unlabeled.
          output: Sequence of output index maps, or :py:obj:`OutputIndexMaps` object
              from an existing transform.  If not specified, constructs an identity
              transform over the domain.

        Examples:

            >>> # Identity transform of rank 3
            >>> ts.IndexTransform(3)
            Rank 3 -> 3 index space transform:
              Input domain:
                0: (-inf*, +inf*)
                1: (-inf*, +inf*)
                2: (-inf*, +inf*)
              Output index maps:
                out[0] = 0 + 1 * in[0]
                out[1] = 0 + 1 * in[1]
                out[2] = 0 + 1 * in[2]
            >>> ts.IndexTransform(
            ...     input_shape=[3, 2],
            ...     output=[
            ...         ts.OutputIndexMap(offset=7, input_dimension=1),
            ...         ts.OutputIndexMap([[1, 2]], offset=2, stride=-1),
            ...         ts.OutputIndexMap(8),
            ...         ts.OutputIndexMap([[1, 2]],
            ...                           offset=2,
            ...                           stride=-1,
            ...                           index_range=ts.Dim(inclusive_min=0,
            ...                                              exclusive_max=8)),
            ...     ],
            ... )
            Rank 2 -> 4 index space transform:
              Input domain:
                0: [0, 3)
                1: [0, 2)
              Output index maps:
                out[0] = 7 + 1 * in[1]
                out[1] = 2 + -1 * bounded((-inf, +inf), array(in)), where array =
                  {{1, 2}}
                out[2] = 8
                out[3] = 2 + -1 * bounded([0, 8), array(in)), where array =
                  {{1, 2}}

        Overload:
          components
        """

    @typing.overload
    def __init__(
        self,
        domain: IndexDomain | collections.abc.Sequence[Dim],
        output: collections.abc.Iterable[OutputIndexMap] | None = None,
    ) -> None:
        """
        Constructs an index transform from a domain and output index maps.

        Args:
          domain: The domain of the index transform.
          output: Sequence of output index maps, or :py:obj:`OutputIndexMaps` object
              from an existing transform.  If not specified, constructs an identity
              transform over the domain.

        Examples:

            >>> domain = ts.IndexDomain(inclusive_min=[1, 2, 3],
            ...                         exclusive_max=[4, 5, 6])
            >>> ts.IndexTransform(domain)
            Rank 3 -> 3 index space transform:
              Input domain:
                0: [1, 4)
                1: [2, 5)
                2: [3, 6)
              Output index maps:
                out[0] = 0 + 1 * in[0]
                out[1] = 0 + 1 * in[1]
                out[2] = 0 + 1 * in[2]
            >>> ts.IndexTransform(
            ...     domain,
            ...     output=[
            ...         ts.OutputIndexMap(offset=7),
            ...         ts.OutputIndexMap(input_dimension=0),
            ...     ],
            ... )
            Rank 3 -> 2 index space transform:
              Input domain:
                0: [1, 4)
                1: [2, 5)
                2: [3, 6)
              Output index maps:
                out[0] = 7
                out[1] = 0 + 1 * in[0]

        Overload:
          domain
        """

    @typing.overload
    def __init__(self, *, json: typing.Any) -> None:
        """
        Constructs an index transform from its :json:schema:`JSON representation<IndexTransform>`.

        Examples:

            >>> ts.IndexTransform(
            ...     json={
            ...         "input_inclusive_min": ["-inf", 7, ["-inf"], [8]],
            ...         "input_exclusive_max": ["+inf", 11, ["+inf"], [17]],
            ...         "input_labels": ["x", "y", "z", ""],
            ...         "output": [
            ...             {
            ...                 "offset": 3
            ...             },
            ...             {
            ...                 "stride": 2,
            ...                 "input_dimension": 2
            ...             },
            ...             {
            ...                 "offset": 7,
            ...                 "index_array": [[[[1]], [[2]], [[3]], [[4]]]],
            ...                 "index_array_bounds": [1, 4]
            ...             },
            ...         ],
            ...     })
            Rank 4 -> 3 index space transform:
              Input domain:
                0: (-inf, +inf) "x"
                1: [7, 11) "y"
                2: (-inf*, +inf*) "z"
                3: [8*, 17*)
              Output index maps:
                out[0] = 3
                out[1] = 0 + 2 * in[2]
                out[2] = 7 + 1 * bounded([1, 5), array(in)), where array =
                  {{{{1}}, {{2}}, {{3}}, {{4}}}}

        Overload:
          json
        """

    def __repr__(self) -> str:
        """
        Returns the string representation.
        """

    def to_json(self) -> typing.Any:
        """
        Returns the :json:schema:`JSON representation<IndexTransform>` of the transform.

        Example:

           >>> transform = ts.IndexTransform(
           ...     input_inclusive_min=[1, 2, -1],
           ...     implicit_lower_bounds=[1, 0, 0],
           ...     input_shape=[3, 2, 2],
           ...     implicit_upper_bounds=[0, 1, 0],
           ...     input_labels=['x', 'y', 'z'],
           ...     output=[
           ...         ts.OutputIndexMap(offset=7, stride=13, input_dimension=1),
           ...         ts.OutputIndexMap(offset=8),
           ...         ts.OutputIndexMap(
           ...             offset=1,
           ...             stride=-2,
           ...             index_array=[[[1, 2]]],
           ...             index_range=ts.Dim(inclusive_min=-3, exclusive_max=10),
           ...         ),
           ...     ],
           ... )
           >>> transform.to_json()
           {'input_exclusive_max': [4, [4], 1],
            'input_inclusive_min': [[1], 2, -1],
            'input_labels': ['x', 'y', 'z'],
            'output': [{'input_dimension': 1, 'offset': 7, 'stride': 13},
                       {'offset': 8},
                       {'index_array': [[[1, 2]]], 'offset': 1, 'stride': -2}]}

        Group:
          Accessors
        """

    def transpose(self, axes: DimSelectionLike | None = None) -> IndexTransform:
        """
        Returns a view with a transposed domain.

        This is equivalent to :python:`self[ts.d[axes].transpose[:]]`.

        Args:

          axes: Specifies the existing dimension corresponding to each dimension of the
            new view.  Dimensions may be specified either by index or label.  Specifying
            `None` is equivalent to specifying :python:`[rank-1, ..., 0]`, which
            reverses the dimension order.

        Raises:

          ValueError: If :py:param:`.axes` does not specify a valid permutation.

        See also:
          - `tensorstore.DimExpression.transpose`
          - :py:obj:`.T`

        Group:
          Indexing
        """

    @property
    def T(self) -> IndexTransform:
        """
        View with transposed domain (reversed dimension order).

        This is equivalent to: :python:`self[ts.d[::-1].transpose[:]]`.

        See also:
          - `.transpose`
          - `tensorstore.DimExpression.transpose`

        Group:
          Indexing
        """

    @property
    def domain(self) -> IndexDomain:
        """
        Input domain of the index transform.

        Example:

            >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
            ...                               input_labels=["x", "y", "z"])
            >>> transform.domain
            { "x": [0, 3), "y": [0, 4), "z": [0, 5) }

        Group:
          Accessors
        """

    @property
    def implicit_lower_bounds(self) -> tuple[bool, ...]:
        """
        Indicates whether the lower bound of each input dimension is :ref:`implicit or explicit<implicit-bounds>`.

        Alias for the :py:obj:`~tensorstore.IndexDomain.implicit_lower_bounds` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_rank=3)
            >>> transform.implicit_lower_bounds
            (True, True, True)
            >>> transform = ts.IndexTransform(input_inclusive_min=[2, 3, 4])
            >>> transform.implicit_lower_bounds
            (False, False, False)
            >>> transform = ts.IndexTransform(input_exclusive_max=[2, 3, 4])
            >>> transform.implicit_lower_bounds
            (True, True, True)
            >>> transform = ts.IndexTransform(input_shape=[4, 5, 6])
            >>> transform.implicit_lower_bounds
            (False, False, False)
            >>> transform = ts.IndexTransform(
            ...     input_inclusive_min=[4, 5, 6],
            ...     implicit_lower_bounds=[False, True, False])
            >>> transform.implicit_lower_bounds
            (False, True, False)

        Group:
          Accessors
        """

    @property
    def implicit_upper_bounds(self) -> tuple[bool, ...]:
        """
        Indicates whether the upper bound of each input dimension is :ref:`implicit or explicit<implicit-bounds>`.

        Alias for the :py:obj:`~tensorstore.IndexDomain.implicit_upper_bounds` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_rank=3)
            >>> transform.implicit_upper_bounds
            (True, True, True)
            >>> transform = ts.IndexTransform(input_shape=[2, 3, 4])
            >>> transform.implicit_upper_bounds
            (False, False, False)
            >>> transform = ts.IndexTransform(input_inclusive_min=[4, 5, 6])
            >>> transform.implicit_upper_bounds
            (True, True, True)
            >>> transform = ts.IndexTransform(
            ...     input_exclusive_max=[4, 5, 6],
            ...     implicit_upper_bounds=[False, True, False])
            >>> transform.implicit_upper_bounds
            (False, True, False)

        Group:
          Accessors
        """

    @property
    def input_exclusive_max(self) -> tuple[int, ...]:
        """
        Exclusive upper bound of the input domain.

        Alias for the :py:obj:`~tensorstore.IndexDomain.exclusive_max` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
            ...                               input_shape=[3, 4, 5])
            >>> transform.input_exclusive_max
            (4, 6, 8)

        Group:
          Accessors
        """

    @property
    def input_inclusive_max(self) -> tuple[int, ...]:
        """
        Inclusive upper bound of the input domain.

        Alias for the :py:obj:`~tensorstore.IndexDomain.inclusive_max` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
            ...                               input_shape=[3, 4, 5])
            >>> transform.input_inclusive_max
            (3, 5, 7)

        Group:
          Accessors
        """

    @property
    def input_inclusive_min(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the input domain, alias for :py:obj:`.input_origin`.

        Alias for the :py:obj:`~tensorstore.IndexDomain.inclusive_min` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
            ...                               input_shape=[3, 4, 5])
            >>> transform.input_inclusive_min
            (1, 2, 3)

        Group:
          Accessors
        """

    @property
    def input_labels(self) -> tuple[str, ...]:
        """
        :ref:`Dimension labels<dimension-labels>` for each input dimension.

        Alias for the :py:obj:`~tensorstore.IndexDomain.labels` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
            ...                               input_shape=[3, 4, 5],
            ...                               input_labels=['x', 'y', 'z'])
            >>> transform.input_labels
            ('x', 'y', 'z')

        Group:
          Accessors
        """

    @property
    def input_origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the input domain.

        Alias for the :py:obj:`~tensorstore.IndexDomain.origin` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_inclusive_min=[1, 2, 3],
            ...                               input_shape=[3, 4, 5])
            >>> transform.input_origin
            (1, 2, 3)

        Group:
          Accessors
        """

    @property
    def input_rank(self) -> int:
        """
        Rank of the input space.

        Example:

            >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
            ...                               input_labels=["x", "y", "z"])
            >>> transform.input_rank
            3

        Group:
          Accessors
        """

    @property
    def input_shape(self) -> tuple[int, ...]:
        """
        Shape of the input domain.

        Alias for the :py:obj:`~tensorstore.IndexDomain.shape` property of the :py:obj:`.domain`.

        Example:

            >>> transform = ts.IndexTransform(input_shape=[3, 4, 5])
            >>> transform.input_shape
            (3, 4, 5)

        Group:
          Accessors
        """

    @property
    def label(self) -> IndexTransform._Label: ...

    @property
    def mark_bounds_implicit(self) -> IndexTransform._MarkBoundsImplicit: ...

    @property
    def ndim(self) -> int:
        """
        Rank of the input space, alias for :py:obj:`.input_rank`.

        Example:

            >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
            ...                               input_labels=["x", "y", "z"])
            >>> transform.ndim
            3

        Group:
          Accessors
        """

    @property
    def oindex(self) -> IndexTransform._Oindex: ...

    @property
    def origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain.

        This is equivalent to :python:`self.domain.origin`.

        Group:
          Accessors
        """

    @property
    def output(self) -> OutputIndexMaps:
        """
        Output index maps.

        Group:
          Accessors
        """

    @property
    def output_rank(self) -> int:
        """
        Rank of the output space.

        Example:

            >>> transform = ts.IndexTransform(input_shape=[3, 4, 5],
            ...                               input_labels=["x", "y", "z"],
            ...                               output=[ts.OutputIndexMap(offset=5)])
            >>> transform.output_rank
            1

        Group:
          Accessors
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the domain.

        This is equivalent to :python:`self.domain.shape`.

        Group:
          Accessors
        """

    @property
    def size(self) -> int:
        """
        Total number of elements in the domain.

        This is equivalent to :python:`self.domain.size`.

        Group:
          Accessors
        """

    @property
    def translate_backward_by(self) -> IndexTransform._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> IndexTransform._TranslateBy: ...

    @property
    def translate_to(self) -> IndexTransform._TranslateTo: ...

    @property
    def vindex(self) -> IndexTransform._Vindex: ...


class KvStore:
    """

    Key-value store that maps an ordered set of byte string keys to byte string values.

    This is used as the storage interface for most of the
    :ref:`TensorStore drivers<tensorstore-drivers>`.

    The actual storage mechanism is determined by the
    :ref:`driver<key-value-store-drivers>`.

    Example:

        >>> store = await ts.KvStore.open({'driver': 'memory'})
        >>> await store.write(b'a', b'value')
        KvStore.TimestampedStorageGeneration(...)
        >>> await store.read(b'a')
        KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
        >>> await store.read(b'b')
        KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))
        >>> await store.list()
        [b'a']

    By default, operations are non-transactional, but transactional operations are
    also supported:

        >>> txn = ts.Transaction()
        >>> store.with_transaction(txn)[b'a']
        b'value'
        >>> store.with_transaction(txn)[b'a'] = b'new value'
        >>> store.with_transaction(txn)[b'a']
        b'new value'
        >>> store[b'a']
        b'value'
        >>> txn.commit_sync()
        >>> store[b'a']
        b'new value'

    Group:
      Core

    Classes
    -------

    Constructors
    ------------

    Accessors
    ---------

    I/O
    ---

    Synchronous I/O
    ---------------

    """

    class KeyRange:
        """

        Half-open interval of byte string keys, according to lexicographical order.
        """

        __hash__: typing.ClassVar[None] = None

        def __copy__(self) -> KvStore.KeyRange: ...

        def __deepcopy__(self, memo: dict) -> KvStore.KeyRange: ...

        def __eq__(self, other: KvStore.KeyRange) -> bool:
            """
            Compares with another range for equality.
            """

        def __init__(
            self, inclusive_min: str | bytes = "", exclusive_max: str | bytes = ""
        ) -> None:
            """
            Constructs a key range from the specified half-open bounds.

            Args:

              inclusive_min: Inclusive lower bound of the range.  In accordance with the
                usual lexicographical order, an empty string indicates no lower bound.

              exclusive_max: Exclusive upper bound of the range.  As a special case, an
                empty string indicates no upper bound.
            """

        def __repr__(self) -> str: ...

        def copy(self) -> KvStore.KeyRange:
            """
            Returns a copy of the range.

            Group:
              Accessors
            """

        @property
        def empty(self) -> bool:
            """
            Indicates if the range contains no keys.

            Example:

                >>> r = ts.KvStore.KeyRange(b'x', b'y')
                >>> r.empty
                False
                >>> r = ts.KvStore.KeyRange(b'x', b'x')
                >>> r.empty
                True
                >>> r = ts.KvStore.KeyRange(b'y', b'x')
                >>> r.empty
                True

            Group:
              Accessors
            """

        @property
        def exclusive_max(self) -> bytes:
            """
            Exclusive upper bound of the range.

            As a special case, an empty string indicates no upper bound.

            Group:
              Accessors
            """

        @exclusive_max.setter
        def exclusive_max(self, arg1: str | bytes) -> None: ...

        @property
        def inclusive_min(self) -> bytes:
            """
            Inclusive lower bound of the range.

            In accordance with the usual lexicographical order, an empty string indicates no
            lower bound.

            Group:
              Accessors
            """

        @inclusive_min.setter
        def inclusive_min(self, arg1: str | bytes) -> None: ...

    class ReadResult:
        """

        Specifies the result of a read operation.
        """

        def __copy__(self) -> KvStore.ReadResult: ...

        def __deepcopy__(self, memo: dict) -> KvStore.ReadResult: ...

        def __init__(
            self,
            state: typing.Literal["unspecified", "missing", "value"] = "unspecified",
            value: str | bytes = "",
            stamp: KvStore.TimestampedStorageGeneration = ...,
        ) -> None:
            """
            Constructs a read result.
            """

        def __repr__(self) -> str: ...

        @property
        def stamp(self) -> KvStore.TimestampedStorageGeneration:
            """
            Generation and timestamp associated with the value.
            """

        @stamp.setter
        def stamp(self, arg1: KvStore.TimestampedStorageGeneration) -> None: ...

        @property
        def state(self) -> typing.Literal["unspecified", "missing", "value"]:
            """
            Indicates the interpretation of :py:obj:`.value`.
            """

        @state.setter
        def state(
            self, arg1: typing.Literal["unspecified", "missing", "value"]
        ) -> None: ...

        @property
        def value(self) -> bytes:
            """
            Value associated with the key.
            """

        @value.setter
        def value(self, arg1: str | bytes) -> None: ...

    class Spec:
        """

        Parsed representation of a :json:schema:`JSON key-value store<KvStore>` specification.
        """

        __hash__: typing.ClassVar[None] = None

        def __add__(self, suffix: str | bytes) -> KvStore.Spec:
            """
            Returns a key-value store with the suffix appended to the path.

            The suffix is appended directly to :py:obj:`.path` without any separator.  To
            ensure there is a :python:`'/'` separator, use :py:obj:`.__truediv__` instead.

            Example:

                >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
                >>> spec + '/abc'
                KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})
                >>> spec + 'abc'
                KvStore.Spec({'driver': 'file', 'path': 'tmp/dataabc'})

            Group:
              Operators
            """

        def __copy__(self) -> KvStore.Spec: ...

        def __deepcopy__(self, memo: dict) -> KvStore.Spec: ...

        def __eq__(self, other: KvStore.Spec) -> bool:
            """
            Compares with another :py:obj:`KvStore.Spec` for equality based on the :json:schema:`JSON representation<KvStore>`.

            The comparison is based on the JSON representation, except that any bound
            context resources are compared by identity (not by their JSON representation).

            Example:

              >>> spec = ts.KvStore.Spec({'driver': 'memory'})
              >>> assert spec == spec
              >>> a, b = spec.copy(), spec.copy()
              >>> context_a, context_b = ts.Context(), ts.Context()
              >>> a.update(context=context_a)
              >>> b.update(context=context_b)
              >>> assert a == a
              >>> assert a != b
            """

        def __new__(self, json: typing.Any) -> KvStore.Spec:
            """
            Constructs from the :json:schema:`JSON representation<KvStore>` or a :json:schema:`URL<KvStoreUrl>`.

            Example of constructing from the :json:schema:`JSON representation<KvStore>`:

                >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
                >>> spec
                KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

            Example of constructing from a :json:schema:`URL<KvStoreUrl>`:

                >>> spec = ts.KvStore.Spec('file:///path/to/data/')
                >>> spec
                KvStore.Spec({'driver': 'file', 'path': '/path/to/data/'})
            """

        def __repr__(self) -> str:
            """
            Returns a string representation based on the  :json:schema:`JSON representation<KvStore>`.

            Example:

                >>> ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
                KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
            """

        def __truediv__(self, component: str | bytes) -> KvStore.Spec:
            """
            Returns a key-value store with an additional path component joined to the path.

            Example:

                >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
                >>> spec / 'abc'
                KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})
                >>> spec / '/abc'
                KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc'})

            Group:
              Operators
            """

        def copy(self) -> KvStore.Spec:
            """
            Returns a copy of the key-value store spec.

            Example:

              >>> a = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})
              >>> b = a.copy()
              >>> a.path = 'tmp/data/abc/'
              >>> a
              KvStore.Spec({'driver': 'file', 'path': 'tmp/data/abc/'})
              >>> b
              KvStore.Spec({'driver': 'file', 'path': 'tmp/data/'})

            Group:
              Accessors
            """

        def to_json(self, include_defaults: bool = False) -> typing.Any:
            """
            Converts to the :json:schema:`JSON representation<KvStore>`.

            Example:

              >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/dataset/'})
              >>> spec /= 'abc/'
              >>> spec.to_json()
              {'driver': 'file', 'path': 'tmp/dataset/abc/'}
              >>> spec.to_json(include_defaults=True)
              {'context': {},
               'driver': 'file',
               'file_io_concurrency': 'file_io_concurrency',
               'file_io_locking': 'file_io_locking',
               'file_io_mode': 'file_io_mode',
               'file_io_sync': 'file_io_sync',
               'path': 'tmp/dataset/abc/'}

            Group:
              Accessors
            """

        def update(
            self,
            *,
            unbind_context: bool | None = None,
            strip_context: bool | None = None,
            context: Context | None = None,
        ) -> None:
            """
            Modifies a spec.

            Example:

                >>> spec = ts.KvStore.Spec({
                ...     'driver': 'memory',
                ...     'path': 'abc/',
                ...     'memory_key_value_store': 'memory_key_value_store#a'
                ... })
                >>> spec.update(context=ts.Context({'memory_key_value_store#a': {}}))
                >>> spec
                KvStore.Spec({
                  'context': {'memory_key_value_store#a': {}},
                  'driver': 'memory',
                  'memory_key_value_store': ['memory_key_value_store#a'],
                  'path': 'abc/',
                })
                >>> spec.update(unbind_context=True)
                >>> spec
                KvStore.Spec({
                  'context': {'memory_key_value_store#a': {}},
                  'driver': 'memory',
                  'memory_key_value_store': 'memory_key_value_store#a',
                  'path': 'abc/',
                })
                >>> spec.update(strip_context=True)
                >>> spec
                KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

            Args:

              unbind_context: Convert any bound context resources to context resource specs that fully capture
                the graph of shared context resources and interdependencies.

                Re-binding/re-opening the resultant spec will result in a new graph of new
                context resources that is isomorphic to the original graph of context resources.
                The resultant spec will not refer to any external context resources;
                consequently, binding it to any specific context will have the same effect as
                binding it to a default context.

                Specifying a value of :python:`False` has no effect.
              strip_context: Replace any bound context resources and unbound context resource specs by
                default context resource specs.

                If the resultant :py:obj:`~tensorstore.KvStore.Spec` is re-opened with, or
                re-bound to, a new context, it will use the default context resources specified
                by that context.

                Specifying a value of :python:`False` has no effect.
              context: Bind any context resource specs using the specified shared resource context.

                Any already-bound context resources remain unchanged.  Additionally, any context
                resources specified by a nested :json:schema:`KvStore.context` spec will be
                created as specified, but won't be overridden by :py:param:`.context`.


            Group:
              Mutators
            """

        @property
        def base(self) -> KvStore.Spec | None:
            """
            Underlying key-value store, if this is a key-value store adapter.

            Adapter key-value stores include:

            - :ref:`OCDBT<kvstore/ocdbt>`
            - :ref:`neuroglancer_uint64_sharded<kvstore/neuroglancer_uint64_sharded>`

            For regular, non-adapter key-value stores, this is :python:`None`.

            Example:

                >>> spec = ts.KvStore.Spec({'driver': 'ocdbt', 'base': 'memory://'})
                >>> spec.base
                KvStore.Spec({'driver': 'memory'})

            See also:

              - :py:obj:`KvStore.base`

            Group:
              Accessors
            """

        @property
        def path(self) -> str:
            """
            Path prefix within the base key-value store.

            Example:

                >>> spec = ts.KvStore.Spec({'driver': 'file', 'path': 'tmp/data'})
                >>> spec.path
                'tmp/data'

            Group:
              Accessors
            """

        @path.setter
        def path(self, arg1: str | bytes) -> None: ...

        @property
        def url(self) -> str:
            """
            :json:schema:`URL representation<KvStoreUrl>` of the key-value store specification.

            Example:

                >>> spec = ts.KvStore.Spec({
                ...     'driver': 'gcs',
                ...     'bucket': 'my-bucket',
                ...     'path': 'path/to/object'
                ... })
                >>> spec.url
                'gs://my-bucket/path/to/object'

            Group:
              Accessors
            """

    class TimestampedStorageGeneration:
        """

        Specifies a storage generation identifier and a timestamp.
        """

        __hash__: typing.ClassVar[None] = None

        def __copy__(self) -> KvStore.TimestampedStorageGeneration: ...

        def __deepcopy__(self, memo: dict) -> KvStore.TimestampedStorageGeneration: ...

        def __eq__(self, other: KvStore.TimestampedStorageGeneration) -> bool:
            """
            Compares two timestamped storage generations for equality.
            """

        def __init__(self, generation: str | bytes = "", time: float = ...) -> None:
            """
            Constructs from a storage generation and time.
            """

        def __repr__(self) -> str: ...

        @property
        def generation(self) -> bytes:
            """
            Identifies a specific version of a key-value store entry.

            An empty string :python:`b''` indicates an unspecified version.

            Group:
              Accessors
            """

        @generation.setter
        def generation(self, arg1: str | bytes) -> None: ...

        @property
        def time(self) -> float:
            """
            Time (seconds since Unix epoch) at which :py:obj:`.generation` is valid.

            Group:
              Accessors
            """

        @time.setter
        def time(self, arg1: float) -> None: ...

    __iter__ = None

    @staticmethod
    def open(
        spec: KvStore.Spec | typing.Any,
        *,
        context: Context | None = None,
        transaction: Transaction | None = None,
    ) -> Future[KvStore]:
        """
        Opens a key-value store.

        Example of opening from a :json:schema:`JSON KvStore spec<KvStore>`:

            >>> kvstore = await ts.KvStore.open({'driver': 'memory', 'path': 'abc/'})
            >>> await kvstore.write(b'x', b'y')
            KvStore.TimestampedStorageGeneration(b'...', ...)
            >>> await kvstore.read(b'x')
            KvStore.ReadResult(state='value', value=b'y', stamp=KvStore.TimestampedStorageGeneration(b'...', ...))

        Example of opening from a :json:schema:`URL<KvStoreUrl>`:

            >>> kvstore = await ts.KvStore.open('memory://abc/')
            >>> kvstore.spec()
            KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

        Example of opening from an existing :py:obj:`KvStore.Spec`:

            >>> spec = ts.KvStore.Spec({'driver': 'memory', 'path': 'abc/'})
            >>> kvstore = await ts.KvStore.open(spec)
            >>> kvstore.spec()
            KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

        Args:

          spec: Key-value store spec to open.  May also be specified as
            :json:schema:`JSON<KvStore>` or a :json:schema:`URL<KvStoreUrl>`.

          context: Bind any context resource specs using the specified shared resource context.

            Any already-bound context resources remain unchanged.  Additionally, any context
            resources specified by a nested :json:schema:`KvStore.context` spec will be
            created as specified, but won't be overridden by :py:param:`.context`.
          transaction: Transaction to use for read/write operations.  By default, operations are
            non-transactional.

            .. note::

               To perform transactional operations using a :py:obj:`KvStore` that was
               previously opened without a transaction, use
               :py:obj:`KvStore.with_transaction`.


        Group:
          Constructors
        """

    def __add__(self, suffix: str | bytes) -> KvStore:
        """
        Returns a key-value store with the suffix appended to the path.

        The suffix is appended directly to :py:obj:`.path` without any separator.  To
        ensure there is a :python:`'/'` separator, use :py:obj:`.__truediv__` instead.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'file', 'path': 'tmp/data'})
            >>> store + '/abc'
            KvStore({
              'context': {
                'file_io_concurrency': {},
                'file_io_locking': {},
                'file_io_mode': {},
                'file_io_sync': True,
              },
              'driver': 'file',
              'path': 'tmp/data/abc',
            })
            >>> store + 'abc'
            KvStore({
              'context': {
                'file_io_concurrency': {},
                'file_io_locking': {},
                'file_io_mode': {},
                'file_io_sync': True,
              },
              'driver': 'file',
              'path': 'tmp/dataabc',
            })

        Group:
          Operators
        """

    def __contains__(self, key: str | bytes) -> bool:
        """
        Synchronously checks if the given key is present.

        Example:

            >>> store = ts.KvStore.open({'driver': 'memory'}).result()
            >>> store[b'a'] = b'value'
            >>> b'a' in store
            True
            >>> b'b' in store
            False

        Args:

          key: The key to check.  This is appended (without any separator) to the
            existing :py:obj:`.path`, if any.

        Returns:

          `True` if the key is present.

        Raises:

          Exception: If an I/O error occurs.

        Note:

          The current thread is blocked until the read completes, but computations in
          other threads may continue.

        See also:

          - :py:obj:`.read`
          - :py:obj:`.__getitem__`

        Group:
          Synchronous I/O
        """

    def __copy__(self) -> KvStore: ...

    def __delitem__(self, key: str | bytes) -> None:
        """
        Synchronously deletes a single key.

        Example:

            >>> store = ts.KvStore.open({'driver': 'memory'}).result()
            >>> store[b'a'] = b'value'
            >>> store[b'a']
            b'value'
            >>> del store[b'a']
            >>> store[b'a']
            Traceback (most recent call last):
                ...
            KeyError...

        Args:

          key: Key to delete.  This is appended (without any separator) to the existing
            :py:obj:`.path`, if any.

        Raises:

          Exception: If an I/O error occurs.

        Note:

          - If no :py:obj:`.transaction` is specified, the current thread is blocked
            until the delete completes and durability is guaranteed (to the extent
            supported by the :ref:`driver<key-value-store-drivers>`).

          - If a :py:obj:`.transaction` is specified, the current thread is blocked
            until the delete is recorded in the transaction.  The actual delete
            operation is not performed until the transaction is committed.

          Computations in other threads may continue even while the current thread is
          blocked.

        See also:

          - :py:obj:`.write`
          - :py:obj:`.__getitem__`
          - :py:obj:`.__setitem__`

        Group:
          Synchronous I/O
        """

    def __getitem__(self, key: str | bytes) -> bytes:
        """
        Synchronously reads the value of a single key.

        Example:

            >>> store = ts.KvStore.open({'driver': 'memory'}).result()
            >>> store[b'a'] = b'value'
            >>> store[b'a']
            b'value'
            >>> store[b'b']
            Traceback (most recent call last):
                ...
            KeyError...

        Args:

          key: The key to read.  This is appended (without any separator) to the
            existing :py:obj:`.path`, if any.

        Returns:

          The value associated with :py:param:`key` on success.

        Raises:

          KeyError: If :py:param:`key` is not found.
          Exception: If an I/O error occurs.

        Note:

          The current thread is blocked until the read completes, but computations in
          other threads may continue.

        See also:

          - :py:obj:`.read`
          - :py:obj:`.__setitem__`
          - :py:obj:`.__delitem__`

        Group:
          Synchronous I/O
        """

    def __repr__(self) -> str:
        """
        Returns a string representation based on the  :json:schema:`JSON representation<KvStore>`.

        Example:

            >>> kvstore = await ts.KvStore.open({
            ...     'driver': 'file',
            ...     'path': 'tmp/data/'
            ... })
            >>> kvstore
            KvStore({
              'context': {
                'file_io_concurrency': {},
                'file_io_locking': {},
                'file_io_mode': {},
                'file_io_sync': True,
              },
              'driver': 'file',
              'path': 'tmp/data/',
            })
        """

    def __setitem__(self, key: str | bytes, value: str | bytes | None) -> None:
        """
        Synchronously writes the value of a single key.

        Example:

            >>> store = ts.KvStore.open({'driver': 'memory'}).result()
            >>> store[b'a'] = b'value'
            >>> store[b'a']
            b'value'
            >>> store[b'b']
            Traceback (most recent call last):
                ...
            KeyError...
            >>> store[b'a'] = None
            >>> store[b'a']
            Traceback (most recent call last):
                ...
            KeyError...

        Args:


          key: Key to write/delete.  This is appended (without any separator) to the
            existing :py:obj:`.path`, if any.

          value: Value to store, or :py:obj:`None` to delete.

        Raises:

          Exception: If an I/O error occurs.

        Note:

          - If no :py:obj:`.transaction` is specified, the current thread is blocked
            until the write completes and durability is guaranteed (to the extent
            supported by the :ref:`driver<key-value-store-drivers>`).

          - If a :py:obj:`.transaction` is specified, the current thread is blocked
            until the write is recorded in the transaction.  The actual write operation
            is not performed until the transaction is committed.

          Computations in other threads may continue even while the current thread is
          blocked.

        See also:

          - :py:obj:`.write`
          - :py:obj:`.__getitem__`
          - :py:obj:`.__delitem__`

        Group:
          Synchronous I/O
        """

    def __truediv__(self, component: str | bytes) -> KvStore:
        """
        Returns a key-value store with an additional path component joined to the path.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'file', 'path': 'tmp/data'})
            >>> store / 'abc'
            KvStore({
              'context': {
                'file_io_concurrency': {},
                'file_io_locking': {},
                'file_io_mode': {},
                'file_io_sync': True,
              },
              'driver': 'file',
              'path': 'tmp/data/abc',
            })
            >>> store / '/abc'
            KvStore({
              'context': {
                'file_io_concurrency': {},
                'file_io_locking': {},
                'file_io_mode': {},
                'file_io_sync': True,
              },
              'driver': 'file',
              'path': 'tmp/data/abc',
            })

        Group:
          Operators
        """

    def copy(self) -> KvStore:
        """
        Returns a copy of the key-value store.

        Example:

          >>> a = await ts.KvStore.open({'driver': 'file', 'path': 'tmp/data/'})
          >>> b = a.copy()
          >>> a.path = 'tmp/data/abc/'
          >>> a
          KvStore({
            'context': {
              'file_io_concurrency': {},
              'file_io_locking': {},
              'file_io_mode': {},
              'file_io_sync': True,
            },
            'driver': 'file',
            'path': 'tmp/data/abc/',
          })
          >>> b
          KvStore({
            'context': {
              'file_io_concurrency': {},
              'file_io_locking': {},
              'file_io_mode': {},
              'file_io_sync': True,
            },
            'driver': 'file',
            'path': 'tmp/data/',
          })

        Group:
          Accessors
        """

    def delete_range(self, range: KvStore.KeyRange) -> Future[None]:
        """
        Deletes a key range.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'memory'})
            >>> await store.write(b'a', b'value')
            >>> await store.write(b'b', b'value')
            >>> await store.write(b'c', b'value')
            >>> await store.list()
            [b'a', b'b', b'c']
            >>> await store.delete_range(ts.KvStore.KeyRange(b'aa', b'cc'))
            >>> await store.list()
            [b'a']

        Args:

          range: Key range to delete.  This is relative to the existing :py:obj:`.path`,
            if any.

        Returns:

          - If no :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
            becomes ready when the delete operation has completed and durability is
            guaranteed (to the extent supported by the
            :ref:`driver<key-value-store-drivers>`).

          - If a :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
            becomes ready when the delete operation is recorded in the transaction.  The
            delete operation is not actually performed until the transaction is
            committed.

        Group:
          I/O
        """

    def experimental_copy_range_to(
        self,
        target: KvStore,
        source_range: KvStore.KeyRange | None = None,
        source_staleness_bound: float | None = None,
    ) -> Future[None]:
        """
        Copies a range of keys.

        .. warning::

           This API is experimental and subject to change.

        Example:

            >>> store = await ts.KvStore.open({
            ...     'driver': 'ocdbt',
            ...     'base': 'memory://'
            ... })
            >>> await store.write(b'x/a', b'value')
            >>> await store.write(b'x/b', b'value')
            >>> await store.list()
            [b'x/a', b'x/b']
            >>> await (store / "x/").experimental_copy_range_to(store / "y/")
            >>> await store.list()
            [b'x/a', b'x/b', b'y/a', b'y/b']

        .. note::

           Depending on the kvstore implementation, this operation may be able to
           perform the copy without actually re-writing the data.

        Args:

          target: Target key-value store.

            .. warning::

               This may refer to the same kvstore as ``self``, but the target key range
               must not overlap with ``self``.  If this requirement is violated, the
               behavior is unspecified.

          source_range: Key range to include.  This is relative to the existing
            :py:obj:`.path`, if any.  If not specified, all keys under :py:obj:`.path`
            are copied.
          source_staleness_bound: Specifies a time in (fractional) seconds since the
            Unix epoch.  If specified, data that is cached internally by the kvstore
            implementation may be used without validation if not older than the
            :py:param:`.source_staleness_bound`.  Cached data older than
            :py:param:`.source_staleness_bound` must be validated before being returned.
            A value of :python:`float('inf')` indicates that the result must be current
            as of the time the :py:obj:`.read` request was made, i.e. it is equivalent
            to specifying a value of :python:`time.time()`.  A value of
            :python:`float('-inf')` indicates that cached data may be returned without
            validation irrespective of its age.

        Returns:

          - If no :py:obj:`.transaction` is specified for :py:param:`.target`, returns a
            :py:obj:`Future` that becomes ready when the copy operation has completed
            and durability is guaranteed (to the extent supported by the
            :ref:`driver<key-value-store-drivers>`).

          - If a :py:obj:`.transaction` is specified for :py:param:`.target`, returns a
            :py:obj:`Future` that becomes ready when the copy operation is recorded in
            the transaction.  The copy operation is not actually performed until the
            transaction is committed.

        Group:
          I/O
        """

    def list(
        self, range: KvStore.KeyRange | None = None, strip_prefix_length: int = 0
    ) -> Future[list[bytes]]:
        """
        Lists the keys in the key-value store.

        Example:

            >>> store = ts.KvStore.open({'driver': 'memory'}).result()
            >>> store[b'a'] = b'value'
            >>> store[b'b'] = b'value'
            >>> store.list().result()
            [b'a', b'b']
            >>> store.list(ts.KvStore.KeyRange(inclusive_min=b'b')).result()
            [b'b']

        Args:

          range: If specified, restricts to the specified key range.

          strip_prefix_length: Strips the specified number of bytes from the start of
            the returned keys.

        Returns:

          Future that resolves to the list of matching keys, in an unspecified order.

        Raises:

          ValueError: If a :py:obj:`.transaction` is specified.

        Warning:

          This returns all keys within :py:param:`range` as a single :py:obj:`list`.  If
          there are a large number of matching keys, this can consume a large amount of
          memory.

        Group:
          I/O
        """

    def read(
        self,
        key: str | bytes,
        *,
        if_not_equal: str | bytes | None = None,
        staleness_bound: float | None = None,
        batch: Batch | None = None,
        byte_range: slice | None = None,
    ) -> Future[KvStore.ReadResult]:
        """
        Reads the value of a single key.

        A missing key is not treated as an error; instead, a :py:obj:`.ReadResult` with
        :py:obj:`.ReadResult.state` set to :python:`'missing'` is returned.

        Note:

          The behavior in the case of a missing key differs from that of
          :py:obj:`.__getitem__`, which raises :py:obj:`KeyError` to indicate a missing
          key.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'memory'})
            >>> await store.write(b'a', b'value')
            KvStore.TimestampedStorageGeneration(...)
            >>> await store.read(b'a')
            KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
            >>> store[b'a']
            b'value'
            >>> await store.read(b'b')
            KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))
            >>> store[b'b']
            Traceback (most recent call last):
                ...
            KeyError...

            >>> store[b'a'] = b'value'
            >>> store[b'b'] = b'value'
            >>> store.list().result()

        If a :py:obj:`.transaction` is bound, the read reflects any writes made within
        the transaction, and the commit of the transaction will fail if the value
        associated with :py:param:`key` changes after the read due to external writes,
        i.e. consistent reads are guaranteed.

        Args:

          key: The key to read.  This is appended (without any separator) to the
            existing :py:obj:`.path`, if any.

          if_not_equal: If specified, the read is aborted if the generation associated
            with :py:param:`key` matches :py:param:`if_not_equal`.  An aborted read due
            to this condition is indicated by a :py:obj:`.ReadResult.state` of
            :python:`'unspecified'`.  This may be useful for validating a cached value
            cache validation at a higher level.

          staleness_bound: Specifies a time in (fractional) seconds since the Unix
            epoch.  If specified, data that is cached internally by the kvstore
            implementation may be used without validation if not older than the
            :py:param:`staleness_bound`.  Cached data older than
            :py:param:`staleness_bound` must be validated before being returned.  A
            value of :python:`float('inf')` indicates that the result must be current as
            of the time the :py:obj:`.read` request was made, i.e. it is equivalent to
            specifying a value of :python:`time.time()`.  A value of
            :python:`float('-inf')` indicates that cached data may be returned without
            validation irrespective of its age.

          batch: Batch to use for the read operation.

            .. warning::

               If specified, the returned :py:obj:`Future` will not, in general, become
               ready until the batch is submitted.  Therefore, immediately awaiting the
               returned future will lead to deadlock.

          byte_range: Byte range to request, specified as a `slice` of the form
            ``slice(50, 100)`` or ``slice(None, 100)`` or ``slice(50, None)`` or
            ``slice(-50, None)``.

        Returns:
          Future that resolves when the read operation completes.

        See also:

          - :py:obj:`.write`
          - :py:obj:`.__getitem__`
          - :py:obj:`.__setitem__`
          - :py:obj:`.__delitem__`

        Group:
          I/O
        """

    def spec(
        self, *, retain_context: bool | None = None, unbind_context: bool | None = None
    ) -> KvStore.Spec:
        """
        Spec that may be used to re-open or re-create the key-value store.

        Example:

            >>> kvstore = await ts.KvStore.open({'driver': 'memory', 'path': 'abc/'})
            >>> kvstore.spec()
            KvStore.Spec({'driver': 'memory', 'path': 'abc/'})
            >>> kvstore.spec(unbind_context=True)
            KvStore.Spec({'context': {'memory_key_value_store': {}}, 'driver': 'memory', 'path': 'abc/'})
            >>> kvstore.spec(retain_context=True)
            KvStore.Spec({
              'context': {'memory_key_value_store': {}},
              'driver': 'memory',
              'memory_key_value_store': ['memory_key_value_store'],
              'path': 'abc/',
            })

        Args:

          retain_context: Retain all bound context resources (e.g. specific concurrency pools, specific
            cache pools).

            The resultant :py:obj:`~tensorstore.KvStore.Spec` may be used to re-open the
            :py:obj:`~tensorstore.KvStore` using the identical context resources.

            Specifying a value of :python:`False` has no effect.
          unbind_context: Convert any bound context resources to context resource specs that fully capture
            the graph of shared context resources and interdependencies.

            Re-binding/re-opening the resultant spec will result in a new graph of new
            context resources that is isomorphic to the original graph of context resources.
            The resultant spec will not refer to any external context resources;
            consequently, binding it to any specific context will have the same effect as
            binding it to a default context.

            Specifying a value of :python:`False` has no effect.


        Group:
          Accessors
        """

    def with_transaction(self, transaction: Transaction | None) -> KvStore:
        """
        Returns a transaction-bound view of this key-value store.

        The returned view may be used to perform transactional read/write operations.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'memory'})
            >>> txn = ts.Transaction()
            >>> await store.with_transaction(txn).write(b'a', b'value')
            >>> (await store.with_transaction(txn).read(b'a')).value
            b'value'
            >>> await txn.commit_async()

        Group:
          Transactions
        """

    def write(
        self,
        key: str | bytes,
        value: str | bytes | None,
        *,
        if_equal: str | bytes | None = None,
    ) -> Future[KvStore.TimestampedStorageGeneration]:
        """
        Writes or deletes a single key.

        Example:

            >>> store = await ts.KvStore.open({'driver': 'memory'})
            >>> await store.write(b'a', b'value')
            KvStore.TimestampedStorageGeneration(...)
            >>> await store.read(b'a')
            KvStore.ReadResult(state='value', value=b'value', stamp=KvStore.TimestampedStorageGeneration(...))
            >>> await store.write(b'a', None)
            KvStore.TimestampedStorageGeneration(...)
            >>> await store.read(b'a')
            KvStore.ReadResult(state='missing', value=b'', stamp=KvStore.TimestampedStorageGeneration(...))

        Args:

          key: Key to write/delete.  This is appended (without any separator) to the
            existing :py:obj:`.path`, if any.

          value: Value to store, or :py:obj:`None` to delete.

          if_equal: If specified, indicates a conditional write operation.  The write is
            performed only if the existing generation associated with :py:param:`key`
            matches :py:param:`if_equal`.

        Returns:

          - If no :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
            resolves to the new storage generation for :py:param:`key` once the write
            operation completes and durability is guaranteed (to the extent supported by
            the :ref:`driver<key-value-store-drivers>`).

          - If a :py:obj:`.transaction` is specified, returns a :py:obj:`Future` that
            resolves to an empty storage generation once the write operation is recorded
            in the transaction.  The write operation is not actually performed until the
            transaction is committed.

        See also:

          - :py:obj:`.__setitem__`
          - :py:obj:`.__delitem__`

        Group:
          I/O
        """

    @property
    def base(self) -> KvStore | None:
        """
        Underlying key-value store, if this is a key-value store adapter.

        Adapter key-value stores include:

        - :ref:`kvstore/ocdbt`
        - :ref:`kvstore/neuroglancer_uint64_sharded`

        For regular, non-adapter key-value stores, this is :python:`None`.

        Example:

            >>> store = await ts.KvStore.open({
            ...     'driver': 'ocdbt',
            ...     'base': 'memory://'
            ... })
            >>> store.base
            KvStore({'context': {'memory_key_value_store': {}}, 'driver': 'memory'})

        See also:

          - :py:obj:`KvStore.Spec.base`

        Group:
          Accessors
        """

    @property
    def path(self) -> str:
        """
        Path prefix within the base key-value store.

        Example:

            >>> store = await ts.KvStore.open({
            ...     'driver': 'gcs',
            ...     'bucket': 'my-bucket',
            ...     'path': 'path/to/object'
            ... })
            >>> store.spec()
            KvStore.Spec({'bucket': 'my-bucket', 'driver': 'gcs', 'path': 'path/to/object'})
            >>> store.path
            'path/to/object'

        Group:
          Accessors
        """

    @path.setter
    def path(self, arg1: str | bytes) -> None: ...

    @property
    def transaction(self) -> Transaction | None:
        """
        Transaction bound to this key-value store.

        Group:
          Transactions
        """

    @transaction.setter
    def transaction(self, arg1: Transaction | None) -> None: ...

    @property
    def url(self) -> str:
        """
        :json:schema:`URL representation<KvStoreUrl>` of the key-value store specification.

        Example:

            >>> store = await ts.KvStore.open({
            ...     'driver': 'gcs',
            ...     'bucket': 'my-bucket',
            ...     'path': 'path/to/object'
            ... })
            >>> store.url
            'gs://my-bucket/path/to/object'

        Group:
          Accessors
        """


class OpenMode:
    """

    Specifies the mode to use when opening a `TensorStore`.

    Group:
      Spec
    """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: OpenMode) -> bool: ...

    def __init__(
        self,
        *,
        open: bool = False,
        create: bool = False,
        delete_existing: bool = False,
        assume_metadata: bool = False,
        assume_cached_metadata: bool = False,
    ) -> None:
        """
        Constructs an open mode.

        Args:
          open: Allow opening an existing TensorStore.
          create: Allow creating a new TensorStore.
          delete_existing: Delete any existing data before creating a new array.
          assume_metadata: Don't access the stored metadata.
          assume_cached_metadata: Skip reading the metadata when opening.
        """

    def __repr__(self) -> str: ...

    @property
    def assume_cached_metadata(self) -> bool:
        """
        Skip reading the metadata when opening.


        Group:
          Accessors
        """

    @assume_cached_metadata.setter
    def assume_cached_metadata(self, arg1: bool) -> None: ...

    @property
    def assume_metadata(self) -> bool:
        """
        Don't access the stored metadata.


        Group:
          Accessors
        """

    @assume_metadata.setter
    def assume_metadata(self, arg1: bool) -> None: ...

    @property
    def create(self) -> bool:
        """
        Allow creating a new TensorStore.


        Group:
          Accessors
        """

    @create.setter
    def create(self, arg1: bool) -> None: ...

    @property
    def delete_existing(self) -> bool:
        """
        Delete any existing data before creating a new array.


        Group:
          Accessors
        """

    @delete_existing.setter
    def delete_existing(self, arg1: bool) -> None: ...

    @property
    def open(self) -> bool:
        """
        Allow opening an existing TensorStore.


        Group:
          Accessors
        """

    @open.setter
    def open(self, arg1: bool) -> None: ...


class OutputIndexMap:
    """

    Represents an output index map for an index transform.

    See also:
      - :py:obj:`IndexTransform.output`
      - :py:obj:`OutputIndexMaps`
      - :py:obj:`OutputIndexMethod`

    Group:
      Indexing
    """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, other: OutputIndexMap) -> bool: ...

    @typing.overload
    def __init__(self, offset: int = 0) -> None:
        """
        Constructs a :ref:`constant map<index-transform-constant-map>`.

        Example:

            >>> transform = ts.IndexTransform(input_rank=0,
            ...                               output=[ts.OutputIndexMap(offset=5)])
            >>> transform([])
            (5,)

        Overload:
          constant
        """

    @typing.overload
    def __init__(self, input_dimension: int, offset: int = 0, stride: int = 1) -> None:
        """
        Constructs a :ref:`single input dimension map<index-transform-single-input-dimension-map>`.

        Example:

            >>> transform = ts.IndexTransform(
            ...     input_rank=1,
            ...     output=[ts.OutputIndexMap(input_dimension=0, offset=5, stride=2)])
            >>> [transform([i]) for i in range(5)]
            [(5,), (7,), (9,), (11,), (13,)]

        Overload:
          input_dimension
        """

    @typing.overload
    def __init__(
        self,
        index_array: numpy.typing.ArrayLike,
        offset: int = 0,
        stride: int = 1,
        index_range: Dim = ...,
    ) -> None:
        """
        Constructs an :ref:`index array map<index-transform-array-map>`.

        Example:

            >>> transform = ts.IndexTransform(
            ...     input_shape=[5],
            ...     output=[ts.OutputIndexMap(index_array=[2, 3, 5, 7, 11])])
            >>> [transform([i]) for i in range(5)]
            [(2,), (3,), (5,), (7,), (11,)]

        Overload:
          index_array
        """

    def __repr__(self) -> str: ...

    @property
    def index_array(self) -> numpy.ndarray | None: ...

    @property
    def index_range(self) -> Dim | None: ...

    @property
    def input_dimension(self) -> int | None: ...

    @property
    def method(self) -> OutputIndexMethod: ...

    @property
    def offset(self) -> int: ...

    @property
    def stride(self) -> int | None: ...


class OutputIndexMaps:
    """

    View of the output index maps for an index transform.

    See also:
      - :py:obj:`IndexTransform.output`
      - :py:obj:`OutputIndexMap`
      - :py:obj:`OutputIndexMethod`

    Group:
      Indexing
    """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, arg0: collections.abc.Iterable[OutputIndexMap]) -> bool: ...

    @typing.overload
    def __getitem__(self, arg0: int) -> OutputIndexMap:
        """
        Returns the output index map for the specified output dimension.

        Overload:
          index

        Group:
          Sequence accessors
        """

    @typing.overload
    def __getitem__(self, arg0: slice) -> list[OutputIndexMap]:
        """
        Returns the list of output index maps corresponding to the slice.

        Overload:
          slice

        Group:
          Sequence accessors
        """

    def __len__(self) -> int:
        """
        Returns the output rank.

        Group:
          Sequence accessors
        """

    def __repr__(self) -> str: ...

    @property
    def rank(self) -> int:
        """
        Returns the output rank.
        """

    def __iter__(self) -> collections.abc.Iterator[OutputIndexMap]: ...


class OutputIndexMethod:
    """

    Indicates the :ref:`output index method<output-index-methods>` of an :py:class:`OutputIndexMap`.

    See also:
      - :py:obj:`IndexTransform.output`
      - :py:obj:`OutputIndexMap`
      - :py:obj:`OutputIndexMaps`

    Group:
      Indexing



    Members:

      constant

      single_input_dimension

      array
    """

    __members__: typing.ClassVar[dict[str, OutputIndexMethod]]
    array: typing.ClassVar[OutputIndexMethod]
    constant: typing.ClassVar[OutputIndexMethod]
    single_input_dimension: typing.ClassVar[OutputIndexMethod]

    def __eq__(self, other: typing.Any) -> bool: ...

    def __hash__(self) -> int: ...

    def __index__(self) -> int: ...

    def __init__(self, value: int) -> None: ...

    def __int__(self) -> int: ...

    def __ne__(self, other: typing.Any) -> bool: ...

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def value(self) -> int: ...


class Promise(typing.Generic[T_contra]):
    """

    Handle for *producing* the result of an asynchronous operation.

    A promise represents the producer interface corresponding to a
    :py:class:`Future`, and may be used to signal the completion of an asynchronous
    operation.

        >>> promise, future = ts.Promise.new()
        >>> future.done()
        False
        >>> promise.set_result(5)
        >>> future.done()
        True
        >>> future.result()
        5

    :py:class:`Promise` and :py:class:`Future` can also be used with type
    parameters:

        >>> promise, future = ts.Promise[int].new()
        >>> typing.assert_type(promise, ts.Promise[int])
        >>> typing.assert_type(future, ts.Future[int])
        >>> promise.set_result(5)
        >>> future.result()
        5

    See also:
      - :py:class:`Future`

    Type parameters:
      T:
        Result type of the asynchronous operation.

    Group:
      Asynchronous support
    """

    @staticmethod
    def new() -> tuple[Promise[T_contra], Future[T_contra]]:
        """
        Creates a linked promise and future pair.

        Group:
          Constructors
        """

    def set_exception(self, exception: typing.Any) -> None:
        """
        Marks the linked future as unsuccessfully completed with the specified error.

        Example:

            >>> promise, future = ts.Promise.new()
            >>> future.done()
            False
            >>> promise.set_exception(Exception(5))
            >>> future.done()
            True
            >>> future.result()
            Traceback (most recent call last):
                ...
            Exception: 5

        Group:
          Operations
        """

    def set_result(self, result: T_contra) -> None:
        """
        Marks the linked future as successfully completed with the specified result.

        Example:

            >>> promise, future = ts.Promise.new()
            >>> future.done()
            False
            >>> promise.set_result(5)
            >>> future.done()
            True
            >>> future.result()
            5

        Group:
          Operations
        """


class Schema:
    """

    Driver-independent options for defining a TensorStore schema.

    Group:
      Spec
    """

    class _Label:
        __iter__ = None

        def __getitem__(self, labels: str | collections.abc.Iterable[str]) -> Schema:
            """
            Returns a new view with the :ref:`dimension labels<dimension-labels>` changed.

            This is equivalent to :python:`self[ts.d[:].label[labels]]`.

            Args:
              labels: Dimension labels for each dimension.

            Raises:

              IndexError: If the number of labels does not match the number of dimensions,
                or if the resultant domain would have duplicate labels.

            See also:
              - `tensorstore.DimExpression.label`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> Schema:
            """
            Returns a new view with the lower/upper bounds changed to
            :ref:`implicit/explicit<implicit-bounds>`.

            This is equivalent to :python:`self[ts.d[:].mark_bounds_implicit[implicit]]`.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.
                Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Raises:

              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            See also:
              - `tensorstore.DimExpression.mark_bounds_implicit`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Oindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> Schema:
            """
            Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
            integer or boolean array indexing terms are applied orthogonally.

            Example:

                >>> schema = ts.Schema(
                ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
                ...                           shape=[1000, 2000, 3000]),
                ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
                ...                                 inner_order=[0, 1, 2]),
                ... )
                >>> schema.oindex[[5, 10, 20], [7, 8, 10]]
                Schema({
                  'chunk_layout': {'grid_origin': [None, None, 300], 'inner_order': [2, 0, 1]},
                  'domain': {
                    'exclusive_max': [3, 3, 3000],
                    'inclusive_min': [0, 0, 0],
                    'labels': ['', '', 'z'],
                  },
                  'rank': 3,
                })

            Returns:
              New schema with the indexing operation applied.

            Raises:
              ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`IndexTransform.oindex`
               - :py:obj:`Schema.__getitem__(indices)`
               - :py:obj:`Schema.vindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> Schema:
            """
            Returns a new view with the `.origin` translated backward by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_backward_by[offsets]]`.

            Args:

              offsets: The offset for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_backward_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> Schema:
            """
            Returns a new view with the `.origin` translated by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_by[offsets]]`.

            Args:

              offsets: The offset for each dimension.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> Schema:
            """
            Returns a new view with `.origin` translated to the specified origin.

            This is equivalent to :python:`self[ts.d[:].translate_to[origins]]`.

            Args:

              origins: The new origin for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same origin is used for all dimensions.
                If :python:`None` is specified for a given dimension, the origin of that
                dimension remains unchanged.

            Raises:

              IndexError:
                If the number origins does not match the number of dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            See also:
              - `tensorstore.DimExpression.translate_to`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Vindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> Schema:
            """
            Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
            :python:`indices` specifies any array indexing terms, the broadcasted array
            dimensions are unconditionally added as the first dimensions of the result
            domain.

            Example:

                >>> schema = ts.Schema(
                ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
                ...                           shape=[1000, 2000, 3000]),
                ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
                ...                                 inner_order=[0, 1, 2]),
                ... )
                >>> schema.vindex[[5, 10, 20], [7, 8, 10]]
                Schema({
                  'chunk_layout': {'grid_origin': [None, 300], 'inner_order': [1, 0]},
                  'domain': {
                    'exclusive_max': [3, 3000],
                    'inclusive_min': [0, 0],
                    'labels': ['', 'z'],
                  },
                  'rank': 2,
                })

            Returns:
              New schema with the indexing operation applied.

            Raises:
              ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`IndexTransform.vindex`
               - :py:obj:`Schema.__getitem__(indices)`
               - :py:obj:`Schema.oindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    __hash__: typing.ClassVar[None] = None

    def __copy__(self) -> Schema: ...

    def __deepcopy__(self, memo: dict) -> Schema: ...

    def __eq__(self, other: Schema) -> bool:
        """
        Compares with another :py:obj:`Schema` for equality based on the :json:schema:`JSON representation<Schema>`.

        The comparison is based on the JSON representation.

        Example:

          >>> schema = ts.Schema(dtype=ts.int32, rank=3)
          >>> assert schema == schema
          >>> a, b = spec.copy(), spec.copy()
          >>> a.update(fill_value=42)
          >>> assert a == a
          >>> assert a != b
        """

    @typing.overload
    def __getitem__(self, transform: IndexTransform) -> Schema:
        """
        Transforms the schema using an explicit :ref:`index transform<index-transform>`.

        Example:

            >>> schema = ts.Schema(
            ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
            ...                           shape=[1000, 2000, 3000]),
            ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
            ...                                 inner_order=[0, 1, 2]),
            ... )
            >>> transform = ts.IndexTransform(
            ...     input_shape=[3],
            ...     output=[
            ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
            ...         ts.OutputIndexMap(index_array=[5, 4, 3])
            ...     ])
            >>> schema[transform]
            Traceback (most recent call last):
                ...
            IndexError: Rank 3 -> 3 transform cannot be composed with rank 1 -> 2 transform...

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.

        Returns:

          New schema of rank :python:`transform.input_rank`.

        Raises:
          ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`IndexTransform.__getitem__(transform)`
           - :py:obj:`Schema.__getitem__(indices)`
           - :py:obj:`Schema.__getitem__(expr)`
           - :py:obj:`Schema.__getitem__(domain)`
           - :py:obj:`Schema.oindex`
           - :py:obj:`Schema.vindex`

        Overload:
          transform

        Group:
          Indexing
        """

    @typing.overload
    def __getitem__(self, domain: IndexDomain) -> Schema:
        """
        Transforms the schema using an explicit :ref:`index domain<index-domain>`.

        The domain of the resultant spec is computed as in
        :py:obj:`IndexDomain.__getitem__(domain)`.

        Example:

            >>> schema = ts.Schema(
            ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
            ...                           shape=[1000, 2000, 3000]),
            ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
            ...                                 inner_order=[0, 1, 2]),
            ... )
            >>> domain = ts.IndexDomain(labels=['x', 'z'],
            ...                         inclusive_min=[5, 6],
            ...                         exclusive_max=[8, 9])
            >>> schema[domain]
            Schema({
              'chunk_layout': {'grid_origin': [100, 200, 300], 'inner_order': [0, 1, 2]},
              'domain': {
                'exclusive_max': [8, 2000, 9],
                'inclusive_min': [5, 0, 6],
                'labels': ['x', 'y', 'z'],
              },
              'rank': 3,
            })

        Args:

          domain: Index domain, must have dimension labels that can be
            :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

        Returns:

          New schema with domain equal to :python:`self.domain[domain]`.

        Raises:
          ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

        Group:
          Indexing

        Overload:
          domain

        See also:

           - :ref:`index-domain`
           - :ref:`index-domain-alignment`
           - :py:obj:`IndexDomain.__getitem__(domain)`
        """

    @typing.overload
    def __getitem__(self, expr: DimExpression) -> Schema:
        """
        Transforms the schema using a :ref:`dimension expression<python-dim-expressions>`.

        Example:

            >>> schema = ts.Schema(
            ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
            ...                           shape=[1000, 2000, 3000]),
            ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
            ...                                 inner_order=[0, 1, 2]),
            ... )
            >>> schema[ts.d['x', 'z'][5:10, 6:9]]
            Schema({
              'chunk_layout': {'grid_origin': [100, 200, 300], 'inner_order': [0, 1, 2]},
              'domain': {
                'exclusive_max': [10, 2000, 9],
                'inclusive_min': [5, 0, 6],
                'labels': ['x', 'y', 'z'],
              },
              'rank': 3,
            })

        Returns:
          New schema with domain equal to :python:`self.domain[expr]`.

        Raises:
          ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

        Group:
          Indexing

        Overload:
          expr

        See also:

           - :ref:`python-dim-expressions`
           - :py:obj:`IndexDomain.__getitem__(expr)`
        """

    @typing.overload
    def __getitem__(self, indices: NumpyIndexingSpec) -> Schema:
        """
        Transforms the schema using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

        Example:

            >>> schema = ts.Schema(
            ...     domain=ts.IndexDomain(labels=['x', 'y', 'z'],
            ...                           shape=[1000, 2000, 3000]),
            ...     chunk_layout=ts.ChunkLayout(grid_origin=[100, 200, 300],
            ...                                 inner_order=[0, 1, 2]),
            ... )
            >>> schema[[5, 10, 20], 6:10]
            Schema({
              'chunk_layout': {'grid_origin': [None, 200, 300], 'inner_order': [1, 2, 0]},
              'domain': {
                'exclusive_max': [3, 10, 3000],
                'inclusive_min': [0, 6, 0],
                'labels': ['', 'y', 'z'],
              },
              'rank': 3,
            })

        Returns:
          New schema with the indexing operation applied.

        Raises:
          ValueError: If :py:obj:`self.rank<.rank>` is :python:`None`.

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`IndexTransform.__getitem__(indices)`
           - :py:obj:`Schema.oindex`
           - :py:obj:`Schema.vindex`

        Group:
          Indexing

        Overload:
          indices
        """

    @typing.overload
    def __init__(self, json: typing.Any) -> None:
        """
        Constructs from its :json:schema:`JSON representation<Schema>`.

        Example:

          >>> ts.Schema({
          ...     'dtype': 'uint8',
          ...     'chunk_layout': {
          ...         'grid_origin': [1, 2, 3],
          ...         'inner_order': [0, 2, 1]
          ...     }
          ... })
          Schema({
            'chunk_layout': {'grid_origin': [1, 2, 3], 'inner_order': [0, 2, 1]},
            'dtype': 'uint8',
            'rank': 3,
          })

        Overload:
          json
        """

    @typing.overload
    def __init__(
        self,
        *,
        rank: int | None = None,
        dtype: DTypeLike | None = None,
        domain: IndexDomain | None = None,
        shape: collections.abc.Iterable[int] | None = None,
        chunk_layout: ChunkLayout | None = None,
        codec: CodecSpec | None = None,
        fill_value: numpy.typing.ArrayLike | None = None,
        dimension_units: (
            collections.abc.Iterable[
                Unit | str | numbers.Real | tuple[numbers.Real, str] | None
            ]
            | None
        ) = None,
        schema: Schema | None = None,
    ) -> None:
        """
        Constructs from component parts.

        Example:

          >>> ts.Schema(dtype=ts.uint8,
          ...           chunk_layout=ts.ChunkLayout(grid_origin=[1, 2, 3],
          ...                                       inner_order=[0, 2, 1]))
          Schema({
            'chunk_layout': {'grid_origin': [1, 2, 3], 'inner_order': [0, 2, 1]},
            'dtype': 'uint8',
            'rank': 3,
          })

        Args:
          rank: Constrains the rank of the TensorStore.  If there is an index transform, the
            rank constraint must match the rank of the *input* space.
          dtype: Constrains the data type of the TensorStore.  If a data type has already been
            set, it is an error to specify a different data type.
          domain: Constrains the domain of the TensorStore.  If there is an existing
            domain, the specified domain is merged with it as follows:

            1. The rank must match the existing rank.

            2. All bounds must match, except that a finite or explicit bound is permitted to
               match an infinite and implicit bound, and takes precedence.

            3. If both the new and existing domain specify non-empty labels for a dimension,
               the labels must be equal.  If only one of the domains specifies a non-empty
               label for a dimension, the non-empty label takes precedence.

            Note that if there is an index transform, the domain must match the *input*
            space, not the output space.
          shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
            :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

            .. note::

               This option also constrains the origin of all dimensions to be zero.
          chunk_layout: Constrains the chunk layout.  If there is an existing chunk layout constraint,
            the constraints are merged.  If the constraints are incompatible, an error
            is raised.
          codec: Constrains the codec.  If there is an existing codec constraint, the constraints
            are merged.  If the constraints are incompatible, an error is raised.
          fill_value: Specifies the fill value for positions that have not been written.

            The fill value data type must be convertible to the actual data type, and the
            shape must be :ref:`broadcast-compatible<index-domain-alignment>` with the
            domain.

            If an existing fill value has already been set as a constraint, it is an
            error to specify a different fill value (where the comparison is done after
            normalization by broadcasting).
          dimension_units: Specifies the physical units of each dimension of the domain.

            The *physical unit* for a dimension is the physical quantity corresponding to a
            single index increment along each dimension.

            A value of :python:`None` indicates that the unit is unknown.  A dimension-less
            quantity can be indicated by a unit of :python:`""`.
          schema: Additional schema constraints to merge with existing constraints.


        Overload:
          components
        """

    def __repr__(self) -> str:
        """
        Returns a string representation based on the  :json:schema:`JSON representation<Schema>`.

        Example:

          >>> schema = ts.Schema(rank=5, dtype=ts.uint8)
          >>> schema
          Schema({'dtype': 'uint8', 'rank': 5})
        """

    def copy(self) -> Schema:
        """
        Returns a copy of the schema.

        Example:

          >>> a = ts.Schema(dtype=ts.uint8)
          >>> b = a.copy()
          >>> a.update(rank=2)
          >>> b.update(rank=3)
          >>> a
          Schema({'dtype': 'uint8', 'rank': 2})
          >>> b
          Schema({'dtype': 'uint8', 'rank': 3})

        Group:
          Accessors
        """

    def to_json(self, include_defaults: bool = False) -> typing.Any:
        """
        Converts to the :json:schema:`JSON representation<Schema>`.

        Example:

          >>> schema = ts.Schema(dtype=ts.uint8,
          ...                    chunk_layout=ts.ChunkLayout(grid_origin=[0, 0, 0],
          ...                                                inner_order=[0, 2, 1]))
          >>> schema.to_json()
          {'chunk_layout': {'grid_origin': [0, 0, 0], 'inner_order': [0, 2, 1]},
           'dtype': 'uint8',
           'rank': 3}

        Group:
          Accessors
        """

    def transpose(self, axes: DimSelectionLike | None = None) -> Schema:
        """
        Returns a view with a transposed domain.

        This is equivalent to :python:`self[ts.d[axes].transpose[:]]`.

        Args:

          axes: Specifies the existing dimension corresponding to each dimension of the
            new view.  Dimensions may be specified either by index or label.  Specifying
            `None` is equivalent to specifying :python:`[rank-1, ..., 0]`, which
            reverses the dimension order.

        Raises:

          ValueError: If :py:param:`.axes` does not specify a valid permutation.

        See also:
          - `tensorstore.DimExpression.transpose`
          - :py:obj:`.T`

        Group:
          Indexing
        """

    def update(
        self,
        *,
        rank: int | None = None,
        dtype: DTypeLike | None = None,
        domain: IndexDomain | None = None,
        shape: collections.abc.Iterable[int] | None = None,
        chunk_layout: ChunkLayout | None = None,
        codec: CodecSpec | None = None,
        fill_value: numpy.typing.ArrayLike | None = None,
        dimension_units: (
            collections.abc.Iterable[
                Unit | str | numbers.Real | tuple[numbers.Real, str] | None
            ]
            | None
        ) = None,
        schema: Schema | None = None,
    ) -> None:
        """
        Adds additional constraints.

        Example:

          >>> schema = ts.Schema(rank=3)
          >>> schema
          Schema({'rank': 3})
          >>> schema.update(dtype=ts.uint8)
          >>> schema
          Schema({'dtype': 'uint8', 'rank': 3})

        Args:
          rank: Constrains the rank of the TensorStore.  If there is an index transform, the
            rank constraint must match the rank of the *input* space.
          dtype: Constrains the data type of the TensorStore.  If a data type has already been
            set, it is an error to specify a different data type.
          domain: Constrains the domain of the TensorStore.  If there is an existing
            domain, the specified domain is merged with it as follows:

            1. The rank must match the existing rank.

            2. All bounds must match, except that a finite or explicit bound is permitted to
               match an infinite and implicit bound, and takes precedence.

            3. If both the new and existing domain specify non-empty labels for a dimension,
               the labels must be equal.  If only one of the domains specifies a non-empty
               label for a dimension, the non-empty label takes precedence.

            Note that if there is an index transform, the domain must match the *input*
            space, not the output space.
          shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
            :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

            .. note::

               This option also constrains the origin of all dimensions to be zero.
          chunk_layout: Constrains the chunk layout.  If there is an existing chunk layout constraint,
            the constraints are merged.  If the constraints are incompatible, an error
            is raised.
          codec: Constrains the codec.  If there is an existing codec constraint, the constraints
            are merged.  If the constraints are incompatible, an error is raised.
          fill_value: Specifies the fill value for positions that have not been written.

            The fill value data type must be convertible to the actual data type, and the
            shape must be :ref:`broadcast-compatible<index-domain-alignment>` with the
            domain.

            If an existing fill value has already been set as a constraint, it is an
            error to specify a different fill value (where the comparison is done after
            normalization by broadcasting).
          dimension_units: Specifies the physical units of each dimension of the domain.

            The *physical unit* for a dimension is the physical quantity corresponding to a
            single index increment along each dimension.

            A value of :python:`None` indicates that the unit is unknown.  A dimension-less
            quantity can be indicated by a unit of :python:`""`.
          schema: Additional schema constraints to merge with existing constraints.


        Group:
          Mutators
        """

    @property
    def T(self) -> Schema:
        """
        View with transposed domain (reversed dimension order).

        This is equivalent to: :python:`self[ts.d[::-1].transpose[:]]`.

        See also:
          - `.transpose`
          - `tensorstore.DimExpression.transpose`

        Group:
          Indexing
        """

    @property
    def chunk_layout(self) -> ChunkLayout:
        """
        Chunk layout constraints specified by the schema.

        Example:

          >>> schema = ts.Schema(chunk_layout=ts.ChunkLayout(inner_order=[0, 1, 2]))
          >>> schema.update(chunk_layout=ts.ChunkLayout(grid_origin=[0, 0, 0]))
          >>> schema.chunk_layout
          ChunkLayout({'grid_origin': [0, 0, 0], 'inner_order': [0, 1, 2]})

        Note:

          Each access to this property returns a new copy of the chunk layout.
          Modifying the returned chunk layout (e.g. by calling
          :py:obj:`tensorstore.ChunkLayout.update`) will not affect the schema object
          from which it was obtained.

        Group:
          Accessors
        """

    @property
    def codec(self) -> CodecSpec | None:
        """
        Codec constraints specified by the schema.

        Example:

          >>> schema = ts.Schema()
          >>> print(schema.codec)
          None
          >>> schema.update(codec=ts.CodecSpec({
          ...     'driver': 'zarr',
          ...     'compressor': None
          ... }))
          >>> schema.update(codec=ts.CodecSpec({'driver': 'zarr', 'filters': None}))
          >>> schema.codec
          CodecSpec({'compressor': None, 'driver': 'zarr', 'filters': None})

        Group:
          Accessors
        """

    @property
    def dimension_units(self) -> tuple[Unit | None, ...] | None:
        """
        Physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown/unconstrained.  A
        dimension-less quantity is indicated by a unit of :python:`ts.Unit(1, "")`.

        When creating a new TensorStore, the specified units may be stored as part of
        the metadata.

        When opening an existing TensorStore, the specified units serve as a constraint,
        to ensure the units are as expected.  Additionally, for drivers like
        :ref:`neuroglancer_precomputed<driver/neuroglancer_precomputed>` that support
        multiple scales, the desired scale can be selected by specifying constraints on
        the units.

        Example:

          >>> schema = ts.Schema()
          >>> print(schema.dimension_units)
          None
          >>> schema.update(rank=3)
          >>> schema.dimension_units
          (None, None, None)
          >>> schema.update(dimension_units=['3nm', None, ''])
          >>> schema.dimension_units
          (Unit(3, "nm"), None, Unit(1, ""))
          >>> schema.update(dimension_units=[None, '4nm', None])
          >>> schema.dimension_units
          (Unit(3, "nm"), Unit(4, "nm"), Unit(1, ""))

        Group:
          Accessors
        """

    @property
    def domain(self) -> IndexDomain | None:
        """
        Domain of the schema, or `None` if unspecified.

        Example:

          >>> schema = ts.Schema()
          >>> print(schema.domain)
          None
          >>> schema.update(domain=ts.IndexDomain(labels=['x', 'y', 'z']))
          >>> schema.update(domain=ts.IndexDomain(shape=[100, 200, 300]))
          >>> schema.domain
          { "x": [0, 100), "y": [0, 200), "z": [0, 300) }

        Group:
          Accessors
        """

    @property
    def dtype(self) -> dtype | None:
        """
        Data type, or :python:`None` if unspecified.

        Example:

          >>> schema = ts.Schema(rank=3)
          >>> print(spec.dtype)
          None

          >>> spec = ts.Schema(dtype=ts.uint8, rank=3)
          >>> spec.dtype
          dtype("uint8")

        Group:
          Accessors
        """

    @property
    def fill_value(self) -> numpy.ndarray | None:
        """
        Fill value specified by the schema.

        Example:

          >>> schema = ts.Schema()
          >>> print(schema.fill_value)
          None
          >>> schema.update(fill_value=42)
          >>> schema.fill_value
          array(42)

        Group:
          Accessors
        """

    @property
    def label(self) -> Schema._Label: ...

    @property
    def mark_bounds_implicit(self) -> Schema._MarkBoundsImplicit: ...

    @property
    def ndim(self) -> int | None:
        """
        Alias for :py:obj:`.rank`.

        Example:

          >>> schema = ts.Schema(rank=3)
          >>> schema.ndim
          3

        Group:
          Accessors
        """

    @property
    def oindex(self) -> Schema._Oindex: ...

    @property
    def origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain.

        This is equivalent to :python:`self.domain.origin`.

        Group:
          Accessors
        """

    @property
    def rank(self) -> int | None:
        """
        Rank of the schema, or `None` if unspecified.

        Example:

          >>> schema = ts.Schema(dtype=ts.uint8)
          >>> print(schema.rank)
          None
          >>> schema.update(chunk_layout=ts.ChunkLayout(grid_origin=[0, 1, 2]))
          >>> schema.rank
          3

        Group:
          Accessors
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the domain.

        This is equivalent to :python:`self.domain.shape`.

        Group:
          Accessors
        """

    @property
    def size(self) -> int:
        """
        Total number of elements in the domain.

        This is equivalent to :python:`self.domain.size`.

        Group:
          Accessors
        """

    @property
    def translate_backward_by(self) -> Schema._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> Schema._TranslateBy: ...

    @property
    def translate_to(self) -> Schema._TranslateTo: ...

    @property
    def vindex(self) -> Schema._Vindex: ...


class Spec(Indexable):
    """

    Specification for opening or creating a :py:obj:`.TensorStore`.

    Group:
      Spec

    Constructors
    ============

    Accessors
    =========

    Indexing
    ========

    Comparison operators
    ====================

    """

    class _Label:
        __iter__ = None

        def __getitem__(self, labels: str | collections.abc.Iterable[str]) -> Spec:
            """
            Returns a new view with the :ref:`dimension labels<dimension-labels>` changed.

            This is equivalent to :python:`self[ts.d[:].label[labels]]`.

            Args:
              labels: Dimension labels for each dimension.

            Raises:

              IndexError: If the number of labels does not match the number of dimensions,
                or if the resultant domain would have duplicate labels.

            See also:
              - `tensorstore.DimExpression.label`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> Spec:
            """
            Returns a new view with the lower/upper bounds changed to
            :ref:`implicit/explicit<implicit-bounds>`.

            This is equivalent to :python:`self[ts.d[:].mark_bounds_implicit[implicit]]`.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.
                Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Raises:

              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            See also:
              - `tensorstore.DimExpression.mark_bounds_implicit`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Oindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> Spec:
            """
            Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
            integer or boolean array indexing terms are applied orthogonally.

            Example:

                >>> spec = ts.Spec({
                ...     'driver': 'zarr',
                ...     'kvstore': {
                ...         'driver': 'memory'
                ...     },
                ...     'transform': {
                ...         'input_shape': [[70], [80]],
                ...     }
                ... })
                >>> spec.oindex[[5, 10, 20], [7, 8, 10]]
                Spec({
                  'driver': 'zarr',
                  'kvstore': {'driver': 'memory'},
                  'transform': {
                    'input_exclusive_max': [3, 3],
                    'input_inclusive_min': [0, 0],
                    'output': [
                      {'index_array': [[5], [10], [20]]},
                      {'index_array': [[7, 8, 10]]},
                    ],
                  },
                })

            Returns:
              New spec with the indexing operation applied.

            Raises:
              ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`IndexTransform.oindex`
               - :py:obj:`Spec.__getitem__(indices)`
               - :py:obj:`Spec.vindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> Spec:
            """
            Returns a new view with the `.origin` translated backward by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_backward_by[offsets]]`.

            Args:

              offsets: The offset for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_backward_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> Spec:
            """
            Returns a new view with the `.origin` translated by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_by[offsets]]`.

            Args:

              offsets: The offset for each dimension.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> Spec:
            """
            Returns a new view with `.origin` translated to the specified origin.

            This is equivalent to :python:`self[ts.d[:].translate_to[origins]]`.

            Args:

              origins: The new origin for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same origin is used for all dimensions.
                If :python:`None` is specified for a given dimension, the origin of that
                dimension remains unchanged.

            Raises:

              IndexError:
                If the number origins does not match the number of dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            See also:
              - `tensorstore.DimExpression.translate_to`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Vindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> Spec:
            """
            Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
            :python:`indices` specifies any array indexing terms, the broadcasted array
            dimensions are unconditionally added as the first dimensions of the result
            domain.

            Example:

                >>> spec = ts.Spec({
                ...     'driver': 'zarr',
                ...     'kvstore': {
                ...         'driver': 'memory'
                ...     },
                ...     'transform': {
                ...         'input_shape': [[70], [80]],
                ...     }
                ... })
                >>> spec.vindex[[5, 10, 20], [7, 8, 10]]
                Spec({
                  'driver': 'zarr',
                  'kvstore': {'driver': 'memory'},
                  'transform': {
                    'input_exclusive_max': [3],
                    'input_inclusive_min': [0],
                    'output': [{'index_array': [5, 10, 20]}, {'index_array': [7, 8, 10]}],
                  },
                })

            Returns:
              New spec with the indexing operation applied.

            Raises:
              ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`IndexTransform.vindex`
               - :py:obj:`Spec.__getitem__(indices)`
               - :py:obj:`Spec.oindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    __hash__: typing.ClassVar[None] = None
    __iter__ = None

    def __copy__(self) -> Spec: ...

    def __deepcopy__(self, memo: dict) -> Spec: ...

    def __eq__(self, other: Spec) -> bool:
        """
        Compares with another :py:obj:`Spec` for equality based on the :json:schema:`JSON representation<TensorStore>`.

        The comparison is based on the JSON representation, except that any bound
        context resources are compared by identity (not by their JSON representation).

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> assert spec == spec
          >>> a, b = spec.copy(), spec.copy()
          >>> context_a, context_b = ts.Context(), ts.Context()
          >>> a.update(context=context_a)
          >>> b.update(context=context_b)
          >>> assert a == a
          >>> assert a != b
        """

    @typing.overload
    def __getitem__(self, transform: IndexTransform) -> Spec:
        """
        Transforms the spec using an explicit :ref:`index transform<index-transform>`.

        This composes :python:`self.transform` with :python:`transform`.

        Example:

            >>> spec = ts.Spec({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     },
            ...     'transform': {
            ...         'input_shape': [[70], [80]],
            ...     }
            ... })
            >>> transform = ts.IndexTransform(
            ...     input_shape=[3],
            ...     output=[
            ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
            ...         ts.OutputIndexMap(index_array=[5, 4, 3])
            ...     ])
            >>> spec[transform]
            Spec({
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [3],
                'input_inclusive_min': [0],
                'output': [{'index_array': [1, 2, 3]}, {'index_array': [5, 4, 3]}],
              },
            })

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.

        Returns:

          New spec of rank :python:`transform.input_rank` and transform
          :python:`self.transform[transform]`.

        Raises:
          ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`IndexTransform.__getitem__(transform)`
           - :py:obj:`Spec.__getitem__(indices)`
           - :py:obj:`Spec.__getitem__(expr)`
           - :py:obj:`Spec.__getitem__(domain)`
           - :py:obj:`Spec.oindex`
           - :py:obj:`Spec.vindex`

        Overload:
          transform

        Group:
          Indexing
        """

    @typing.overload
    def __getitem__(self, domain: IndexDomain) -> Spec:
        """
        Transforms the spec using an explicit :ref:`index domain<index-domain>`.

        The transform of the resultant spec is computed as in
        :py:obj:`IndexTransform.__getitem__(domain)`.

        Example:

            >>> spec = ts.Spec({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     },
            ...     'transform': {
            ...         'input_shape': [[60], [70], [80]],
            ...         'input_labels': ['x', 'y', 'z'],
            ...     }
            ... })
            >>> domain = ts.IndexDomain(labels=['x', 'z'],
            ...                         inclusive_min=[5, 6],
            ...                         exclusive_max=[8, 9])
            >>> spec[domain]
            Spec({
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [8, [70], 9],
                'input_inclusive_min': [5, 0, 6],
                'input_labels': ['x', 'y', 'z'],
              },
            })

        Args:

          domain: Index domain, must have dimension labels that can be
            :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

        Returns:

          New spec with transform equal to :python:`self.transform[domain]`.

        Raises:
          ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

        Group:
          Indexing

        Overload:
          domain

        See also:

           - :ref:`index-domain`
           - :ref:`index-domain-alignment`
           - :py:obj:`IndexTransform.__getitem__(domain)`
        """

    @typing.overload
    def __getitem__(self, expr: DimExpression) -> Spec:
        """
        Transforms the spec using a :ref:`dimension expression<python-dim-expressions>`.

        Example:

            >>> spec = ts.Spec({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     },
            ...     'transform': {
            ...         'input_shape': [[60], [70], [80]],
            ...         'input_labels': ['x', 'y', 'z'],
            ...     }
            ... })
            >>> spec[ts.d['x', 'z'][5:10, 6:9]]
            Spec({
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [10, [70], 9],
                'input_inclusive_min': [5, 0, 6],
                'input_labels': ['x', 'y', 'z'],
              },
            })

        Returns:
          New spec with transform equal to :python:`self.transform[expr]`.

        Raises:
          ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

        Group:
          Indexing

        Overload:
          expr

        See also:

           - :ref:`python-dim-expressions`
           - :py:obj:`IndexTransform.__getitem__(expr)`
        """

    @typing.overload
    def __getitem__(self, indices: NumpyIndexingSpec) -> Spec:
        """
        Transforms the spec using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

        Example:

            >>> spec = ts.Spec({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     },
            ...     'transform': {
            ...         'input_shape': [[70], [80]],
            ...     }
            ... })
            >>> spec[[5, 10, 20], 6:10]
            Spec({
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [3, 10],
                'input_inclusive_min': [0, 6],
                'output': [{'index_array': [[5], [10], [20]]}, {'input_dimension': 1}],
              },
            })

        Returns:
          New spec with the indexing operation applied.

        Raises:
          ValueError: If :py:obj:`self.transform<.transform>` is :python:`None`.

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`IndexTransform.__getitem__(indices)`
           - :py:obj:`Spec.oindex`
           - :py:obj:`Spec.vindex`

        Group:
          Indexing

        Overload:
          indices
        """

    def __new__(self, json: typing.Any) -> Spec:
        """
        Constructs from the :json:schema:`JSON representation<TensorStore>`.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> spec
          Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
        """

    def __repr__(self) -> str:
        """
        Returns a string representation based on the :json:schema:`JSON representation<TensorStore>`.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> spec
          Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})

          Bound :json:schema:`context resources<ContextResource>` are indicated by
          single-element arrays:

          >>> spec.update(context=ts.Context())
          >>> spec
          Spec({
            'cache_pool': ['cache_pool'],
            'context': {
              'cache_pool': {},
              'data_copy_concurrency': {},
              'memory_key_value_store': {},
            },
            'data_copy_concurrency': ['data_copy_concurrency'],
            'driver': 'n5',
            'kvstore': {
              'driver': 'memory',
              'memory_key_value_store': ['memory_key_value_store'],
            },
          })
        """

    def copy(self) -> Spec:
        """
        Returns a copy of the spec.

        Example:

          >>> a = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> b = a.copy()
          >>> a.update(dtype=ts.uint8)
          >>> b.update(dtype=ts.uint16)
          >>> a
          Spec({'driver': 'n5', 'dtype': 'uint8', 'kvstore': {'driver': 'memory'}})
          >>> b
          Spec({'driver': 'n5', 'dtype': 'uint16', 'kvstore': {'driver': 'memory'}})

        Group:
          Accessors
        """

    def to_json(self, include_defaults: bool = False) -> typing.Any:
        """
        Converts to the :json:schema:`JSON representation<TensorStore>`.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dimensions': [100, 200]
          ...     }
          ... })
          >>> spec = spec[ts.d[0].translate_by[5]]
          >>> spec.to_json()
          {'driver': 'n5',
           'kvstore': {'driver': 'memory'},
           'metadata': {'dimensions': [100, 200]},
           'transform': {'input_exclusive_max': [[105], [200]],
                         'input_inclusive_min': [5, 0],
                         'output': [{'input_dimension': 0, 'offset': -5},
                                    {'input_dimension': 1}]}}

        Group:
          Accessors
        """

    def transpose(self, axes: DimSelectionLike | None = None) -> Spec:
        """
        Returns a view with a transposed domain.

        This is equivalent to :python:`self[ts.d[axes].transpose[:]]`.

        Args:

          axes: Specifies the existing dimension corresponding to each dimension of the
            new view.  Dimensions may be specified either by index or label.  Specifying
            `None` is equivalent to specifying :python:`[rank-1, ..., 0]`, which
            reverses the dimension order.

        Raises:

          ValueError: If :py:param:`.axes` does not specify a valid permutation.

        See also:
          - `tensorstore.DimExpression.transpose`
          - :py:obj:`.T`

        Group:
          Indexing
        """

    def update(
        self,
        *,
        open_mode: OpenMode | None = None,
        open: bool | None = None,
        create: bool | None = None,
        delete_existing: bool | None = None,
        assume_metadata: bool | None = None,
        assume_cached_metadata: bool | None = None,
        unbind_context: bool | None = None,
        strip_context: bool | None = None,
        context: Context | None = None,
        kvstore: KvStore.Spec | None = None,
        minimal_spec: bool | None = None,
        recheck_cached_metadata: RecheckCacheOption | None = None,
        recheck_cached_data: RecheckCacheOption | None = None,
        recheck_cached: RecheckCacheOption | None = None,
        rank: int | None = None,
        dtype: DTypeLike | None = None,
        domain: IndexDomain | None = None,
        shape: collections.abc.Iterable[int] | None = None,
        chunk_layout: ChunkLayout | None = None,
        codec: CodecSpec | None = None,
        fill_value: numpy.typing.ArrayLike | None = None,
        dimension_units: (
            collections.abc.Iterable[
                Unit | str | numbers.Real | tuple[numbers.Real, str] | None
            ]
            | None
        ) = None,
        schema: Schema | None = None,
    ) -> None:
        """
        Adds additional constraints or changes the open mode.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> spec.update(shape=[100, 200, 300])
          >>> spec
          Spec({
            'driver': 'n5',
            'kvstore': {'driver': 'memory'},
            'schema': {
              'domain': {'exclusive_max': [100, 200, 300], 'inclusive_min': [0, 0, 0]},
            },
            'transform': {
              'input_exclusive_max': [[100], [200], [300]],
              'input_inclusive_min': [0, 0, 0],
            },
          })

        Args:
          open_mode: Overrides the existing open mode.
          open: Allow opening an existing TensorStore.  Overrides the existing open mode.
          create: Allow creating a new TensorStore.  Overrides the existing open mode.  To open or
            create, specify :python:`create=True` and :python:`open=True`.
          delete_existing: Delete any existing data before creating a new array.  Overrides the existing
            open mode.  Must be specified in conjunction with :python:`create=True`.
          assume_metadata: Neither read nor write stored metadata.  Instead, just assume any necessary
            metadata based on constraints in the spec, using the same defaults for any
            unspecified metadata as when creating a new TensorStore.  The stored metadata
            need not even exist.  Operations such as resizing that modify the stored
            metadata are not supported.  Overrides the existing open mode.  Requires that
            :py:param:`.open` is `True` and :py:param:`.delete_existing` is `False`.  This
            option takes precedence over `.assume_cached_metadata` if that option is also
            specified.

            .. warning::

               This option can lead to data corruption if the assumed metadata does
               not match the stored metadata, or multiple concurrent writers use
               different assumed metadata.

            .. seealso:

               - :ref:`python-open-assume-metadata`
          assume_cached_metadata: Skip reading the metadata when opening.  Instead, just assume any necessary
            metadata based on constraints in the spec, using the same defaults for any
            unspecified metadata as when creating a new TensorStore.  The stored metadata
            may still be accessed by subsequent operations that need to re-validate or
            modify the metadata.  Requires that :py:param:`.open` is `True` and
            :py:param:`.delete_existing` is `False`.  The :py:param:`.assume_metadata`
            option takes precedence if also specified.

            .. warning::

               This option can lead to data corruption if the assumed metadata does
               not match the stored metadata, or multiple concurrent writers use
               different assumed metadata.

            .. seealso:

               - :ref:`python-open-assume-metadata`
          unbind_context: Convert any bound context resources to context resource specs that fully capture
            the graph of shared context resources and interdependencies.

            Re-binding/re-opening the resultant spec will result in a new graph of new
            context resources that is isomorphic to the original graph of context resources.
            The resultant spec will not refer to any external context resources;
            consequently, binding it to any specific context will have the same effect as
            binding it to a default context.

            Specifying a value of :python:`False` has no effect.
          strip_context: Replace any bound context resources and unbound context resource specs by
            default context resource specs.

            If the resultant :py:obj:`~tensorstore.Spec` is re-opened with, or re-bound to,
            a new context, it will use the default context resources specified by that
            context.

            Specifying a value of :python:`False` has no effect.
          context: Bind any context resource specs using the specified shared resource context.

            Any already-bound context resources remain unchanged.  Additionally, any context
            resources specified by a nested :json:schema:`TensorStore.context` spec will be
            created as specified, but won't be overridden by :py:param:`.context`.
          kvstore: Sets the associated key-value store used as the underlying storage.

            If the :py:obj:`~tensorstore.Spec.kvstore` has already been set, it is
            overridden.

            It is an error to specify this if the TensorStore driver does not use a
            key-value store.
          minimal_spec: Indicates whether to include in the :py:obj:`~tensorstore.Spec` returned by
            :py:obj:`tensorstore.TensorStore.spec` the metadata necessary to re-create the
            :py:obj:`~tensorstore.TensorStore`. By default, the returned
            :py:obj:`~tensorstore.Spec` includes the full metadata, but it is skipped if
            :py:param:`.minimal_spec` is set to :python:`True`.

            When applied to an existing :py:obj:`~tensorstore.Spec` via
            :py:obj:`tensorstore.open` or :py:obj:`tensorstore.Spec.update`, only ``False``
            has any effect.
          recheck_cached_metadata: Time after which cached metadata is assumed to be fresh. Cached metadata older
            than the specified time is revalidated prior to use. The metadata is used to
            check the bounds of every read or write operation.

            Specifying ``True`` means that the metadata will be revalidated prior to every
            read or write operation. With the default value of ``"open"``, any cached
            metadata is revalidated when the TensorStore is opened but is not rechecked for
            each read or write operation.
          recheck_cached_data: Time after which cached data is assumed to be fresh. Cached data older than the
            specified time is revalidated prior to being returned from a read operation.
            Partial chunk writes are always consistent regardless of the value of this
            option.

            The default value of ``True`` means that cached data is revalidated on every
            read. To enable in-memory data caching, you must both specify a
            :json:schema:`~Context.cache_pool` with a non-zero
            :json:schema:`~Context.cache_pool.total_bytes_limit` and also specify ``False``,
            ``"open"``, or an explicit time bound for :py:param:`.recheck_cached_data`.
          recheck_cached: Sets both :py:param:`.recheck_cached_data` and
            :py:param:`.recheck_cached_metadata`.
          rank: Constrains the rank of the TensorStore.  If there is an index transform, the
            rank constraint must match the rank of the *input* space.
          dtype: Constrains the data type of the TensorStore.  If a data type has already been
            set, it is an error to specify a different data type.
          domain: Constrains the domain of the TensorStore.  If there is an existing
            domain, the specified domain is merged with it as follows:

            1. The rank must match the existing rank.

            2. All bounds must match, except that a finite or explicit bound is permitted to
               match an infinite and implicit bound, and takes precedence.

            3. If both the new and existing domain specify non-empty labels for a dimension,
               the labels must be equal.  If only one of the domains specifies a non-empty
               label for a dimension, the non-empty label takes precedence.

            Note that if there is an index transform, the domain must match the *input*
            space, not the output space.
          shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
            :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

            .. note::

               This option also constrains the origin of all dimensions to be zero.
          chunk_layout: Constrains the chunk layout.  If there is an existing chunk layout constraint,
            the constraints are merged.  If the constraints are incompatible, an error
            is raised.
          codec: Constrains the codec.  If there is an existing codec constraint, the constraints
            are merged.  If the constraints are incompatible, an error is raised.
          fill_value: Specifies the fill value for positions that have not been written.

            The fill value data type must be convertible to the actual data type, and the
            shape must be :ref:`broadcast-compatible<index-domain-alignment>` with the
            domain.

            If an existing fill value has already been set as a constraint, it is an
            error to specify a different fill value (where the comparison is done after
            normalization by broadcasting).
          dimension_units: Specifies the physical units of each dimension of the domain.

            The *physical unit* for a dimension is the physical quantity corresponding to a
            single index increment along each dimension.

            A value of :python:`None` indicates that the unit is unknown.  A dimension-less
            quantity can be indicated by a unit of :python:`""`.
          schema: Additional schema constraints to merge with existing constraints.


        Group:
          Mutators
        """

    @property
    def T(self) -> Spec:
        """
        View with transposed domain (reversed dimension order).

        This is equivalent to: :python:`self[ts.d[::-1].transpose[:]]`.

        See also:
          - `.transpose`
          - `tensorstore.DimExpression.transpose`

        Group:
          Indexing
        """

    @property
    def base(self) -> Spec | None:
        """
        Spec of the underlying `TensorStore`, if this is an adapter of a single
        underlying `TensorStore`.

        Otherwise, equal to :python:`None`.

        Drivers that support this method include:

        - :ref:`driver/cast`
        - :ref:`driver/downsample`

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': 'memory://',
          ... })
          >>> spec.update(shape=[100, 200], dtype=ts.uint32)
          >>> cast_spec = ts.cast(spec, ts.float32)
          >>> cast_spec
          Spec({
            'base': {
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'schema': {
                'domain': {'exclusive_max': [100, 200], 'inclusive_min': [0, 0]},
              },
            },
            'driver': 'cast',
            'dtype': 'float32',
            'transform': {
              'input_exclusive_max': [[100], [200]],
              'input_inclusive_min': [0, 0],
            },
          })
          >>> cast_spec[30:40, 20:25].base
          Spec({
            'driver': 'zarr',
            'dtype': 'uint32',
            'kvstore': {'driver': 'memory'},
            'schema': {'domain': {'exclusive_max': [100, 200], 'inclusive_min': [0, 0]}},
            'transform': {
              'input_exclusive_max': [40, 25],
              'input_inclusive_min': [30, 20],
            },
          })
          >>> downsampled_spec = ts.downsample(spec,
          ...                                  downsample_factors=[2, 4],
          ...                                  method='mean')
          >>> downsampled_spec
          Spec({
            'base': {
              'driver': 'zarr',
              'kvstore': {'driver': 'memory'},
              'schema': {
                'domain': {'exclusive_max': [100, 200], 'inclusive_min': [0, 0]},
              },
              'transform': {
                'input_exclusive_max': [[100], [200]],
                'input_inclusive_min': [0, 0],
              },
            },
            'downsample_factors': [2, 4],
            'downsample_method': 'mean',
            'driver': 'downsample',
            'dtype': 'uint32',
            'transform': {
              'input_exclusive_max': [[50], [50]],
              'input_inclusive_min': [0, 0],
            },
          })
          >>> downsampled_spec[30:40, 20:25].base

        Group:
          Accessors
        """

    @property
    def chunk_layout(self) -> ChunkLayout:
        """
        Effective :ref:`chunk layout<chunk-layout>`, including any constraints implied
        by driver-specific options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'chunks': [100, 200, 300],
          ...         'order': 'C'
          ...     }
          ... })
          >>> spec.chunk_layout
          ChunkLayout({})

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def codec(self) -> CodecSpec | None:
        """
        Effective :ref:`codec<codec>`, including any constraints implied
        by driver-specific options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'compressor': None,
          ...     }
          ... })
          >>> spec.codec
          CodecSpec({'compressor': None, 'driver': 'zarr'})

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def dimension_units(self) -> tuple[Unit | None, ...] | None:
        """
        Effective physical units of each dimension of the domain, including any
        constraints implied by driver-specific options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'units': ['nm', 'nm', 'um'],
          ...         'resolution': [200, 300, 1],
          ...     }
          ... })
          >>> spec.dimension_units
          (Unit(200, "nm"), Unit(300, "nm"), Unit(1, "um"))

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def domain(self) -> IndexDomain | None:
        """
        Effective :ref:`index domain<index-domain>`, including any constraints implied
        by driver-specific options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dtype': '<u2',
          ...         'shape': [1000, 2000, 3000],
          ...     }
          ... })
          >>> spec.domain
          { [0, 1000*), [0, 2000*), [0, 3000*) }

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def dtype(self) -> dtype | None:
        """
        Data type, or :python:`None` if unspecified.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> print(spec.dtype)
          None

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dataType': 'uint8'
          ...     }
          ... })
          >>> spec.dtype
          dtype("uint8")

        Group:
          Accessors
        """

    @property
    def fill_value(self) -> numpy.ndarray | None:
        """
        Effective fill value, including any constraints implied by driver-specific
        options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'compressor': None,
          ...         'dtype': '<f4',
          ...         'fill_value': 42,
          ...     }
          ... })
          >>> spec.fill_value
          array(42., dtype=float32)

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def kvstore(self) -> KvStore.Spec | None:
        """
        Spec of the associated key-value store used as the underlying storage.

        Equal to :python:`None` if the driver does not use a key-value store or the
        key-value store has not been specified.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory',
          ...         'path': 'abc/',
          ...     },
          ... })
          >>> spec.kvstore
          KvStore.Spec({'driver': 'memory', 'path': 'abc/'})

        Group:
          Accessors
        """

    @property
    def label(self) -> Spec._Label: ...

    @property
    def mark_bounds_implicit(self) -> Spec._MarkBoundsImplicit: ...

    @property
    def ndim(self) -> int | None:
        """
        Alias for :py:obj:`.rank`.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> print(spec.ndim)
          None

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dimensions': [100, 200]
          ...     }
          ... })
          >>> spec.ndim
          2

        Group:
          Accessors
        """

    @property
    def oindex(self) -> Spec._Oindex: ...

    @property
    def open_mode(self) -> OpenMode:
        """
        Open mode with which the driver will be opened.

        If not applicable, equal to :python:`OpenMode()`.

        Example:

          >>> spec = ts.Spec({'driver': 'zarr', 'kvstore': 'memory://'})
          >>> spec.open_mode
          OpenMode(open=True)
          >>> spec.update(create=True, delete_existing=True)
          >>> spec.open_mode
          OpenMode(create=True, delete_existing=True)
          >>> spec.update(open_mode=ts.OpenMode(open=True, create=True))
          >>> spec.open_mode
          OpenMode(open=True, create=True)

        .. note::

           This is a read-only accessor.  Mutating the returned `OpenMode` object does
           not affect this `Spec` object.  To change the open mode, use `.update`.

        Group:
          Accessors
        """

    @property
    def origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain.

        This is equivalent to :python:`self.domain.origin`.

        Group:
          Accessors
        """

    @property
    def rank(self) -> int | None:
        """
        Returns the rank of the domain, or `None` if unspecified.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> print(spec.rank)
          None

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dimensions': [100, 200]
          ...     }
          ... })
          >>> spec.rank
          2

        Group:
          Accessors
        """

    @property
    def schema(self) -> Schema:
        """
        Effective :ref:`schema<schema>`, including any constraints implied by driver-specific options.

        Example:

          >>> spec = ts.Spec({
          ...     'driver': 'zarr',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dtype': '<u2',
          ...         'chunks': [100, 200, 300],
          ...         'shape': [1000, 2000, 3000],
          ...         'order': 'C'
          ...     }
          ... })
          >>> spec.schema
          Schema({
            'chunk_layout': {
              'grid_origin': [0, 0, 0],
              'inner_order': [0, 1, 2],
              'read_chunk': {'shape': [100, 200, 300]},
              'write_chunk': {'shape': [100, 200, 300]},
            },
            'codec': {'driver': 'zarr'},
            'domain': {
              'exclusive_max': [[1000], [2000], [3000]],
              'inclusive_min': [0, 0, 0],
            },
            'dtype': 'uint16',
            'rank': 3,
          })

        Note:

          This does not perform any I/O.  Only directly-specified constraints are
          included.

        Group:
          Accessors
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the domain.

        This is equivalent to :python:`self.domain.shape`.

        Group:
          Accessors
        """

    @property
    def size(self) -> int:
        """
        Total number of elements in the domain.

        This is equivalent to :python:`self.domain.size`.

        Group:
          Accessors
        """

    @property
    def transform(self) -> IndexTransform | None:
        """
        The :ref:`index transform<index-transform>`, or `None` if unspecified.

        Example:

          >>> spec = ts.Spec({'driver': 'n5', 'kvstore': {'driver': 'memory'}})
          >>> print(spec.transform)
          None

          >>> spec = ts.Spec({
          ...     'driver': 'n5',
          ...     'kvstore': {
          ...         'driver': 'memory'
          ...     },
          ...     'metadata': {
          ...         'dimensions': [100, 200],
          ...         'axes': ['x', 'y']
          ...     }
          ... })
          >>> spec.transform
          Rank 2 -> 2 index space transform:
            Input domain:
              0: [0, 100*) "x"
              1: [0, 200*) "y"
            Output index maps:
              out[0] = 0 + 1 * in[0]
              out[1] = 0 + 1 * in[1]
          >>> spec[ts.d['x'].translate_by[5]].transform
          Rank 2 -> 2 index space transform:
            Input domain:
              0: [5, 105*) "x"
              1: [0, 200*) "y"
            Output index maps:
              out[0] = -5 + 1 * in[0]
              out[1] = 0 + 1 * in[1]

        Group:
          Accessors
        """

    @property
    def translate_backward_by(self) -> Spec._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> Spec._TranslateBy: ...

    @property
    def translate_to(self) -> Spec._TranslateTo: ...

    @property
    def url(self) -> str:
        """
        :json:schema:`URL representation<TensorStoreUrl>` of the TensorStore specification.

        Example:

            >>> spec = ts.Spec({
            ...     'driver': 'n5',
            ...     'kvstore': {
            ...         'driver': 'gcs',
            ...         'bucket': 'my-bucket',
            ...         'path': 'path/to/array/'
            ...     }
            ... })
            >>> spec.url
            'gs://my-bucket/path/to/array/|n5:'

        Group:
          Accessors
        """

    @property
    def vindex(self) -> Spec._Vindex: ...


class TensorStore(Indexable):
    """

    Asynchronous multi-dimensional array handle.

    Examples:

        >>> dataset = await ts.open(
        ...     {
        ...         'driver': 'zarr',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         },
        ...     },
        ...     dtype=ts.uint32,
        ...     shape=[1000, 20000],
        ...     create=True)
        >>> dataset
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'zarr',
          'dtype': 'uint32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'chunks': [1000, 1048],
            'compressor': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'id': 'blosc',
              'shuffle': -1,
            },
            'dimension_separator': '.',
            'dtype': '<u4',
            'fill_value': None,
            'filters': None,
            'order': 'C',
            'shape': [1000, 20000],
            'zarr_format': 2,
          },
          'transform': {
            'input_exclusive_max': [[1000], [20000]],
            'input_inclusive_min': [0, 0],
          },
        })
        >>> await dataset[5:10, 6:8].write(42)
        >>> await dataset[0:10, 0:10].read()
        array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
               [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0]], dtype=uint32)

    Group:
      Core
    """

    class StorageStatistics:
        """

        Statistics related to the storage of an array specified by a :py:class:`TensorStore`.

        .. seealso::

           :py:obj:`tensorstore.TensorStore.storage_statistics`

        These statistics provide information about the elements of an array that are
        *stored*, but depending on the :ref:`driver<tensorstore-drivers>`, whether data
        is stored for a given element is not necessarily equivalent to whether that
        element has been successfully written:

        - There are cases where an element may be stored even if it has not been
          explicitly written.  For example, when using a
          :ref:`chunked storage driver<chunked-drivers>`, an entire chunk must be stored
          in order to store any element within the chunk, and it is not possible to
          determine which elements of the chunk were explicitly written.  If any chunk
          corresponding to a region that intersects the domain is stored, then
          :py:obj:`.not_stored` will be :python:`False`, even if no element actually within
          the domain was explicitly written.  Similarly, if at least one element of each
          chunk that intersects the domain is stored, then :py:obj:`.fully_stored` will be
          :python:`True`, even if no element of the domain was every explicitly written.

        - Some drivers may not store chunks that are entirely equal to the
          :py:obj:`TensorStore.fill_value`.  With such drivers, if all elements of the
          domain are equal to the fill value, even if some or all of the elements have
          been explicitly written, :py:obj:`.not_stored` may be :python:`True`.

        Group:
          I/O
        """

        __hash__: typing.ClassVar[None] = None

        def __eq__(self, arg0: TensorStore.StorageStatistics) -> bool: ...

        def __init__(
            self, *, not_stored: bool | None = None, fully_stored: bool | None = None
        ) -> None:
            """
            Constructs from attribute values.
            """

        def __repr__(self) -> str: ...

        @property
        def fully_stored(self) -> bool | None:
            """
            Indicates whether data is stored for *all* elements of the specified :py:obj:`~TensorStore.domain`.

            For the statistics returned by :py:obj:`TensorStore.storage_statistics`, if
            :py:param:`~TensorStore.storage_statistics.query_fully_stored` is not set to
            :python:`True`, then this will be `None`.
            """

        @fully_stored.setter
        def fully_stored(self, arg1: bool | None) -> None: ...

        @property
        def not_stored(self) -> bool | None:
            """
            Indicates whether *no* data is stored for the specified :py:obj:`~TensorStore.domain`.

            For the statistics returned by :py:obj:`TensorStore.storage_statistics`, if
            :py:param:`~TensorStore.storage_statistics.query_not_stored` is not set to
            :python:`True`, then this will be `None`.

            If :python:`False`, it is guaranteed that all elements within the domain are equal
            to the :py:obj:`~TensorStore.fill_value`.
            """

        @not_stored.setter
        def not_stored(self, arg1: bool | None) -> None: ...

    class _Label:
        __iter__ = None

        def __getitem__(
            self, labels: str | collections.abc.Iterable[str]
        ) -> TensorStore:
            """
            Returns a new view with the :ref:`dimension labels<dimension-labels>` changed.

            This is equivalent to :python:`self[ts.d[:].label[labels]]`.

            Args:
              labels: Dimension labels for each dimension.

            Raises:

              IndexError: If the number of labels does not match the number of dimensions,
                or if the resultant domain would have duplicate labels.

            See also:
              - `tensorstore.DimExpression.label`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _MarkBoundsImplicit:
        __iter__ = None

        def __getitem__(self, implicit: bool | None | slice) -> TensorStore:
            """
            Returns a new view with the lower/upper bounds changed to
            :ref:`implicit/explicit<implicit-bounds>`.

            This is equivalent to :python:`self[ts.d[:].mark_bounds_implicit[implicit]]`.

            Args:

              implicit: Indicates the new implicit value for the lower and upper bounds.
                Must be one of:

                - `None` to indicate no change;
                - `True` to change both lower and upper bounds to implicit;
                - `False` to change both lower and upper bounds to explicit.
                - a `slice`, where :python:`start` and :python:`stop` specify the new
                  implicit value for the lower and upper bounds, respectively, and each must
                  be one of `None`, `True`, or `False`.

            Raises:

              IndexError: If the resultant domain would have an input dimension referenced
                by an index array marked as implicit.

            See also:
              - `tensorstore.DimExpression.mark_bounds_implicit`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Oindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> TensorStore:
            """
            Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that any
            integer or boolean array indexing terms are applied orthogonally:

                >>> dataset = await ts.open(
                ...     {
                ...         'driver': 'zarr',
                ...         'kvstore': {
                ...             'driver': 'memory'
                ...         }
                ...     },
                ...     dtype=ts.uint32,
                ...     shape=[70, 80],
                ...     create=True)
                >>> view = dataset.oindex[[5, 10, 20], [7, 8, 10]]
                >>> view
                TensorStore({
                  'context': {
                    'cache_pool': {},
                    'data_copy_concurrency': {},
                    'memory_key_value_store': {},
                  },
                  'driver': 'zarr',
                  'dtype': 'uint32',
                  'kvstore': {'driver': 'memory'},
                  'metadata': {
                    'chunks': [70, 80],
                    'compressor': {
                      'blocksize': 0,
                      'clevel': 5,
                      'cname': 'lz4',
                      'id': 'blosc',
                      'shuffle': -1,
                    },
                    'dimension_separator': '.',
                    'dtype': '<u4',
                    'fill_value': None,
                    'filters': None,
                    'order': 'C',
                    'shape': [70, 80],
                    'zarr_format': 2,
                  },
                  'transform': {
                    'input_exclusive_max': [3, 3],
                    'input_inclusive_min': [0, 0],
                    'output': [
                      {'index_array': [[5], [10], [20]]},
                      {'index_array': [[7, 8, 10]]},
                    ],
                  },
                })

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`TensorStore.__getitem__(indices)`
               - :py:obj:`TensorStore.vindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

        def __setitem__(
            self,
            indices: NumpyIndexingSpec,
            source: TensorStore | numpy.typing.ArrayLike,
        ) -> None:
            """
            Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`outer indexing semantics<python-oindex-indexing>`.

            This is similar to :py:obj:`.__setitem__(indices)`, but differs in that any integer or
            boolean array indexing terms are applied orthogonally:

                >>> dataset = ts.open({
                ...     'driver': 'zarr',
                ...     'kvstore': {
                ...         'driver': 'memory'
                ...     }
                ... },
                ...                   dtype=ts.uint32,
                ...                   shape=[70, 80],
                ...                   create=True).result()
                >>> dataset.oindex[[5, 6, 8], [2, 5]] = [1, 2]
                >>> dataset[5:10, 0:6].read().result()
                array([[0, 0, 1, 0, 0, 2],
                       [0, 0, 1, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 2],
                       [0, 0, 0, 0, 0, 0]], dtype=uint32)

            Args:
              indices: NumPy-style indexing terms.
              source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
                :python:`self.oindex[indices].domain` and with a data type convertible to
                :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
                :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

            .. warning::

               When *not* using a transaction, the subscript assignment syntax always blocks
               synchronously on the completion of the write operation.  When performing
               multiple, fine-grained writes, it is recommended to either use a transaction
               or use the asynchronous :py:obj:`TensorStore.write` interface directly.

            Group:
              I/O

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`TensorStore.write`
               - :py:obj:`TensorStore.__setitem__(indices)`
               - :py:obj:`TensorStore.vindex.__setitem__`
            """

    class _TranslateBackwardBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> TensorStore:
            """
            Returns a new view with the `.origin` translated backward by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_backward_by[offsets]]`.

            Args:

              offsets: The offset for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_backward_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateBy:
        __iter__ = None

        def __getitem__(
            self, offsets: collections.abc.Iterable[int | None] | int | None
        ) -> TensorStore:
            """
            Returns a new view with the `.origin` translated by the specified offsets.

            This is equivalent to :python:`self[ts.d[:].translate_by[offsets]]`.

            Args:

              offsets: The offset for each dimension.  May also be a scalar,
                e.g. :python:`5`, in which case the same offset is used for all dimensions.
                Specifying :python:`None` for a given dimension (equivalent to specifying an
                offset of :python:`0`) leaves the origin of that dimension unchanged.

            See also:
              - `tensorstore.DimExpression.translate_by`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _TranslateTo:
        __iter__ = None

        def __getitem__(
            self, origins: collections.abc.Iterable[int | None] | int | None
        ) -> TensorStore:
            """
            Returns a new view with `.origin` translated to the specified origin.

            This is equivalent to :python:`self[ts.d[:].translate_to[origins]]`.

            Args:

              origins: The new origin for each dimensions.  May also be a scalar,
                e.g. :python:`5`, in which case the same origin is used for all dimensions.
                If :python:`None` is specified for a given dimension, the origin of that
                dimension remains unchanged.

            Raises:

              IndexError:
                If the number origins does not match the number of dimensions.

              IndexError:
                If any of the selected dimensions has a lower bound of :python:`-inf`.

            See also:
              - `tensorstore.DimExpression.translate_to`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

    class _Vindex:
        __iter__ = None

        def __getitem__(self, indices: NumpyIndexingSpec) -> TensorStore:
            """
            Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`.__getitem__(indices)`, but differs in that if
            :python:`indices` specifies any array indexing terms, the broadcasted array
            dimensions are unconditionally added as the first dimensions of the result
            domain:

                >>> dataset = await ts.open(
                ...     {
                ...         'driver': 'zarr',
                ...         'kvstore': {
                ...             'driver': 'memory'
                ...         }
                ...     },
                ...     dtype=ts.uint32,
                ...     shape=[60, 70, 80],
                ...     create=True)
                >>> view = dataset.vindex[:, [5, 10, 20], [7, 8, 10]]
                >>> view
                TensorStore({
                  'context': {
                    'cache_pool': {},
                    'data_copy_concurrency': {},
                    'memory_key_value_store': {},
                  },
                  'driver': 'zarr',
                  'dtype': 'uint32',
                  'kvstore': {'driver': 'memory'},
                  'metadata': {
                    'chunks': [60, 70, 80],
                    'compressor': {
                      'blocksize': 0,
                      'clevel': 5,
                      'cname': 'lz4',
                      'id': 'blosc',
                      'shuffle': -1,
                    },
                    'dimension_separator': '.',
                    'dtype': '<u4',
                    'fill_value': None,
                    'filters': None,
                    'order': 'C',
                    'shape': [60, 70, 80],
                    'zarr_format': 2,
                  },
                  'transform': {
                    'input_exclusive_max': [3, [60]],
                    'input_inclusive_min': [0, 0],
                    'output': [
                      {'input_dimension': 1},
                      {'index_array': [[5], [10], [20]]},
                      {'index_array': [[7], [8], [10]]},
                    ],
                  },
                })

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`TensorStore.__getitem__(indices)`
               - :py:obj:`TensorStore.oindex`

            Group:
              Indexing
            """

        def __repr__(self) -> str: ...

        def __setitem__(
            self,
            indices: NumpyIndexingSpec,
            source: TensorStore | numpy.typing.ArrayLike,
        ) -> None:
            """
            Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with :ref:`vectorized indexing semantics<python-vindex-indexing>`.

            This is similar to :py:obj:`.__setitem__(indices)`, but differs in that if
            :python:`indices` specifies any array indexing terms, the broadcasted array
            dimensions are unconditionally added as the first dimensions of the
            domain to be aligned to :python:`source`:

                >>> dataset = ts.open({
                ...     'driver': 'zarr',
                ...     'kvstore': {
                ...         'driver': 'memory'
                ...     }
                ... },
                ...                   dtype=ts.uint32,
                ...                   shape=[2, 70, 80],
                ...                   create=True).result()
                >>> dataset.vindex[:, [5, 6, 8], [2, 5, 6]] = [[1, 2], [3, 4], [5, 6]]
                >>> dataset[:, 5:10, 0:6].read().result()
                array([[[0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 3],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]],
                <BLANKLINE>
                       [[0, 0, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0, 4],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]]], dtype=uint32)

            Args:
              indices: NumPy-style indexing terms.
              source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
                :python:`self.vindex[indices].domain` and with a data type convertible to
                :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
                :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

            .. warning::

               When *not* using a transaction, the subscript assignment syntax always blocks
               synchronously on the completion of the write operation.  When performing
               multiple, fine-grained writes, it is recommended to either use a transaction
               or use the asynchronous :py:obj:`TensorStore.write` interface directly.

            Group:
              I/O

            See also:

               - :ref:`python-numpy-style-indexing`
               - :py:obj:`TensorStore.write`
               - :py:obj:`TensorStore.__setitem__(indices)`
               - :py:obj:`TensorStore.oindex.__setitem__`
            """

    __iter__ = None

    def __array__(
        self,
        dtype: numpy.dtype[typing.Any] | None = None,
        copy: bool | None = None,
        context: typing.Any | None = None,
    ) -> numpy.ndarray:
        """
        Automatic conversion to `numpy.ndarray` for interoperability with NumPy.

        *Synchronously* reads from the current domain and returns the result as an array.
        Equivalent to :python:`self.read().result()`.

        Examples:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> dataset[10:20, 5:10] + np.array(5, dtype=np.uint32)
            array([[5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5],
                   [5, 5, 5, 5, 5]], dtype=uint32)

        .. warning::

           This reads the entire domain into memory and blocks the current thread while
           reading.  For large arrays, it may be better to partition the domain into
           blocks and process each block separately.

        See also:

           - :py:obj:`.read`

        Group:
          I/O
        """

    def __copy__(self) -> typing.Any: ...

    @typing.overload
    def __getitem__(self, transform: IndexTransform) -> TensorStore:
        """
        Computes a virtual view using an explicit :ref:`index transform<index-transform>`.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> transform = ts.IndexTransform(
            ...     input_shape=[3],
            ...     output=[
            ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
            ...         ts.OutputIndexMap(index_array=[5, 4, 3])
            ...     ])
            >>> dataset[transform]
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                'chunks': [70, 80],
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'dimension_separator': '.',
                'dtype': '<u4',
                'fill_value': None,
                'filters': None,
                'order': 'C',
                'shape': [70, 80],
                'zarr_format': 2,
              },
              'transform': {
                'input_exclusive_max': [3],
                'input_inclusive_min': [0],
                'output': [{'index_array': [1, 2, 3]}, {'index_array': [5, 4, 3]}],
              },
            })
            >>> await dataset[transform].write([1, 2, 3])
            >>> await dataset[1:6, 1:6].read()
            array([[0, 0, 0, 0, 1],
                   [0, 0, 0, 2, 0],
                   [0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=uint32)

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.

        Returns:

          View of rank :python:`transform.input_rank` and domain
          :python:`self.domain[transform]`.

        This is the most general form of indexing, to which all other indexing methods
        reduce:

        - :python:`self[expr]` is equivalent to
          :python:`self[ts.IndexTransform(self.domain)[expr]]`

        - :python:`self.oindex[expr]` is equivalent to
          :python:`self[ts.IndexTransform(self.domain).oindex[expr]]`

        - :python:`self.vindex[expr]` is equivalent to
          :python:`self[ts.IndexTransform(self.domain).vindex[expr]]`

        In most cases it is more convenient to use one of those other indexing forms
        instead.

        Group:
          Indexing

        Overload:
          transform

        See also:

           - :ref:`index-transform`
           - :py:obj:`TensorStore.__getitem__(indices)`
           - :py:obj:`TensorStore.__getitem__(domain)`
           - :py:obj:`TensorStore.__getitem__(expr)`
           - :py:obj:`TensorStore.oindex`
           - :py:obj:`TensorStore.vindex`
        """

    @typing.overload
    def __getitem__(self, domain: IndexDomain) -> TensorStore:
        """
        Computes a virtual view using an explicit :ref:`index domain<index-domain>`.

        The domain of the resultant view is computed as in
        :py:obj:`IndexDomain.__getitem__(domain)`.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     domain=ts.IndexDomain(shape=[60, 70, 80], labels=['x', 'y', 'z']),
            ...     create=True)
            >>> domain = ts.IndexDomain(labels=['x', 'z'],
            ...                         inclusive_min=[5, 6],
            ...                         exclusive_max=[8, 9])
            >>> dataset[domain]
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'n5',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                'axes': ['x', 'y', 'z'],
                'blockSize': [60, 70, 80],
                'compression': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'shuffle': 1,
                  'type': 'blosc',
                },
                'dataType': 'uint32',
                'dimensions': [60, 70, 80],
              },
              'transform': {
                'input_exclusive_max': [8, [70], 9],
                'input_inclusive_min': [5, 0, 6],
                'input_labels': ['x', 'y', 'z'],
              },
            })

        Args:

          domain: Index domain, must have dimension labels that can be
            :ref:`aligned<index-domain-alignment>` to :python:`self.domain`.

        Returns:

          Virtual view with domain equal to :python:`self.domain[domain]`.

        Group:
          Indexing

        Overload:
          domain

        See also:

           - :ref:`index-domain`
           - :ref:`index-domain-alignment`
           - :py:obj:`IndexTransform.__getitem__(domain)`
        """

    @typing.overload
    def __getitem__(self, expr: DimExpression) -> TensorStore:
        """
        Computes a virtual view using a :ref:`dimension expression<python-dim-expressions>`.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     domain=ts.IndexDomain(shape=[60, 70, 80], labels=['x', 'y', 'z']),
            ...     create=True)
            >>> dataset[ts.d['x', 'z'][5:10, 6:9]]
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'n5',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                'axes': ['x', 'y', 'z'],
                'blockSize': [60, 70, 80],
                'compression': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'shuffle': 1,
                  'type': 'blosc',
                },
                'dataType': 'uint32',
                'dimensions': [60, 70, 80],
              },
              'transform': {
                'input_exclusive_max': [10, [70], 9],
                'input_inclusive_min': [5, 0, 6],
                'input_labels': ['x', 'y', 'z'],
              },
            })

        Returns:

          Virtual view with the dimension expression applied.

        Group:
          Indexing

        Overload:
          expr

        See also:

           - :ref:`python-dim-expressions`
           - :py:obj:`IndexTransform.__getitem__(expr)`
        """

    @typing.overload
    def __getitem__(self, indices: NumpyIndexingSpec) -> TensorStore:
        """
        Computes a virtual view using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

        This operation does not actually read any data; it merely returns a virtual view
        that reflects the result of the indexing operation.  To read data, call
        :py:obj:`.read` on the returned view.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> view = dataset[[5, 10, 20], 6:10]
            >>> view
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                'chunks': [70, 80],
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'dimension_separator': '.',
                'dtype': '<u4',
                'fill_value': None,
                'filters': None,
                'order': 'C',
                'shape': [70, 80],
                'zarr_format': 2,
              },
              'transform': {
                'input_exclusive_max': [3, 10],
                'input_inclusive_min': [0, 6],
                'output': [{'index_array': [[5], [10], [20]]}, {'input_dimension': 1}],
              },
            })

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`TensorStore.oindex`
           - :py:obj:`TensorStore.vindex`

        Group:
          Indexing

        Overload:
          indices
        """

    def __repr__(self) -> str: ...

    @typing.overload
    def __setitem__(
        self, transform: IndexTransform, source: TensorStore | numpy.typing.ArrayLike
    ) -> None:
        """
        Synchronously writes using an explicit :ref:`index transform<index-transform>`.

        This allows Python subscript assignment syntax to be used as a shorthand for
        :python:`self[transform].write(source).result()`.

        Example:

            >>> dataset = ts.open({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   shape=[70, 80],
            ...                   create=True).result()
            >>> transform = ts.IndexTransform(
            ...     input_shape=[3],
            ...     output=[
            ...         ts.OutputIndexMap(index_array=[1, 2, 3]),
            ...         ts.OutputIndexMap(index_array=[5, 4, 3])
            ...     ])
            >>> dataset[transform] = [1, 2, 3]
            >>> dataset[1:6, 1:6].read().result()
            array([[0, 0, 0, 0, 1],
                   [0, 0, 0, 2, 0],
                   [0, 0, 3, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=uint32)

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.
          source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
            :python:`self.domain[transform]` and with a data type convertible to
            :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
            :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

        .. warning::

           When *not* using a transaction, the subscript assignment syntax always blocks
           synchronously on the completion of the write operation.  When performing
           multiple, fine-grained writes, it is recommended to either use a transaction
           or use the asynchronous :py:obj:`TensorStore.write` interface directly.

        Group:
          I/O

        Overload:
          transform

        See also:

           - :ref:`index-transform`
           - :py:obj:`TensorStore.write`
           - :py:obj:`TensorStore.__setitem__(indices)`
           - :py:obj:`TensorStore.__setitem__(domain)`
           - :py:obj:`TensorStore.__setitem__(expr)`
           - :py:obj:`TensorStore.vindex.__setitem__`
           - :py:obj:`TensorStore.oindex.__setitem__`
        """

    @typing.overload
    def __setitem__(
        self, transform: IndexDomain, source: TensorStore | numpy.typing.ArrayLike
    ) -> None:
        """
        Synchronously writes using an explicit :ref:`index domain<index-domain>`.

        Example:

            >>> dataset = ts.open({
            ...     'driver': 'n5',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   domain=ts.IndexDomain(shape=[60, 70, 80],
            ...                                         labels=['x', 'y', 'z']),
            ...                   create=True).result()
            >>> domain = ts.IndexDomain(labels=['x', 'z'],
            ...                         inclusive_min=[5, 6],
            ...                         exclusive_max=[8, 9])
            >>> dataset[domain] = 42
            >>> dataset[5:10, 0, 5:10].read().result()
            array([[ 0, 42, 42, 42,  0],
                   [ 0, 42, 42, 42,  0],
                   [ 0, 42, 42, 42,  0],
                   [ 0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0]], dtype=uint32)

        Args:

          transform: Index transform, :python:`transform.output_rank` must equal
            :python:`self.rank`.
          source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
            :python:`self.domain[transform]` and with a data type convertible to
            :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
            :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

        .. warning::

           When *not* using a transaction, the subscript assignment syntax always blocks
           synchronously on the completion of the write operation.  When performing
           multiple, fine-grained writes, it is recommended to either use a transaction
           or use the asynchronous :py:obj:`TensorStore.write` interface directly.

        Group:
          I/O

        Overload:
          domain

        See also:

           - :ref:`index-domain`
           - :ref:`index-domain-alignment`
           - :py:obj:`TensorStore.write`
        """

    @typing.overload
    def __setitem__(
        self, transform: DimExpression, source: TensorStore | numpy.typing.ArrayLike
    ) -> None:
        """
        Synchronously writes using a :ref:`dimension expression<python-dim-expressions>`.

        Example:

            >>> dataset = ts.open({
            ...     'driver': 'n5',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   domain=ts.IndexDomain(shape=[60, 70, 80],
            ...                                         labels=['x', 'y', 'z']),
            ...                   create=True).result()
            >>> dataset[ts.d['x', 'z'][5:10, 6:9]] = [1, 2, 3]
            >>> dataset[5:10, 0, 5:10].read().result()
            array([[0, 1, 2, 3, 0],
                   [0, 1, 2, 3, 0],
                   [0, 1, 2, 3, 0],
                   [0, 1, 2, 3, 0],
                   [0, 1, 2, 3, 0]], dtype=uint32)

        .. warning::

           When *not* using a transaction, the subscript assignment syntax always blocks
           synchronously on the completion of the write operation.  When performing
           multiple, fine-grained writes, it is recommended to either use a transaction
           or use the asynchronous :py:obj:`TensorStore.write` interface directly.

        Group:
          I/O

        Overload:
          expr

        See also:

           - :ref:`python-dim-expressions`
           - :py:obj:`TensorStore.write`
           - :py:obj:`TensorStore.__getitem__(expr)`
        """

    @typing.overload
    def __setitem__(
        self, indices: NumpyIndexingSpec, source: TensorStore | numpy.typing.ArrayLike
    ) -> None:
        """
        Synchronously writes using :ref:`NumPy-style indexing<python-numpy-style-indexing>` with default index array semantics.

        This allows Python subscript assignment syntax to be used as a shorthand for
        :python:`self[indices].write(source).result()`.

        Example:

            >>> dataset = ts.open({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   shape=[70, 80],
            ...                   create=True).result()
            >>> dataset[5:10, 6:8] = [1, 2]
            >>> dataset[4:10, 5:9].read().result()
            array([[0, 0, 0, 0],
                   [0, 1, 2, 0],
                   [0, 1, 2, 0],
                   [0, 1, 2, 0],
                   [0, 1, 2, 0],
                   [0, 1, 2, 0]], dtype=uint32)

        Args:
          indices: NumPy-style indexing terms.
          source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
            :python:`self[indices].domain` and with a data type convertible to
            :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
            :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

        Transactional writes are also supported:

            >>> txn = ts.Transaction()
            >>> dataset = ts.open({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   shape=[70, 80],
            ...                   create=True).result()
            >>> dataset.with_transaction(txn)[5:10, 6:8] = [1, 2]
            >>> txn.commit_sync()

        .. warning::

           When *not* using a transaction, the subscript assignment syntax always blocks
           synchronously on the completion of the write operation.  When performing
           multiple, fine-grained writes, it is recommended to either use a transaction
           or use the asynchronous :py:obj:`TensorStore.write` interface directly.

        Group:
          I/O

        Overload:
          indices

        See also:

           - :ref:`python-numpy-style-indexing`
           - :py:obj:`TensorStore.write`
           - :py:obj:`TensorStore.oindex.__setitem__`
           - :py:obj:`TensorStore.vindex.__setitem__`
        """

    def astype(self, dtype: DTypeLike) -> TensorStore:
        """
        Returns a read/write view as the specified data type.

        Example:

          >>> store = ts.array([1, 2, 3], dtype=ts.uint32)
          >>> store.astype(ts.string)
          TensorStore({
            'base': {'array': [1, 2, 3], 'driver': 'array', 'dtype': 'uint32'},
            'context': {'data_copy_concurrency': {}},
            'driver': 'cast',
            'dtype': 'string',
            'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
          })

        Group:
          Data type
        """

    def read(
        self, *, order: typing.Literal["C", "F"] = "C", batch: Batch | None = None
    ) -> Future[numpy.ndarray]:
        """
        Reads the data within the current domain.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> await dataset[5:10, 8:12].read()
            array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]], dtype=uint32)

        .. tip::

           Depending on the cache behavior of the driver, the read may be satisfied by
           the cache and not require any I/O.

        When *not* using a :py:obj:`.transaction`, the
        read result only reflects committed data; the result never includes uncommitted
        writes.

        When using a transaction, the read result reflects all writes completed (but not
        yet committed) to the transaction.

        Args:
          order: Contiguous layout order of the returned array:

            :python:`'C'`
              Specifies C order, i.e. lexicographic/row-major order.

            :python:`'F'`
              Specifies Fortran order, i.e. colexicographic/column-major order.

          batch: Batch to use for the read operation.

            .. warning::

               If specified, the returned :py:obj:`Future` will not, in general, become
               ready until the batch is submitted.  Therefore, immediately awaiting the
               returned future will lead to deadlock.

        Returns:
          A future representing the asynchronous read result.

        .. tip::

           Synchronous reads (blocking the current thread) may be performed by calling
           :py:obj:`Future.result` on the returned future:

           >>> dataset[5:10, 8:12].read().result()
           array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=uint32)

        See also:

          - :py:obj:`.__array__`

        Group:
          I/O
        """

    def resize(
        self,
        inclusive_min: collections.abc.Iterable[int | None] | None = None,
        exclusive_max: collections.abc.Iterable[int | None] | None = None,
        resize_metadata_only: bool = False,
        resize_tied_bounds: bool = False,
        expand_only: bool = False,
        shrink_only: bool = False,
    ) -> Future[TensorStore]:
        """
        Resizes the current domain, persistently modifying the stored representation.

        Depending on the :py:param`resize_metadata_only`, if the bounds are shrunk,
        existing elements outside of the new bounds may be deleted. If the bounds are
        expanded, elements outside the existing bounds will initially contain either the
        fill value, or existing out-of-bounds data remaining after a prior resize
        operation.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[3, 3],
            ...     create=True)
            >>> await dataset.write(np.arange(9, dtype=np.uint32).reshape((3, 3)))
            >>> dataset = await dataset.resize(exclusive_max=(3, 2))
            >>> await dataset.read()
            array([[0, 1],
                   [3, 4],
                   [6, 7]], dtype=uint32)

        Args:

          inclusive_min: Sequence of length :python:`self.rank()` specifying the new
            inclusive min bounds.  A bound of :python:`None` indicates no change.
          exclusive_max: Sequence of length :python:`self.rank()` specifying the new
            exclusive max bounds.  A bound of :python:`None` indicates no change.
          resize_metadata_only: Requests that, if applicable, the resize operation
            affect only the metadata but not delete data chunks that are outside of the
            new bounds.
          resize_tied_bounds: Requests that the resize be permitted even if other
            bounds tied to the specified bounds must also be resized.  This option
            should be used with caution.
          expand_only: Fail if any bounds would be reduced.
          shrink_only: Fail if any bounds would be increased.

        Returns:

          Future that resolves to a copy of :python:`self` with the updated bounds, once
          the resize operation completes.

        Group:
          I/O
        """

    def resolve(
        self, fix_resizable_bounds: bool = False, batch: Batch | None = None
    ) -> Future[TensorStore]:
        """
        Obtains updated bounds, subject to the cache policy.

        Args:

          fix_resizable_bounds: Mark all resizable bounds as explicit.

          batch: Batch to use for resolving the bounds.

            .. warning::

               If specified, the returned :py:obj:`Future` will not, in general, become
               ready until the batch is submitted.  Therefore, immediately awaiting the
               returned future will lead to deadlock.

        Group:
          I/O
        """

    def spec(
        self,
        *,
        open_mode: OpenMode | None = None,
        open: bool | None = None,
        create: bool | None = None,
        delete_existing: bool | None = None,
        assume_metadata: bool | None = None,
        assume_cached_metadata: bool | None = None,
        minimal_spec: bool | None = None,
        retain_context: bool | None = None,
        unbind_context: bool | None = None,
        recheck_cached_metadata: RecheckCacheOption | None = None,
        recheck_cached_data: RecheckCacheOption | None = None,
        recheck_cached: RecheckCacheOption | None = None,
    ) -> Spec:
        """
        Spec that may be used to re-open or re-create the TensorStore.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> dataset.spec()
            Spec({
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'metadata': {
                'chunks': [70, 80],
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'dimension_separator': '.',
                'dtype': '<u4',
                'fill_value': None,
                'filters': None,
                'order': 'C',
                'shape': [70, 80],
                'zarr_format': 2,
              },
              'transform': {
                'input_exclusive_max': [[70], [80]],
                'input_inclusive_min': [0, 0],
              },
            })
            >>> dataset.spec(minimal_spec=True)
            Spec({
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [[70], [80]],
                'input_inclusive_min': [0, 0],
              },
            })
            >>> dataset.spec(minimal_spec=True, unbind_context=True)
            Spec({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory'},
              'transform': {
                'input_exclusive_max': [[70], [80]],
                'input_inclusive_min': [0, 0],
              },
            })

        If neither :py:param:`.retain_context` nor :py:param:`.unbind_context` is
        specified, the returned :py:obj:`~tensorstore.Spec` does not include any context
        resources, equivalent to specifying
        :py:param:`tensorstore.Spec.update.strip_context`.

        Args:

          open_mode: Overrides the existing open mode.
          open: Allow opening an existing TensorStore.  Overrides the existing open mode.
          create: Allow creating a new TensorStore.  Overrides the existing open mode.  To open or
            create, specify :python:`create=True` and :python:`open=True`.
          delete_existing: Delete any existing data before creating a new array.  Overrides the existing
            open mode.  Must be specified in conjunction with :python:`create=True`.
          assume_metadata: Neither read nor write stored metadata.  Instead, just assume any necessary
            metadata based on constraints in the spec, using the same defaults for any
            unspecified metadata as when creating a new TensorStore.  The stored metadata
            need not even exist.  Operations such as resizing that modify the stored
            metadata are not supported.  Overrides the existing open mode.  Requires that
            :py:param:`.open` is `True` and :py:param:`.delete_existing` is `False`.  This
            option takes precedence over `.assume_cached_metadata` if that option is also
            specified.

            .. warning::

               This option can lead to data corruption if the assumed metadata does
               not match the stored metadata, or multiple concurrent writers use
               different assumed metadata.

            .. seealso:

               - :ref:`python-open-assume-metadata`
          assume_cached_metadata: Skip reading the metadata when opening.  Instead, just assume any necessary
            metadata based on constraints in the spec, using the same defaults for any
            unspecified metadata as when creating a new TensorStore.  The stored metadata
            may still be accessed by subsequent operations that need to re-validate or
            modify the metadata.  Requires that :py:param:`.open` is `True` and
            :py:param:`.delete_existing` is `False`.  The :py:param:`.assume_metadata`
            option takes precedence if also specified.

            .. warning::

               This option can lead to data corruption if the assumed metadata does
               not match the stored metadata, or multiple concurrent writers use
               different assumed metadata.

            .. seealso:

               - :ref:`python-open-assume-metadata`
          minimal_spec: Indicates whether to include in the :py:obj:`~tensorstore.Spec` returned by
            :py:obj:`tensorstore.TensorStore.spec` the metadata necessary to re-create the
            :py:obj:`~tensorstore.TensorStore`. By default, the returned
            :py:obj:`~tensorstore.Spec` includes the full metadata, but it is skipped if
            :py:param:`.minimal_spec` is set to :python:`True`.

            When applied to an existing :py:obj:`~tensorstore.Spec` via
            :py:obj:`tensorstore.open` or :py:obj:`tensorstore.Spec.update`, only ``False``
            has any effect.
          retain_context: Retain all bound context resources (e.g. specific concurrency pools, specific
            cache pools).

            The resultant :py:obj:`~tensorstore.Spec` may be used to re-open the
            :py:obj:`~tensorstore.TensorStore` using the identical context resources.

            Specifying a value of :python:`False` has no effect.
          unbind_context: Convert any bound context resources to context resource specs that fully capture
            the graph of shared context resources and interdependencies.

            Re-binding/re-opening the resultant spec will result in a new graph of new
            context resources that is isomorphic to the original graph of context resources.
            The resultant spec will not refer to any external context resources;
            consequently, binding it to any specific context will have the same effect as
            binding it to a default context.

            Specifying a value of :python:`False` has no effect.
          recheck_cached_metadata: Time after which cached metadata is assumed to be fresh. Cached metadata older
            than the specified time is revalidated prior to use. The metadata is used to
            check the bounds of every read or write operation.

            Specifying ``True`` means that the metadata will be revalidated prior to every
            read or write operation. With the default value of ``"open"``, any cached
            metadata is revalidated when the TensorStore is opened but is not rechecked for
            each read or write operation.
          recheck_cached_data: Time after which cached data is assumed to be fresh. Cached data older than the
            specified time is revalidated prior to being returned from a read operation.
            Partial chunk writes are always consistent regardless of the value of this
            option.

            The default value of ``True`` means that cached data is revalidated on every
            read. To enable in-memory data caching, you must both specify a
            :json:schema:`~Context.cache_pool` with a non-zero
            :json:schema:`~Context.cache_pool.total_bytes_limit` and also specify ``False``,
            ``"open"``, or an explicit time bound for :py:param:`.recheck_cached_data`.
          recheck_cached: Sets both :py:param:`.recheck_cached_data` and
            :py:param:`.recheck_cached_metadata`.


        Group:
          Accessors
        """

    def storage_statistics(
        self, *, query_not_stored: bool = False, query_fully_stored: bool = False
    ) -> Future[TensorStore.StorageStatistics]:
        """
        Obtains statistics of the data stored for the :py:obj:`.domain`.

        Only the specific information indicated by the parameters will be returned.  If
        no query options are specified, no information will be computed.

        Example:

            >>> store = await ts.open({
            ...     "driver": "zarr",
            ...     "kvstore": "memory://"
            ... },
            ...                       shape=(100, 200),
            ...                       dtype=ts.uint32,
            ...                       create=True)
            >>> await store.storage_statistics(query_not_stored=True)
            TensorStore.StorageStatistics(not_stored=True, fully_stored=None)
            >>> await store[10:20, 30:40].write(5)
            >>> await store.storage_statistics(query_not_stored=True)
            TensorStore.StorageStatistics(not_stored=False, fully_stored=None)
            >>> await store.storage_statistics(query_not_stored=True,
            ...                                query_fully_stored=True)
            TensorStore.StorageStatistics(not_stored=False, fully_stored=True)
            >>> await store[10:20, 30:40].storage_statistics(query_fully_stored=True)
            TensorStore.StorageStatistics(not_stored=None, fully_stored=True)

        Args:

          query_not_stored: Check whether there is data stored for *any* element of the
            :py:obj:`.domain`.

          query_fully_stored: Check whether there is data stored for *all* elements of
            the :py:obj:`.domain`.

            .. warning::

                 Enabling this option may significantly increase the cost of the
                 :py:obj:`.storage_statistics` query.

        Returns:
          The requested statistics.

        Raises:
          NotImplementedError: If the :ref:`driver<tensorstore-drivers>` does not
            support this operation.

        Group:
          I/O
        """

    def transpose(self, axes: DimSelectionLike | None = None) -> TensorStore:
        """
        Returns a view with a transposed domain.

        This is equivalent to :python:`self[ts.d[axes].transpose[:]]`.

        Args:

          axes: Specifies the existing dimension corresponding to each dimension of the
            new view.  Dimensions may be specified either by index or label.  Specifying
            `None` is equivalent to specifying :python:`[rank-1, ..., 0]`, which
            reverses the dimension order.

        Raises:

          ValueError: If :py:param:`.axes` does not specify a valid permutation.

        See also:
          - `tensorstore.DimExpression.transpose`
          - :py:obj:`.T`

        Group:
          Indexing
        """

    def with_transaction(self, transaction: Transaction | None) -> TensorStore:
        """
        Returns a transaction-bound view of this TensorStore.

        The returned view may be used to perform transactional read/write operations.

        Group:
          Transactions
        """

    def write(
        self,
        source: TensorStore | numpy.typing.ArrayLike,
        *,
        batch: Batch | None = None,
        can_reference_source_data_indefinitely: bool | None = None,
    ) -> WriteFutures:
        """
        Writes to the current domain.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> await dataset[5:10, 6:8].write(42)
            >>> await dataset[0:10, 0:10].read()
            array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0],
                   [ 0,  0,  0,  0,  0,  0, 42, 42,  0,  0]], dtype=uint32)
            >>> await dataset[5:10, 6:8].write([1, 2])
            >>> await dataset[5:10, 6:8].read()
            array([[1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2]], dtype=uint32)

        Args:

          source: Source array, :ref:`broadcast-compatible<index-domain-alignment>` with
            :python:`self.domain` and with a data type convertible to
            :python:`self.dtype`.  May be an existing :py:obj:`TensorStore` or any
            :py:obj:`~numpy.typing.ArrayLike`, including a scalar.

          batch: Batch to use for reading any metadata required for opening.

            .. warning::

               If specified, the returned :py:obj:`Future` will not, in general, become
               ready until the batch is submitted.  Therefore, immediately awaiting the
               returned future will lead to deadlock.
          can_reference_source_data_indefinitely: References to the source data may be retained indefinitely, even after the write
            is committed.  The source data must not be modified until all references are
            released.


        Returns:

          Future representing the asynchronous result of the write operation.

        Logically there are two steps to the write operation:

        1. reading/copying from the :python:`source`, and

        2. waiting for the write to be committed, such that it will be reflected in
           subsequent reads.

        The completion of these two steps can be tracked separately using the returned
        :py:obj:`WriteFutures.copy` and :py:obj:`WriteFutures.commit` futures,
        respectively:

        Waiting on the returned `WriteFutures` object itself waits for
        the entire write operation to complete, and is equivalent to waiting on the
        :py:obj:`WriteFutures.commit` future.  The returned
        :py:obj:`WriteFutures.copy` future becomes ready once the data has been fully
        read from :python:`source`.  After this point, :python:`source` may be safely
        modified without affecting the write operation.

        .. warning::

           You must either synchronously or asynchronously wait on the returned future
           in order to ensure the write actually completes.  If all references to the
           future are dropped without waiting on it, the write may be cancelled.

        Group:
          I/O

        Non-transactional semantics
        ---------------------------

        When *not* using a :py:obj:`Transaction`, the returned `WriteFutures.commit`
        future becomes ready only once the data has been durably committed by the
        underlying storage layer.  The precise durability guarantees depend on the
        driver, but for example:

        - when using the :ref:`kvstore/file`, the data is only considered
          committed once the ``fsync`` system call completes, which should normally
          guarantee that it will survive a system crash;

        - when using the :ref:`kvstore/gcs`, the data is only considered
          committed once the write is acknowledged and durability is guaranteed by
          Google Cloud Storage.

        Because committing a write often has significant latency, it is advantageous to
        issue multiple writes concurrently and then wait on all of them jointly:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> await asyncio.wait([
            ...     asyncio.ensure_future(dataset[i * 5].write(i)) for i in range(10)
            ... ])

        This can also be accomplished with synchronous blocking:

            >>> dataset = ts.open({
            ...     'driver': 'zarr',
            ...     'kvstore': {
            ...         'driver': 'memory'
            ...     }
            ... },
            ...                   dtype=ts.uint32,
            ...                   shape=[70, 80],
            ...                   create=True).result()
            >>> futures = [dataset[i * 5].write(i) for i in range(10)]
            >>> for f in futures:
            ...     f.result()

        Note:

          When issuing writes asynchronously, keep in mind that uncommitted writes are
          never reflected in non-transactional reads.

        For most drivers, data is written in fixed-size
        :ref:`write chunks<chunk-layout>` arranged in a regular grid.  When concurrently
        issuing multiple writes that are not perfectly aligned to disjoint write chunks,
        specifying a :json:schema:`Context.cache_pool` enables writeback caching, which
        can improve efficiency by coalescing multiple writes to the same chunk.

        Alternatively, for more explicit control over writeback behavior, you can use a
        :py:obj:`Transaction`.

        Transactional semantics
        -----------------------

        Transactions provide explicit control over writeback, and allow uncommitted
        writes to be read:

            >>> txn = ts.Transaction()
            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> await dataset.with_transaction(txn)[5:10, 6:8].write([1, 2])
            >>> # Transactional read reflects uncommitted write
            >>> await dataset.with_transaction(txn)[5:10, 6:8].read()
            array([[1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2]], dtype=uint32)
            >>> # Non-transactional read does not reflect uncommitted write
            >>> await dataset[5:10, 6:8].read()
            array([[0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0],
                   [0, 0]], dtype=uint32)
            >>> await txn.commit_async()
            >>> # Now, non-transactional read reflects committed write
            >>> await dataset[5:10, 6:8].read()
            array([[1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2]], dtype=uint32)

        .. warning::

           When using a :py:obj:`Transaction`, the returned `WriteFutures.commit` future
           does *not* indicate that the data is durably committed by the underlying
           storage layer.  Instead, it merely indicates that the write will be reflected
           in any subsequent reads *using the same transaction*.  The write is only
           durably committed once the *transaction* is committed successfully.
        """

    @property
    def T(self) -> TensorStore:
        """
        View with transposed domain (reversed dimension order).

        This is equivalent to: :python:`self[ts.d[::-1].transpose[:]]`.

        See also:
          - `.transpose`
          - `tensorstore.DimExpression.transpose`

        Group:
          Indexing
        """

    @property
    def base(self) -> TensorStore | None:
        """
        Underlying `TensorStore`, if this is an adapter.

        Equal to :python:`None` if the driver is not an adapter of a single underlying
        `TensorStore`.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory',
            ...             'path': 'abc/',
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100, 200],
            ...     dtype=ts.uint32,
            ... )
            >>> downsampled = ts.downsample(store,
            ...                             downsample_factors=[2, 4],
            ...                             method="mean")
            >>> downsampled
            TensorStore({
              'base': {
                'driver': 'zarr',
                'kvstore': {'driver': 'memory', 'path': 'abc/'},
                'metadata': {
                  'chunks': [100, 200],
                  'compressor': {
                    'blocksize': 0,
                    'clevel': 5,
                    'cname': 'lz4',
                    'id': 'blosc',
                    'shuffle': -1,
                  },
                  'dimension_separator': '.',
                  'dtype': '<u4',
                  'fill_value': None,
                  'filters': None,
                  'order': 'C',
                  'shape': [100, 200],
                  'zarr_format': 2,
                },
                'transform': {
                  'input_exclusive_max': [[100], [200]],
                  'input_inclusive_min': [0, 0],
                },
              },
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'downsample_factors': [2, 4],
              'downsample_method': 'mean',
              'driver': 'downsample',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [[50], [50]],
                'input_inclusive_min': [0, 0],
              },
            })
            >>> downsampled[30:40, 20:25].base
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory', 'path': 'abc/'},
              'metadata': {
                'chunks': [100, 200],
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'dimension_separator': '.',
                'dtype': '<u4',
                'fill_value': None,
                'filters': None,
                'order': 'C',
                'shape': [100, 200],
                'zarr_format': 2,
              },
              'transform': {
                'input_exclusive_max': [80, 100],
                'input_inclusive_min': [60, 80],
              },
            })
            >>> converted = ts.cast(store, ts.float32)
            >>> converted.base
            TensorStore({
              'context': {
                'cache_pool': {},
                'data_copy_concurrency': {},
                'memory_key_value_store': {},
              },
              'driver': 'zarr',
              'dtype': 'uint32',
              'kvstore': {'driver': 'memory', 'path': 'abc/'},
              'metadata': {
                'chunks': [100, 200],
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'dimension_separator': '.',
                'dtype': '<u4',
                'fill_value': None,
                'filters': None,
                'order': 'C',
                'shape': [100, 200],
                'zarr_format': 2,
              },
              'transform': {
                'input_exclusive_max': [[100], [200]],
                'input_inclusive_min': [0, 0],
              },
            })

        Drivers that support this method include:

        - :ref:`driver/cast`
        - :ref:`driver/downsample`

        Group:
          Accessors
        """

    @property
    def chunk_layout(self) -> ChunkLayout:
        """
        :ref:`Chunk layout<chunk-layout>` of the TensorStore.

        Example:

          >>> store = await ts.open(
          ...     {
          ...         'driver': 'zarr',
          ...         'kvstore': {
          ...             'driver': 'memory'
          ...         }
          ...     },
          ...     shape=[1000, 2000, 3000],
          ...     dtype=ts.float32,
          ...     create=True)
          >>> store.chunk_layout
          ChunkLayout({
            'grid_origin': [0, 0, 0],
            'inner_order': [0, 1, 2],
            'read_chunk': {'shape': [101, 101, 101]},
            'write_chunk': {'shape': [101, 101, 101]},
          })

        Group:
          Accessors
        """

    @property
    def codec(self) -> CodecSpec | None:
        """
        Data codec spec.

        This may be used to create a new TensorStore with the same codec.

        Equal to :py:obj:`None` if the codec is unknown or not applicable.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100],
            ...     dtype=ts.uint32)
            >>> store.codec
            CodecSpec({
              'compressor': {
                'blocksize': 0,
                'clevel': 5,
                'cname': 'lz4',
                'id': 'blosc',
                'shuffle': -1,
              },
              'driver': 'zarr',
              'filters': None,
            })

        Group:
          Accessors
        """

    @property
    def dimension_units(self) -> tuple[Unit | None, ...]:
        """
        Physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity is indicated by a unit of :python:`ts.Unit(1, "")`.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100, 200],
            ...     dtype=ts.uint32,
            ...     dimension_units=['5nm', '8nm'])
            >>> store.dimension_units
            (Unit(5, "nm"), Unit(8, "nm"))

        Group:
          Accessors
        """

    @property
    def domain(self) -> IndexDomain:
        """
        Domain of the array.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     domain=ts.IndexDomain(shape=[70, 80], labels=['x', 'y']),
            ...     create=True)
            >>> dataset.domain
            { "x": [0, 70*), "y": [0, 80*) }

        The bounds of the domain reflect any transformations that have been applied:

            >>> dataset[30:50].domain
            { "x": [30, 50), "y": [0, 80*) }

        Group:
          Accessors
        """

    @property
    def dtype(self) -> dtype:
        """
        Data type of the array.

        Example:

            >>> dataset = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     dtype=ts.uint32,
            ...     shape=[70, 80],
            ...     create=True)
            >>> dataset.dtype
            dtype("uint32")

        Group:
          Data type
        """

    @property
    def fill_value(self) -> numpy.ndarray | None:
        """
        Fill value for positions not yet written.

        Equal to :py:obj:`None` if the fill value is unknown or not applicable.

        The fill value has data type equal to :python:`self.dtype` and a shape that is
        :ref:`broadcast-compatible<index-domain-alignment>` with :python:`self.shape`.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100],
            ...     dtype=ts.uint32,
            ...     fill_value=42)
            >>> store.fill_value
            array(42, dtype=uint32)

        Group:
          Accessors
        """

    @property
    def kvstore(self) -> KvStore | None:
        """
        Associated key-value store used as the underlying storage.

        Equal to :python:`None` if the driver does not use a key-value store.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory',
            ...             'path': 'abc/',
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100, 200],
            ...     dtype=ts.uint32,
            ... )
            >>> store.kvstore
            KvStore({'context': {'memory_key_value_store': {}}, 'driver': 'memory', 'path': 'abc/'})

        Group:
          Accessors
        """

    @property
    def label(self) -> TensorStore._Label: ...

    @property
    def mark_bounds_implicit(self) -> TensorStore._MarkBoundsImplicit: ...

    @property
    def mode(self) -> str:
        """
        Read/write mode.

        Returns:

          :python:`'r'`
            read only

          :python:`'w'`
            write only

          :python:`'rw'`
            read-write

        Group:
          Accessors
        """

    @property
    def ndim(self) -> int:
        """
        Alias for :py:obj:`.rank`.

        Example:

            >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
            >>> dataset.ndim
            2

        Group:
          Accessors
        """

    @property
    def oindex(self) -> TensorStore._Oindex: ...

    @property
    def origin(self) -> tuple[int, ...]:
        """
        Inclusive lower bound of the domain.

        This is equivalent to :python:`self.domain.origin`.

        Group:
          Accessors
        """

    @property
    def rank(self) -> int:
        """
        Number of dimensions in the domain.

        This is equivalent to :python:`self.domain.rank`.

        Example:

            >>> dataset = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.int32)
            >>> dataset.rank
            2

        Group:
          Accessors
        """

    @property
    def readable(self) -> bool:
        """
        Indicates if reading is supported.

        Group:
          Accessors
        """

    @property
    def schema(self) -> Schema:
        """
        :ref:`Schema<schema>` of this TensorStore.

        This schema may be used to create a new TensorStore with the same schema, but
        possibly using a different driver, storage location, etc.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'zarr',
            ...         'kvstore': {
            ...             'driver': 'memory'
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100],
            ...     dtype=ts.uint32,
            ...     fill_value=42)
            >>> store.schema
            Schema({
              'chunk_layout': {
                'grid_origin': [0],
                'inner_order': [0],
                'read_chunk': {'shape': [100]},
                'write_chunk': {'shape': [100]},
              },
              'codec': {
                'compressor': {
                  'blocksize': 0,
                  'clevel': 5,
                  'cname': 'lz4',
                  'id': 'blosc',
                  'shuffle': -1,
                },
                'driver': 'zarr',
                'filters': None,
              },
              'domain': {'exclusive_max': [[100]], 'inclusive_min': [0]},
              'dtype': 'uint32',
              'fill_value': 42,
              'rank': 1,
            })

        .. note:

           Each access to this property results in a new copy of the schema.  Modifying
           that copying by calling `Schema.update` does not affect this TensorStore.

        Group:
          Accessors
        """

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Shape of the domain.

        This is equivalent to :python:`self.domain.shape`.

        Group:
          Accessors
        """

    @property
    def size(self) -> int:
        """
        Total number of elements in the domain.

        This is equivalent to :python:`self.domain.size`.

        Group:
          Accessors
        """

    @property
    def transaction(self) -> Transaction | None:
        """
        Associated transaction used for read/write operations.

        Group:
          Transactions
        """

    @property
    def translate_backward_by(self) -> TensorStore._TranslateBackwardBy: ...

    @property
    def translate_by(self) -> TensorStore._TranslateBy: ...

    @property
    def translate_to(self) -> TensorStore._TranslateTo: ...

    @property
    def url(self) -> str:
        """
        :json:schema:`URL representation<TensorStoreUrl>` of the TensorStore specification.

        Example:

            >>> store = await ts.open(
            ...     {
            ...         'driver': 'n5',
            ...         'kvstore': {
            ...             'driver': 'memory',
            ...             'path': 'abc/',
            ...         }
            ...     },
            ...     create=True,
            ...     shape=[100, 200],
            ...     dtype=ts.uint32,
            ... )
            >>> store.url
            'memory://abc/|n5:'

        Group:
          Accessors
        """

    @property
    def vindex(self) -> TensorStore._Vindex: ...

    @property
    def writable(self) -> bool:
        """
        Indicates if writing is supported.

        Group:
          Accessors
        """


class Transaction:
    """


    Transactions are used to stage a group of modifications (e.g. writes to
    :py:obj:`tensorstore.TensorStore` objects) in memory, and then either commit the
    group all at once or abort it.

    Two transaction modes are currently supported:

    "Isolated" transactions provide write isolation: no modifications made are
    visible or persist outside the transactions until the transaction is committed.
    In addition to allowing modifications to be aborted/rolled back, this can also
    improve efficiency by ensuring multiple writes to the same underlying storage
    key are coalesced.

    "Atomic isolated" transactions have all the properties of "isolated"
    transactions but additionally guarantee that all of the modifications will be
    committed atomically, i.e. at no point will an external reader observe only some
    but not all of the modifications.  If the modifications made in the transaction
    cannot be committed atomically, the transaction will fail (without any changes
    being made).

    Example usage:

        >>> txn = ts.Transaction()
        >>> store = ts.open({
        ...     'driver': 'n5',
        ...     'kvstore': {
        ...         'driver': 'file',
        ...         'path': 'tmp/dataset/'
        ...     },
        ...     'metadata': {
        ...         'dataType': 'uint16',
        ...         'blockSize': [2, 3],
        ...         'dimensions': [5, 6],
        ...         'compression': {
        ...             'type': 'raw'
        ...         }
        ...     },
        ...     'create': True,
        ...     'delete_existing': True
        ... }).result()
        >>> store.with_transaction(txn)[1:4, 2:5] = 42
        >>> store.with_transaction(txn)[0:2, 4] = 43

    Uncommitted changes made in a transaction are visible from a transactional read
    using the same transaction, but not from a non-transactional read:

        >>> store.with_transaction(txn).read().result()
        array([[ 0,  0,  0,  0, 43,  0],
               [ 0,  0, 42, 42, 43,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0,  0,  0,  0,  0]], dtype=uint16)
        >>> store.read().result()
        array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]], dtype=uint16)

    The transaction can be committed using :py:meth:`.commit_async`.

        >>> txn.commit_async().result()
        >>> store.read().result()
        array([[ 0,  0,  0,  0, 43,  0],
               [ 0,  0, 42, 42, 43,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

    The :py:obj:`tensorstore.Transaction` class can also be used as a regular or
    asynchronous context manager:

        >>> with ts.Transaction() as txn:
        ...     store.with_transaction(txn)[0:2, 1:3] = 44
        ...     store.with_transaction(txn)[0, 0] = 45
        >>> store.read().result()
        array([[45, 44, 44,  0, 43,  0],
               [ 0, 44, 44, 42, 43,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

        >>> async with ts.Transaction() as txn:
        ...     store.with_transaction(txn)[0:2, 1:3] = 44
        ...     store.with_transaction(txn)[0, 0] = 45
        >>> await store.read()
        array([[45, 44, 44,  0, 43,  0],
               [ 0, 44, 44, 42, 43,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0, 42, 42, 42,  0],
               [ 0,  0,  0,  0,  0,  0]], dtype=uint16)

    If the block exits normally, the transaction is committed automatically.  If the
    block raises an exception, the transaction is aborted.

    Group:
      Core

    Constructors
    ============

    Accessors
    =========

    Operations
    ==========

    """

    def __aenter__(self) -> Future[Transaction]: ...

    def __aexit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> Future[None]: ...

    def __enter__(self) -> Transaction: ...

    def __exit__(
        self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any
    ) -> None: ...

    def __init__(self, atomic: bool = False, repeatable_read: bool = False) -> None:
        """
        Creates a new transaction.

        Write isolation is currently always implied.

        Args:
          atomic: Requires atomicity when committing.
          repeatable_read: Requires that repeated reads return the same result.
        """

    def abort(self) -> None:
        """
        Aborts the transaction.

        Has no effect if :py:meth:`.commit_async` or :py:meth:`.abort` has already been
        called.

          - :py:obj:`.commit_async`

        Group:
          Operations
        """

    def commit_async(self) -> Future[None]:
        """
        Asynchronously commits the transaction.

        Has no effect if :py:meth:`.commit_async` or :py:meth:`.abort` has already been
        called.

        Returns the associated :py:obj:`.future`, which may be used to check if the
        commit was successful.

        See also:

          - :py:obj:`.commit_sync`
          - :py:obj:`.abort`

        Group:
          Operations
        """

    def commit_sync(self) -> None:
        """
        Synchronously commits the transaction.

        Equivalent to :python:`self.commit_async().result()`.

        Returns:

           :py:obj:`None` if the commit is successful, and raises an error otherwise.

        See also:

          - :py:obj:`.commit_async`
          - :py:obj:`.abort`

        Group:
          Operations
        """

    @property
    def aborted(self) -> bool:
        """
        Indicates whether the transaction has been aborted.

        Group:
          Accessors
        """

    @property
    def atomic(self) -> bool:
        """
        Indicates whether the transaction is atomic.

        Group:
          Accessors
        """

    @property
    def commit_started(self) -> bool:
        """
        Indicates whether the commit of the transaction has already started.

        Group:
          Accessors
        """

    @property
    def future(self) -> Future[None]:
        """
        Commit result future.

        Becomes ready when the transaction has either been committed successfully or
        aborted.

        Group:
          Accessors
        """

    @property
    def open(self) -> bool:
        """
        Indicates whether the transaction is still open.

        The transaction remains open until commit starts or it is aborted.  Once commit
        starts or it has been aborted, it may not be used for any additional
        transactional operations.

        Group:
          Accessors
        """


class Unit:
    """

    Specifies a physical quantity/unit.

    The quantity is specified as the combination of:

    - A numerical :py:obj:`.multiplier`, represented as a
      `double-precision floating-point number <https://en.wikipedia.org/wiki/Double-precision_floating-point_format>`_.
      A multiplier of :python:`1` may be used to indicate a quantity equal to a
      single base unit.

    - A :py:obj:`.base_unit`, represented as a string.  An empty string may be used
      to indicate a dimensionless quantity.  In general, TensorStore does not
      interpret the base unit string; some drivers impose additional constraints on
      the base unit, while other drivers may store the specified unit directly.  It
      is recommended to follow the
      `udunits2 syntax <https://www.unidata.ucar.edu/software/udunits/udunits-2.0.4/udunits2lib.html#Syntax>`_
      unless there is a specific need to deviate.

    Objects of this type are immutable.

    Group:
      Spec
    """

    __hash__: typing.ClassVar[None] = None

    def __eq__(self, other: Unit) -> bool:
        """
        Compares two units for equality.

        Example:

          >>> ts.Unit('3nm') == ts.Unit(3, 'nm')
          >>> True
        """

    @typing.overload
    def __init__(self, multiplier: float = 1) -> None:
        """
        Constructs a dimension-less quantity of the specified value.

        This is equivalent to specifying a :py:obj:`.base_unit` of :python:`""`.

        Example:

          >>> ts.Unit(3.5)
          Unit(3.5, "")
          >>> ts.Unit()
          Unit(1, "")

        Overload:
          multiplier
        """

    @typing.overload
    def __init__(self, unit: str) -> None:
        """
        Constructs a unit from a string.

        If the string contains a leading number, it is parsed as the
        :py:obj:`.multiplier` and the remaining portion, after stripping leading and
        trailing whitespace, is used as the :py:obj:`.base_unit`.  If there is no
        leading number, the :py:obj:`.multiplier` is :python:`1` and the entire string,
        after stripping leading and trailing whitespace, is used as the
        :py:obj:`.base_unit`.

        Example:

          >>> ts.Unit('4nm')
          Unit(4, "nm")
          >>> ts.Unit('nm')
          Unit(1, "nm")
          >>> ts.Unit('3e5')
          Unit(300000, "")
          >>> ts.Unit('')
          Unit(1, "")

        Overload:
          unit
        """

    @typing.overload
    def __init__(self, multiplier: float, base_unit: str) -> None:
        """
        Constructs a unit from a multiplier and base unit.

        Example:

          >>> ts.Unit(3.5, 'nm')
          Unit(3.5, "nm")

        Overload:
          components
        """

    @typing.overload
    def __init__(self, unit: tuple[float, str]) -> None:
        """
        Constructs a unit from a multiplier and base unit pair.

        Example:

          >>> ts.Unit((3.5, 'nm'))
          Unit(3.5, "nm")

        Overload:
          pair
        """

    @typing.overload
    def __init__(self, *, json: typing.Any) -> None:
        """
        Constructs a unit from its :json:schema:`JSON representation<Unit>`.

        Example:

          >>> ts.Unit(json=[3.5, 'nm'])
          Unit(3.5, "nm")

        Overload:
          json
        """

    def __mul__(self, multiplier: float) -> Unit:
        """
        Multiplies this unit by the specified multiplier.

        Example:

          >>> ts.Unit('3.5nm') * 2
          Unit(7, "nm")

        Group:
          Arithmetic operators
        """

    def __repr__(self) -> str: ...

    def __str__(self) -> str: ...

    def __truediv__(self, divisor: float) -> Unit:
        """
        Divides this unit by the specified divisor.

        Example:

          >>> ts.Unit('7nm') / 2
          Unit(3.5, "nm")

        Group:
          Arithmetic operators
        """

    def to_json(self) -> typing.Any:
        """
        Converts to the :json:schema:`JSON representation<Unit>`.

        Example:

          >>> ts.Unit('3nm').to_json()
          [3.0, 'nm']

        Group:
          Accessors
        """

    @property
    def base_unit(self) -> str:
        """
        Base unit from which this unit is derived.

        Example:

          >>> u = ts.Unit('3.5nm')
          >>> u.base_unit
          'nm'

        Group:
          Accessors
        """

    @property
    def multiplier(self) -> float:
        """
        Multiplier for the :py:obj:`.base_unit`.

        Example:

          >>> u = ts.Unit('3.5nm')
          >>> u.multiplier
          3.5

        Group:
          Accessors
        """


class VirtualChunkedReadParameters:
    """

    Options passed to read callbacks used with :py:obj:`.virtual_chunked`.

    Group:
      Virtual views
    """

    @property
    def batch(self) -> Batch | None:
        """
        Batch associated with the read request.
        """

    @property
    def if_not_equal(self) -> bytes:
        """
        Cached generation, read request can be skipped if no newer data is available.
        """

    @property
    def staleness_bound(self) -> float:
        """
        Read may be fulfilled with cached data no older than the specified bound.
        """


class VirtualChunkedWriteParameters:
    """

    Options passed to write callbacks used with :py:obj:`.virtual_chunked`.

    Group:
      Virtual views
    """

    @property
    def if_equal(self) -> bytes:
        """
        If non-empty, writeback should be conditioned on the existing data matching the specified generation.
        """


class WriteFutures:
    """

    Handle for consuming the result of an asynchronous write operation.

    This holds two futures:

    - The :py:obj:`.copy` future indicates when reading has completed, after which
      the source is no longer accessed.

    - The :py:obj:`.commit` future indicates when the write is guaranteed to be
      reflected in subsequent reads.  For non-transactional writes, the
      :py:obj:`.commit` future completes successfully only once durability of the
      write is guaranteed (subject to the limitations of the underlying storage
      mechanism).  For transactional writes, the :py:obj:`.commit` future merely
      indicates when the write is reflected in subsequent reads using the same
      transaction.  Durability is *not* guaranteed until the transaction itself is
      committed successfully.

    In addition, this class also provides the same interface as :py:class:`Future`,
    which simply forwards to the corresponding operation on the :py:obj:`.commit`
    future.

    See also:
      - :py:meth:`TensorStore.write`

    Group:
      Asynchronous support
    """

    def __await__(self) -> typing.Any: ...

    def add_done_callback(self, callback: typing.Callable[[Future], None]) -> None: ...

    def cancel(self) -> bool: ...

    def cancelled(self) -> bool: ...

    def done(self) -> bool: ...

    def exception(
        self, timeout: float | None = None, deadline: float | None = None
    ) -> typing.Any: ...

    def remove_done_callback(
        self, callback: typing.Callable[[Future], None]
    ) -> int: ...

    def result(
        self, timeout: float | None = None, deadline: float | None = None
    ) -> typing.Any: ...

    @property
    def commit(self) -> Future[None]: ...

    @property
    def copy(self) -> Future[None]: ...


class _DimSelection:
    __iter__ = None

    @staticmethod
    def __getitem__(selection: DimSelectionLike) -> DimSelection:
        """
        Constructs a `DimSelection` from a sequence of dimension indices, ranges, and/or labels.

        Examples:

           >>> ts.d[0, 1, 2]
           d[0,1,2]
           >>> ts.d[0:1, 2, "x"]
           d[0:1,2,'x']
           >>> ts.d[[0, 1], [2]]
           d[0,1,2]
           >>> ts.d[[0, 1], ts.d[2, 3]]
           d[0,1,2,3]

        Group:
          Indexing
        """


class dtype:
    """

    TensorStore data type representation.

    Group:
      Data types
    """

    def __call__(self, arg: typing.Any) -> typing.Any:
        """
        Construct a scalar instance of this data type.

        Group:
          Conversion
        """

    def __eq__(self, other: DTypeLike) -> bool: ...

    def __hash__(self) -> int: ...

    def __init__(self, dtype: DTypeLike) -> None:
        """
        Construct by name or from an existing TensorStore or NumPy data type.
        """

    def __repr__(self) -> str: ...

    def to_json(self) -> str:
        """
        :json:schema:`JSON representation<dtype>` of the data type.

        Group:
          Accessors
        """

    @property
    def name(self) -> str:
        """
        Name of the data type.

        This is equivalent to the :json:schema:`JSON representation<dtype>`.

        Group:
          Accessors
        """

    @property
    def numpy_dtype(self) -> numpy.dtype[typing.Any]:
        """
        NumPy data type corresponding to this TensorStore data type.

        For TensorStore data types without a specific associated NumPy data type, the
        NumPy object data type ``np.dtype("O")`` is returned.

        Group:
          Accessors
        """

    @property
    def type(self) -> type:
        """
        Python type object corresponding to this TensorStore data type.

        Group:
          Accessors
        """


def array(
    array: numpy.typing.ArrayLike,
    dtype: DTypeLike | None = None,
    *,
    context: Context = None,
    copy: bool | None = None,
    write: bool | None = None,
) -> TensorStore:
    """
    Returns a TensorStore that reads/writes from an in-memory array.

    Args:
      array: Source array.
      dtype: Data type to which :python:`array` will be converted.
      context: Optional context to use, for specifying
        :json:schema:`Context.data_copy_concurrency`.
      copy: Indicates whether the returned TensorStore may be a copy of the source
        array, rather than a reference to it.

        - If `None` (default), the source array is copied only if it cannot be
          referenced.

        - If `True`, a copy is always made.

        - If `False`, a copy is never made and an error is raised if the source data
          can't be referenced.
      write: Indicates whether the returned TensorStore is writable.

        - If `None` (default), the returned TensorStore may or may not be writable,
          depending on whether the source array converts to a writable NumPy array.

        - If `True`, the returned TensorStore is required to be writable, and an
          error is returned if the source array does not support writing.

        - If `False`, the returned TensorStore is never writable, even if the source
          array is writable.

    Group:
      Views
    """


@typing.overload
def cast(base: TensorStore, dtype: DTypeLike) -> TensorStore:
    """
    Returns a read/write view with the data type converted.

    Example:

        >>> array = ts.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype=ts.float32)
        >>> view = ts.cast(array, ts.uint32)
        >>> view
        TensorStore({
          'base': {
            'array': [1.5, 2.5, 3.5, 4.5, 5.5],
            'driver': 'array',
            'dtype': 'float32',
          },
          'context': {'data_copy_concurrency': {}},
          'driver': 'cast',
          'dtype': 'uint32',
          'transform': {'input_exclusive_max': [5], 'input_inclusive_min': [0]},
        })
        >>> await view.read()
        array([1, 2, 3, 4, 5], dtype=uint32)

    Overload:
      store

    Group:
      Views
    """


@typing.overload
def cast(base: Spec, dtype: DTypeLike) -> Spec:
    """
    Returns a view with the data type converted.

    Example:

        >>> base = ts.Spec({"driver": "zarr", "kvstore": "memory://"})
        >>> view = ts.cast(base, ts.uint32)
        >>> view
        Spec({
          'base': {'driver': 'zarr', 'kvstore': {'driver': 'memory'}},
          'driver': 'cast',
          'dtype': 'uint32',
        })

    Overload:
      spec

    Group:
      Views
    """


def concat(
    layers: collections.abc.Iterable[TensorStore | Spec],
    axis: int | str,
    *,
    read: bool | None = None,
    write: bool | None = None,
    context: Context | None = None,
    transaction: Transaction | None = None,
    rank: int | None = None,
    dtype: DTypeLike | None = None,
    domain: IndexDomain | None = None,
    shape: collections.abc.Iterable[int] | None = None,
    dimension_units: (
        collections.abc.Iterable[
            Unit | str | numbers.Real | tuple[numbers.Real, str] | None
        ]
        | None
    ) = None,
    schema: Schema | None = None,
) -> TensorStore:
    """
    Virtually concatenates a sequence of :py:obj:`TensorStore` layers along an existing dimension.

        >>> store = ts.concat([
        ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
        ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
        ... ],
        ...                   axis=0)
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [1, 2, 3, 4],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {'input_exclusive_max': [4], 'input_inclusive_min': [0]},
            },
            {
              'array': [5, 6, 7, 8],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [8],
                'input_inclusive_min': [4],
                'output': [{'input_dimension': 0, 'offset': -4}],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [8], 'inclusive_min': [0]}},
          'transform': {'input_exclusive_max': [8], 'input_inclusive_min': [0]},
        })
        >>> await store.read()
        array([1, 2, 3, 4, 5, 6, 7, 8], dtype=uint32)
        >>> store = ts.concat([
        ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32),
        ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
        ... ],
        ...                   axis=0)
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [[1, 2, 3], [4, 5, 6]],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [2, 3],
                'input_inclusive_min': [0, 0],
              },
            },
            {
              'array': [[7, 8, 9], [10, 11, 12]],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [4, 3],
                'input_inclusive_min': [2, 0],
                'output': [
                  {'input_dimension': 0, 'offset': -2},
                  {'input_dimension': 1},
                ],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [4, 3], 'inclusive_min': [0, 0]}},
          'transform': {'input_exclusive_max': [4, 3], 'input_inclusive_min': [0, 0]},
        })
        >>> await store.read()
        array([[ 1,  2,  3],
               [ 4,  5,  6],
               [ 7,  8,  9],
               [10, 11, 12]], dtype=uint32)
        >>> store = ts.concat([
        ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32),
        ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
        ... ],
        ...                   axis=-1)
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [[1, 2, 3], [4, 5, 6]],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [2, 3],
                'input_inclusive_min': [0, 0],
              },
            },
            {
              'array': [[7, 8, 9], [10, 11, 12]],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [2, 6],
                'input_inclusive_min': [0, 3],
                'output': [
                  {'input_dimension': 0},
                  {'input_dimension': 1, 'offset': -3},
                ],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [2, 6], 'inclusive_min': [0, 0]}},
          'transform': {'input_exclusive_max': [2, 6], 'input_inclusive_min': [0, 0]},
        })
        >>> await store.read()
        array([[ 1,  2,  3,  7,  8,  9],
               [ 4,  5,  6, 10, 11, 12]], dtype=uint32)
        >>> await ts.concat([
        ...     ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32).label["x", "y"],
        ...     ts.array([[7, 8, 9], [10, 11, 12]], dtype=ts.uint32)
        ... ],
        ...                 axis="y").read()
        array([[ 1,  2,  3,  7,  8,  9],
               [ 4,  5,  6, 10, 11, 12]], dtype=uint32)

    Args:

      layers: Sequence of layers to concatenate.  If a layer is specified as a
        :py:obj:`Spec` rather than a :py:obj:`TensorStore`, it must have a known
        :py:obj:`~Spec.domain` and will be opened on-demand as needed for individual
        read and write operations.

      axis: Existing dimension along which to concatenate.  A negative number counts
        from the end.  May also be specified by a
        :ref:`dimension label<dimension-labels>`.
      read: Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      write: Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      context: Shared resource context.  Defaults to a new (unshared) context with default
        options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
        such as cache pools, between multiple open TensorStores, you must specify a
        context.
      transaction: Transaction to use for opening/creating, and for subsequent operations.  By
        default, the open is non-transactional.

        .. note::

           To perform transactional operations using a :py:obj:`TensorStore` that was
           previously opened without a transaction, use
           :py:obj:`TensorStore.with_transaction`.
      rank: Constrains the rank of the TensorStore.  If there is an index transform, the
        rank constraint must match the rank of the *input* space.
      dtype: Constrains the data type of the TensorStore.  If a data type has already been
        set, it is an error to specify a different data type.
      domain: Constrains the domain of the TensorStore.  If there is an existing
        domain, the specified domain is merged with it as follows:

        1. The rank must match the existing rank.

        2. All bounds must match, except that a finite or explicit bound is permitted to
           match an infinite and implicit bound, and takes precedence.

        3. If both the new and existing domain specify non-empty labels for a dimension,
           the labels must be equal.  If only one of the domains specifies a non-empty
           label for a dimension, the non-empty label takes precedence.

        Note that if there is an index transform, the domain must match the *input*
        space, not the output space.
      shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
        :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

        .. note::

           This option also constrains the origin of all dimensions to be zero.
      dimension_units: Specifies the physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity can be indicated by a unit of :python:`""`.
      schema: Additional schema constraints to merge with existing constraints.


    See also:
      - :py:obj:`numpy.concatenate`
      - :ref:`driver/stack`
      - :py:obj:`tensorstore.overlay`
      - :py:obj:`tensorstore.stack`

    Group:
      Views
    """


@typing.overload
def downsample(
    base: TensorStore,
    downsample_factors: collections.abc.Iterable[int],
    method: DownsampleMethod,
) -> TensorStore:
    """
    Returns a virtual :ref:`downsampled view<driver/downsample>` of a :py:obj:`TensorStore`.

    Group:
      Views

    Overload:
      store
    """


@typing.overload
def downsample(
    base: Spec,
    downsample_factors: collections.abc.Iterable[int],
    method: DownsampleMethod,
) -> Spec:
    """
    Returns a virtual :ref:`downsampled view<driver/downsample>` view of a :py:obj:`Spec`.

    Group:
      Views

    Overload:
      spec
    """


def experimental_collect_matching_metrics(
    metric_prefix: str = "", include_zero_metrics: bool = False
) -> list[typing.Any]:
    """
    Collects metrics with a matching prefix.

    Args:
      metric_prefix: Prefix of the metric names to collect.
      include_zero_metrics: Indicate whether zero-valued metrics are included.

    Returns:
      :py:obj:`list` of a :py:obj:`dict` of metrics.

    Group:
      Experimental
    """


def experimental_collect_prometheus_format_metrics(
    metric_prefix: str = "",
) -> list[str]:
    """
    Collects metrics in prometheus exposition format.
    See: https://prometheus.io/docs/instrumenting/exposition_formats/

    Args:
      metric_prefix: Prefix of the metric names to collect.

    Returns:
      :py:obj:`list` of a :py:obj:`str` of prometheus exposition format metrics.

    Group:
      Experimental
    """


def experimental_push_metrics_to_prometheus(
    pushgateway: str = "", job: str = "", instance: str = "", metric_prefix: str = ""
) -> Future[int]:
    """
    Publishes metrics to the prometheus pushgateway.
    See: https://github.com/prometheus/pushgateway

    Args:
      pushgateway: prometheus pushgateway url, like 'http://localhost:1234/'
      job: prometheus job name
      instance: prometheus instance identifier
      metric_prefix: Prefix of the metric names to publish.

    Returns:
      A future with the response status code.

    Group:
      Experimental
    """


def experimental_update_verbose_logging(
    flags: str = "", overwrite: bool = False
) -> None:
    """
    Updates verbose logging flags associated with --tensorstore_verbose_logging and
    TENSORSTORE_VERBOSE_LOGGING flags.

    Args:
      flags: :py:obj:`str` comma separated list of flags with optional values.
      overwrite: When true overwrites existing flags, otherwise updates.

    Group:
      Experimental
    """


def open(
    spec: Spec | typing.Any,
    *,
    read: bool | None = None,
    write: bool | None = None,
    open_mode: OpenMode | None = None,
    open: bool | None = None,
    create: bool | None = None,
    delete_existing: bool | None = None,
    assume_metadata: bool | None = None,
    assume_cached_metadata: bool | None = None,
    context: Context | None = None,
    transaction: Transaction | None = None,
    batch: Batch | None = None,
    kvstore: KvStore.Spec | KvStore | None = None,
    recheck_cached_metadata: RecheckCacheOption | None = None,
    recheck_cached_data: RecheckCacheOption | None = None,
    recheck_cached: RecheckCacheOption | None = None,
    rank: int | None = None,
    dtype: DTypeLike | None = None,
    domain: IndexDomain | None = None,
    shape: collections.abc.Iterable[int] | None = None,
    chunk_layout: ChunkLayout | None = None,
    codec: CodecSpec | None = None,
    fill_value: numpy.typing.ArrayLike | None = None,
    dimension_units: (
        collections.abc.Iterable[
            Unit | str | numbers.Real | tuple[numbers.Real, str] | None
        ]
        | None
    ) = None,
    schema: Schema | None = None,
) -> Future[TensorStore]:
    """
    Opens or creates a :py:class:`TensorStore` from a :py:class:`Spec`.

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'zarr',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         }
        ...     },
        ...     create=True,
        ...     dtype=ts.int32,
        ...     shape=[1000, 2000, 3000],
        ...     chunk_layout=ts.ChunkLayout(inner_order=[2, 1, 0]),
        ... )
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'zarr',
          'dtype': 'int32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'chunks': [101, 101, 101],
            'compressor': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'id': 'blosc',
              'shuffle': -1,
            },
            'dimension_separator': '.',
            'dtype': '<i4',
            'fill_value': None,
            'filters': None,
            'order': 'F',
            'shape': [1000, 2000, 3000],
            'zarr_format': 2,
          },
          'transform': {
            'input_exclusive_max': [[1000], [2000], [3000]],
            'input_inclusive_min': [0, 0, 0],
          },
        })

    Args:
      spec: TensorStore Spec to open.  May also be specified as
        :json:schema:`JSON<TensorStore>` or a :json:schema:`URL<TensorStoreUrl>`.

      read: Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      write: Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      open_mode: Overrides the existing open mode.
      open: Allow opening an existing TensorStore.  Overrides the existing open mode.
      create: Allow creating a new TensorStore.  Overrides the existing open mode.  To open or
        create, specify :python:`create=True` and :python:`open=True`.
      delete_existing: Delete any existing data before creating a new array.  Overrides the existing
        open mode.  Must be specified in conjunction with :python:`create=True`.
      assume_metadata: Neither read nor write stored metadata.  Instead, just assume any necessary
        metadata based on constraints in the spec, using the same defaults for any
        unspecified metadata as when creating a new TensorStore.  The stored metadata
        need not even exist.  Operations such as resizing that modify the stored
        metadata are not supported.  Overrides the existing open mode.  Requires that
        :py:param:`.open` is `True` and :py:param:`.delete_existing` is `False`.  This
        option takes precedence over `.assume_cached_metadata` if that option is also
        specified.

        .. warning::

           This option can lead to data corruption if the assumed metadata does
           not match the stored metadata, or multiple concurrent writers use
           different assumed metadata.

        .. seealso:

           - :ref:`python-open-assume-metadata`
      assume_cached_metadata: Skip reading the metadata when opening.  Instead, just assume any necessary
        metadata based on constraints in the spec, using the same defaults for any
        unspecified metadata as when creating a new TensorStore.  The stored metadata
        may still be accessed by subsequent operations that need to re-validate or
        modify the metadata.  Requires that :py:param:`.open` is `True` and
        :py:param:`.delete_existing` is `False`.  The :py:param:`.assume_metadata`
        option takes precedence if also specified.

        .. warning::

           This option can lead to data corruption if the assumed metadata does
           not match the stored metadata, or multiple concurrent writers use
           different assumed metadata.

        .. seealso:

           - :ref:`python-open-assume-metadata`
      context: Shared resource context.  Defaults to a new (unshared) context with default
        options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
        such as cache pools, between multiple open TensorStores, you must specify a
        context.
      transaction: Transaction to use for opening/creating, and for subsequent operations.  By
        default, the open is non-transactional.

        .. note::

           To perform transactional operations using a :py:obj:`TensorStore` that was
           previously opened without a transaction, use
           :py:obj:`TensorStore.with_transaction`.
      batch: Batch to use for reading any metadata required for opening.

        .. warning::

           If specified, the returned :py:obj:`Future` will not, in general, become
           ready until the batch is submitted.  Therefore, immediately awaiting the
           returned future will lead to deadlock.
      kvstore: Sets the associated key-value store used as the underlying storage.

        If the :py:obj:`~tensorstore.Spec.kvstore` has already been set, it is
        overridden.

        It is an error to specify this if the TensorStore driver does not use a
        key-value store.
      recheck_cached_metadata: Time after which cached metadata is assumed to be fresh. Cached metadata older
        than the specified time is revalidated prior to use. The metadata is used to
        check the bounds of every read or write operation.

        Specifying ``True`` means that the metadata will be revalidated prior to every
        read or write operation. With the default value of ``"open"``, any cached
        metadata is revalidated when the TensorStore is opened but is not rechecked for
        each read or write operation.
      recheck_cached_data: Time after which cached data is assumed to be fresh. Cached data older than the
        specified time is revalidated prior to being returned from a read operation.
        Partial chunk writes are always consistent regardless of the value of this
        option.

        The default value of ``True`` means that cached data is revalidated on every
        read. To enable in-memory data caching, you must both specify a
        :json:schema:`~Context.cache_pool` with a non-zero
        :json:schema:`~Context.cache_pool.total_bytes_limit` and also specify ``False``,
        ``"open"``, or an explicit time bound for :py:param:`.recheck_cached_data`.
      recheck_cached: Sets both :py:param:`.recheck_cached_data` and
        :py:param:`.recheck_cached_metadata`.
      rank: Constrains the rank of the TensorStore.  If there is an index transform, the
        rank constraint must match the rank of the *input* space.
      dtype: Constrains the data type of the TensorStore.  If a data type has already been
        set, it is an error to specify a different data type.
      domain: Constrains the domain of the TensorStore.  If there is an existing
        domain, the specified domain is merged with it as follows:

        1. The rank must match the existing rank.

        2. All bounds must match, except that a finite or explicit bound is permitted to
           match an infinite and implicit bound, and takes precedence.

        3. If both the new and existing domain specify non-empty labels for a dimension,
           the labels must be equal.  If only one of the domains specifies a non-empty
           label for a dimension, the non-empty label takes precedence.

        Note that if there is an index transform, the domain must match the *input*
        space, not the output space.
      shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
        :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

        .. note::

           This option also constrains the origin of all dimensions to be zero.
      chunk_layout: Constrains the chunk layout.  If there is an existing chunk layout constraint,
        the constraints are merged.  If the constraints are incompatible, an error
        is raised.
      codec: Constrains the codec.  If there is an existing codec constraint, the constraints
        are merged.  If the constraints are incompatible, an error is raised.
      fill_value: Specifies the fill value for positions that have not been written.

        The fill value data type must be convertible to the actual data type, and the
        shape must be :ref:`broadcast-compatible<index-domain-alignment>` with the
        domain.

        If an existing fill value has already been set as a constraint, it is an
        error to specify a different fill value (where the comparison is done after
        normalization by broadcasting).
      dimension_units: Specifies the physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity can be indicated by a unit of :python:`""`.
      schema: Additional schema constraints to merge with existing constraints.


    Examples
    ========

    Opening an existing TensorStore
    -------------------------------

    To open an existing TensorStore, you can use a *minimal* :py:class:`.Spec` that
    specifies required driver-specific options, like the storage location.
    Information that can be determined automatically from the existing metadata,
    like the data type, domain, and chunk layout, may be omitted:

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'neuroglancer_precomputed',
        ...         'kvstore': {
        ...             'driver': 'gcs',
        ...             'bucket': 'neuroglancer-janelia-flyem-hemibrain',
        ...             'path': 'v1.2/segmentation/',
        ...         },
        ...     },
        ...     read=True)
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'gcs_request_concurrency': {},
            'gcs_request_retries': {},
            'gcs_user_project': {},
          },
          'driver': 'neuroglancer_precomputed',
          'dtype': 'uint64',
          'kvstore': {
            'bucket': 'neuroglancer-janelia-flyem-hemibrain',
            'driver': 'gcs',
            'path': 'v1.2/segmentation/',
          },
          'multiscale_metadata': {'num_channels': 1, 'type': 'segmentation'},
          'scale_index': 0,
          'scale_metadata': {
            'chunk_size': [64, 64, 64],
            'compressed_segmentation_block_size': [8, 8, 8],
            'encoding': 'compressed_segmentation',
            'key': '8.0x8.0x8.0',
            'resolution': [8.0, 8.0, 8.0],
            'sharding': {
              '@type': 'neuroglancer_uint64_sharded_v1',
              'data_encoding': 'gzip',
              'hash': 'identity',
              'minishard_bits': 6,
              'minishard_index_encoding': 'gzip',
              'preshift_bits': 9,
              'shard_bits': 15,
            },
            'size': [34432, 39552, 41408],
            'voxel_offset': [0, 0, 0],
          },
          'transform': {
            'input_exclusive_max': [34432, 39552, 41408, 1],
            'input_inclusive_min': [0, 0, 0, 0],
            'input_labels': ['x', 'y', 'z', 'channel'],
          },
        })

    Opening by URL
    --------------

    The same TensorStore opened in the previous section can be specified more concisely using a
    :json:schema:`TensorStore URL<TensorStoreUrl>`:

        >>> store = await ts.open(
        ...     'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/|neuroglancer-precomputed:',
        ...     read=True)

    .. note::

       The URL syntax is very limited in the options and parameters that may be
       specified but is convenient in simple cases.

    Opening with format auto-detection
    ----------------------------------

    Many formats can be :ref:`auto-detected<driver/auto>` from a
    :json:schema:`KvStore URL<KvStoreUrl>` alone:

        >>> store = await ts.open(
        ...     'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/',
        ...     read=True)
        >>> store.url
        'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/|neuroglancer-precomputed:'

    A full :json:schema:`KvStore JSON spec<KvStore>` can also be specified instead of a URL:


        >>> store = await ts.open(
        ...     {
        ...         'driver': 'gcs',
        ...         'bucket': 'neuroglancer-janelia-flyem-hemibrain',
        ...         'path': 'v1.2/segmentation/'
        ...     },
        ...     read=True)
        >>> store.url
        'gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation/|neuroglancer-precomputed:'

    Creating a new TensorStore
    --------------------------

    To create a new TensorStore, you must specify required driver-specific options,
    like the storage location, as well as :py:class:`Schema` constraints like the
    data type and domain.  Suitable defaults are chosen automatically for schema
    properties that are left unconstrained:

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'zarr',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         },
        ...     },
        ...     create=True,
        ...     dtype=ts.float32,
        ...     shape=[1000, 2000, 3000],
        ...     fill_value=42)
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'zarr',
          'dtype': 'float32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'chunks': [101, 101, 101],
            'compressor': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'id': 'blosc',
              'shuffle': -1,
            },
            'dimension_separator': '.',
            'dtype': '<f4',
            'fill_value': 42.0,
            'filters': None,
            'order': 'C',
            'shape': [1000, 2000, 3000],
            'zarr_format': 2,
          },
          'transform': {
            'input_exclusive_max': [[1000], [2000], [3000]],
            'input_inclusive_min': [0, 0, 0],
          },
        })

    Partial constraints may be specified on the chunk layout, and the driver will
    determine a matching chunk layout automatically:

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'zarr',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         },
        ...     },
        ...     create=True,
        ...     dtype=ts.float32,
        ...     shape=[1000, 2000, 3000],
        ...     chunk_layout=ts.ChunkLayout(
        ...         chunk_shape=[10, None, None],
        ...         chunk_aspect_ratio=[None, 2, 1],
        ...         chunk_elements=10000000,
        ...     ),
        ... )
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'zarr',
          'dtype': 'float32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'chunks': [10, 1414, 707],
            'compressor': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'id': 'blosc',
              'shuffle': -1,
            },
            'dimension_separator': '.',
            'dtype': '<f4',
            'fill_value': None,
            'filters': None,
            'order': 'C',
            'shape': [1000, 2000, 3000],
            'zarr_format': 2,
          },
          'transform': {
            'input_exclusive_max': [[1000], [2000], [3000]],
            'input_inclusive_min': [0, 0, 0],
          },
        })

    The schema constraints allow key storage characteristics to be specified
    independent of the driver/format:

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'n5',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         },
        ...     },
        ...     create=True,
        ...     dtype=ts.float32,
        ...     shape=[1000, 2000, 3000],
        ...     chunk_layout=ts.ChunkLayout(
        ...         chunk_shape=[10, None, None],
        ...         chunk_aspect_ratio=[None, 2, 1],
        ...         chunk_elements=10000000,
        ...     ),
        ... )
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'n5',
          'dtype': 'float32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'blockSize': [10, 1414, 707],
            'compression': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'shuffle': 1,
              'type': 'blosc',
            },
            'dataType': 'float32',
            'dimensions': [1000, 2000, 3000],
          },
          'transform': {
            'input_exclusive_max': [[1000], [2000], [3000]],
            'input_inclusive_min': [0, 0, 0],
          },
        })

    Driver-specific constraints can be used in combination with, or instead of,
    schema constraints:

        >>> store = await ts.open(
        ...     {
        ...         'driver': 'zarr',
        ...         'kvstore': {
        ...             'driver': 'memory'
        ...         },
        ...         'metadata': {
        ...             'dtype': '>f4'
        ...         },
        ...     },
        ...     create=True,
        ...     shape=[1000, 2000, 3000])
        >>> store
        TensorStore({
          'context': {
            'cache_pool': {},
            'data_copy_concurrency': {},
            'memory_key_value_store': {},
          },
          'driver': 'zarr',
          'dtype': 'float32',
          'kvstore': {'driver': 'memory'},
          'metadata': {
            'chunks': [101, 101, 101],
            'compressor': {
              'blocksize': 0,
              'clevel': 5,
              'cname': 'lz4',
              'id': 'blosc',
              'shuffle': -1,
            },
            'dimension_separator': '.',
            'dtype': '>f4',
            'fill_value': None,
            'filters': None,
            'order': 'C',
            'shape': [1000, 2000, 3000],
            'zarr_format': 2,
          },
          'transform': {
            'input_exclusive_max': [[1000], [2000], [3000]],
            'input_inclusive_min': [0, 0, 0],
          },
        })

    .. _python-open-assume-metadata:

    Using :py:param:`.assume_metadata` for improved concurrent open efficiency
    --------------------------------------------------------------------------

    Normally, when opening or creating a chunked format like
    :ref:`zarr<driver/zarr2>`, TensorStore first attempts to read the existing
    metadata (and confirms that it matches any specified constraints), or (if
    creating is allowed) creates a new metadata file based on any specified
    constraints.

    When the same TensorStore stored on a distributed filesystem or cloud storage is
    opened concurrently from many machines, the simultaneous requests to read and
    write the metadata file by every machine can create contention and result in
    high latency on some distributed filesystems.

    The :py:param:`.assume_metadata` open mode allows redundant reading and writing
    of the metadata file to be avoided, but requires careful use to avoid data
    corruption.

    .. admonition:: Example of skipping reading the metadata when opening an existing array
       :class: example

       >>> context = ts.Context()
       >>> # First create the array normally
       >>> store = await ts.open({
       ...     "driver": "zarr",
       ...     "kvstore": "memory://"
       ... },
       ...                       context=context,
       ...                       dtype=ts.float32,
       ...                       shape=[5],
       ...                       create=True)
       >>> # Note that the .zarray metadata has been written.
       >>> await store.kvstore.list()
       [b'.zarray']
       >>> await store.write([1, 2, 3, 4, 5])
       >>> spec = store.spec()
       >>> spec
       Spec({
         'driver': 'zarr',
         'dtype': 'float32',
         'kvstore': {'driver': 'memory'},
         'metadata': {
           'chunks': [5],
           'compressor': {
             'blocksize': 0,
             'clevel': 5,
             'cname': 'lz4',
             'id': 'blosc',
             'shuffle': -1,
           },
           'dimension_separator': '.',
           'dtype': '<f4',
           'fill_value': None,
           'filters': None,
           'order': 'C',
           'shape': [5],
           'zarr_format': 2,
         },
         'transform': {'input_exclusive_max': [[5]], 'input_inclusive_min': [0]},
       })
       >>> # Re-open later without re-reading metadata
       >>> store2 = await ts.open(spec,
       ...                        context=context,
       ...                        open=True,
       ...                        assume_metadata=True)
       >>> # Read data using the unverified metadata from `spec`
       >>> await store2.read()

    .. admonition:: Example of skipping writing the metadata when creating a new array
       :class: example

       >>> context = ts.Context()
       >>> spec = ts.Spec(json={"driver": "zarr", "kvstore": "memory://"})
       >>> spec.update(dtype=ts.float32, shape=[5])
       >>> # Open the array without writing the metadata.  If using a distributed
       >>> # filesystem, this can safely be executed on multiple machines concurrently,
       >>> # provided that the `spec` is identical and the metadata is either fully
       >>> # constrained, or exactly the same TensorStore version is used to ensure the
       >>> # same defaults are applied.
       >>> store = await ts.open(spec,
       ...                       context=context,
       ...                       open=True,
       ...                       create=True,
       ...                       assume_metadata=True)
       >>> await store.write([1, 2, 3, 4, 5])
       >>> # Note that the data chunk has been written but not the .zarray metadata
       >>> await store.kvstore.list()
       [b'0']
       >>> # From a single machine, actually write the metadata to ensure the array
       >>> # can be re-opened knowing the metadata.  This can be done in parallel with
       >>> # any other writing.
       >>> await ts.open(spec, context=context, open=True, create=True)
       >>> # Metadata has now been written.
       >>> await store.kvstore.list()
       [b'.zarray', b'0']

    Group:
      Core
    """


def overlay(
    layers: collections.abc.Iterable[TensorStore | Spec],
    *,
    read: bool | None = None,
    write: bool | None = None,
    context: Context | None = None,
    transaction: Transaction | None = None,
    rank: int | None = None,
    dtype: DTypeLike | None = None,
    domain: IndexDomain | None = None,
    shape: collections.abc.Iterable[int] | None = None,
    dimension_units: (
        collections.abc.Iterable[
            Unit | str | numbers.Real | tuple[numbers.Real, str] | None
        ]
        | None
    ) = None,
    schema: Schema | None = None,
) -> TensorStore:
    """
    Virtually overlays a sequence of :py:obj:`TensorStore` layers within a common domain.

        >>> store = ts.overlay([
        ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
        ...     ts.array([5, 6, 7, 8], dtype=ts.uint32).translate_to[3]
        ... ])
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [1, 2, 3, 4],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {'input_exclusive_max': [4], 'input_inclusive_min': [0]},
            },
            {
              'array': [5, 6, 7, 8],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [7],
                'input_inclusive_min': [3],
                'output': [{'input_dimension': 0, 'offset': -3}],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [7], 'inclusive_min': [0]}},
          'transform': {'input_exclusive_max': [7], 'input_inclusive_min': [0]},
        })
        >>> await store.read()
        array([1, 2, 3, 5, 6, 7, 8], dtype=uint32)

    Args:

      layers: Sequence of layers to overlay.  Later layers take precedence.  If a
        layer is specified as a :py:obj:`Spec` rather than a :py:obj:`TensorStore`,
        it must have a known :py:obj:`~Spec.domain` and will be opened on-demand as
        neneded for individual read and write operations.

      read: Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      write: Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      context: Shared resource context.  Defaults to a new (unshared) context with default
        options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
        such as cache pools, between multiple open TensorStores, you must specify a
        context.
      transaction: Transaction to use for opening/creating, and for subsequent operations.  By
        default, the open is non-transactional.

        .. note::

           To perform transactional operations using a :py:obj:`TensorStore` that was
           previously opened without a transaction, use
           :py:obj:`TensorStore.with_transaction`.
      rank: Constrains the rank of the TensorStore.  If there is an index transform, the
        rank constraint must match the rank of the *input* space.
      dtype: Constrains the data type of the TensorStore.  If a data type has already been
        set, it is an error to specify a different data type.
      domain: Constrains the domain of the TensorStore.  If there is an existing
        domain, the specified domain is merged with it as follows:

        1. The rank must match the existing rank.

        2. All bounds must match, except that a finite or explicit bound is permitted to
           match an infinite and implicit bound, and takes precedence.

        3. If both the new and existing domain specify non-empty labels for a dimension,
           the labels must be equal.  If only one of the domains specifies a non-empty
           label for a dimension, the non-empty label takes precedence.

        Note that if there is an index transform, the domain must match the *input*
        space, not the output space.
      shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
        :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

        .. note::

           This option also constrains the origin of all dimensions to be zero.
      dimension_units: Specifies the physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity can be indicated by a unit of :python:`""`.
      schema: Additional schema constraints to merge with existing constraints.


    See also:
      - :ref:`driver/stack`
      - :py:obj:`tensorstore.stack`
      - :py:obj:`tensorstore.concat`

    Group:
      Views
    """


def parse_tensorstore_flags(argv: list[str]) -> None:
    """
    Parses and initializes internal tensorstore flags from argv.

    Args:
      argv: list of command line argument strings, such as sys.argv.

    Group:
      Experimental
    """


def stack(
    layers: collections.abc.Iterable[TensorStore | Spec],
    axis: int = 0,
    *,
    read: bool | None = None,
    write: bool | None = None,
    context: Context | None = None,
    transaction: Transaction | None = None,
    rank: int | None = None,
    dtype: DTypeLike | None = None,
    domain: IndexDomain | None = None,
    shape: collections.abc.Iterable[int] | None = None,
    dimension_units: (
        collections.abc.Iterable[
            Unit | str | numbers.Real | tuple[numbers.Real, str] | None
        ]
        | None
    ) = None,
    schema: Schema | None = None,
) -> TensorStore:
    """
    Virtually stacks a sequence of :py:obj:`TensorStore` layers along a new dimension.

        >>> store = ts.stack([
        ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
        ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
        ... ])
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [1, 2, 3, 4],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [1, 4],
                'input_inclusive_min': [0, 0],
                'output': [{'input_dimension': 1}],
              },
            },
            {
              'array': [5, 6, 7, 8],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [2, 4],
                'input_inclusive_min': [1, 0],
                'output': [{'input_dimension': 1}],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [2, 4], 'inclusive_min': [0, 0]}},
          'transform': {'input_exclusive_max': [2, 4], 'input_inclusive_min': [0, 0]},
        })
        >>> await store.read()
        array([[1, 2, 3, 4],
               [5, 6, 7, 8]], dtype=uint32)
        >>> store = ts.stack([
        ...     ts.array([1, 2, 3, 4], dtype=ts.uint32),
        ...     ts.array([5, 6, 7, 8], dtype=ts.uint32)
        ... ],
        ...                  axis=-1)
        >>> store
        TensorStore({
          'context': {'data_copy_concurrency': {}},
          'driver': 'stack',
          'dtype': 'uint32',
          'layers': [
            {
              'array': [1, 2, 3, 4],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [4, 1],
                'input_inclusive_min': [0, 0],
                'output': [{'input_dimension': 0}],
              },
            },
            {
              'array': [5, 6, 7, 8],
              'driver': 'array',
              'dtype': 'uint32',
              'transform': {
                'input_exclusive_max': [4, 2],
                'input_inclusive_min': [0, 1],
                'output': [{'input_dimension': 0}],
              },
            },
          ],
          'schema': {'domain': {'exclusive_max': [4, 2], 'inclusive_min': [0, 0]}},
          'transform': {'input_exclusive_max': [4, 2], 'input_inclusive_min': [0, 0]},
        })
        >>> await store.read()
        array([[1, 5],
               [2, 6],
               [3, 7],
               [4, 8]], dtype=uint32)

    Args:

      layers: Sequence of layers to stack.  If a layer is specified as a
        :py:obj:`Spec` rather than a :py:obj:`TensorStore`, it must have a known
        :py:obj:`~Spec.domain` and will be opened on-demand as needed for individual
        read and write operations.

      axis: New dimension along which to stack.  A negative number counts from the end.
      read: Allow read access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      write: Allow write access.  Defaults to `True` if neither ``read`` nor ``write`` is specified.
      context: Shared resource context.  Defaults to a new (unshared) context with default
        options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
        such as cache pools, between multiple open TensorStores, you must specify a
        context.
      transaction: Transaction to use for opening/creating, and for subsequent operations.  By
        default, the open is non-transactional.

        .. note::

           To perform transactional operations using a :py:obj:`TensorStore` that was
           previously opened without a transaction, use
           :py:obj:`TensorStore.with_transaction`.
      rank: Constrains the rank of the TensorStore.  If there is an index transform, the
        rank constraint must match the rank of the *input* space.
      dtype: Constrains the data type of the TensorStore.  If a data type has already been
        set, it is an error to specify a different data type.
      domain: Constrains the domain of the TensorStore.  If there is an existing
        domain, the specified domain is merged with it as follows:

        1. The rank must match the existing rank.

        2. All bounds must match, except that a finite or explicit bound is permitted to
           match an infinite and implicit bound, and takes precedence.

        3. If both the new and existing domain specify non-empty labels for a dimension,
           the labels must be equal.  If only one of the domains specifies a non-empty
           label for a dimension, the non-empty label takes precedence.

        Note that if there is an index transform, the domain must match the *input*
        space, not the output space.
      shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
        :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

        .. note::

           This option also constrains the origin of all dimensions to be zero.
      dimension_units: Specifies the physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity can be indicated by a unit of :python:`""`.
      schema: Additional schema constraints to merge with existing constraints.


    See also:
      - :py:obj:`numpy.stack`
      - :ref:`driver/stack`
      - :py:obj:`tensorstore.overlay`
      - :py:obj:`tensorstore.concat`

    Group:
      Views
    """


def virtual_chunked(
    read_function: (
        typing.Callable[
            [IndexDomain, numpy.ndarray, VirtualChunkedReadParameters],
            FutureLike[KvStore.TimestampedStorageGeneration | None],
        ]
        | None
    ) = None,
    write_function: (
        typing.Callable[
            [IndexDomain, numpy.ndarray, VirtualChunkedWriteParameters],
            FutureLike[KvStore.TimestampedStorageGeneration | None],
        ]
        | None
    ) = None,
    *,
    loop: asyncio.AbstractEventLoop | None = None,
    rank: int | None = None,
    dtype: DTypeLike | None = None,
    domain: IndexDomain | None = None,
    shape: collections.abc.Iterable[int] | None = None,
    chunk_layout: ChunkLayout | None = None,
    dimension_units: (
        collections.abc.Iterable[
            Unit | str | numbers.Real | tuple[numbers.Real, str] | None
        ]
        | None
    ) = None,
    schema: Schema | None = None,
    context: Context | None = None,
    transaction: Transaction | None = None,
) -> TensorStore:
    """
    Creates a :py:obj:`.TensorStore` where the content is read/written chunk-wise by an arbitrary function.

    Example (read-only):

        >>> a = ts.array([[1, 2, 3], [4, 5, 6]], dtype=ts.uint32)
        >>> async def do_read(domain: ts.IndexDomain, array: np.ndarray,
        ...                   read_params: ts.VirtualChunkedReadParameters):
        ...     print(f'Computing content for: {domain}')
        ...     array[...] = (await a[domain].read()) + 100
        >>> t = ts.virtual_chunked(do_read, dtype=a.dtype, domain=a.domain)
        >>> await t.read()
        Computing content for: { [0, 2), [0, 3) }
        array([[101, 102, 103],
               [104, 105, 106]], dtype=uint32)

    Example (read/write):

        >>> array = np.zeros(shape=[4, 5], dtype=np.uint32)
        >>> array[1] = 50
        >>> def do_read(domain, chunk, read_context):
        ...     chunk[...] = array[domain.index_exp]
        >>> def do_write(domain, chunk, write_context):
        ...     array[domain.index_exp] = chunk
        >>> t = ts.virtual_chunked(
        ...     do_read,
        ...     do_write,
        ...     dtype=array.dtype,
        ...     shape=array.shape,
        ...     chunk_layout=ts.ChunkLayout(read_chunk_shape=(2, 3)))
        >>> await t.read()
        array([[ 0,  0,  0,  0,  0],
               [50, 50, 50, 50, 50],
               [ 0,  0,  0,  0,  0],
               [ 0,  0,  0,  0,  0]], dtype=uint32)
        >>> t[1:3, 1:3] = 42
        >>> array
        array([[ 0,  0,  0,  0,  0],
               [50, 42, 42, 50, 50],
               [ 0, 42, 42,  0,  0],
               [ 0,  0,  0,  0,  0]], dtype=uint32)

    Args:

      read_function: Callback that handles chunk read requests.  Must be specified
        to create a virtual view that supports reads.  To create a write-only view,
        leave this unspecified (as :py:obj:`None`).

        This function should assign to the array the content for the specified
        :py:obj:`~tensorstore.IndexDomain`.

        The returned :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration`
        identifies the version of the content, for caching purposes.  If versioning
        is not applicable, :py:obj:`None` may be returned to indicate a value that
        may be cached indefinitely.

        If it returns a :ref:`coroutine<python:async>`, the coroutine will be
        executed using the event loop indicated by :py:param:`.loop`.

      write_function: Callback that handles chunk write requests.  Must be specified
        to create a virtual view that supports writes.  To create a read-only view,
        leave this unspecified (as :py:obj:`None`).

        This function store the content of the array for the specified
        :py:obj:`~tensorstore.IndexDomain`.

        The returned :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration`
        identifies the stored version of the content, for caching purposes.  If
        versioning is not applicable, :py:obj:`None` may be returned to indicate a
        value that may be cached indefinitely.

        If it returns a :ref:`coroutine<python:async>`, the coroutine will be
        executed using the event loop indicated by :py:param:`.loop`.

      loop: Event loop on which to execute :py:param:`.read_function` and/or
        :py:param:`.write_function` if they are
        :ref:`async functions<python:async def>`.  If not specified (or
        :py:obj:`None` is specified), defaults to the loop returned by
        :py:obj:`asyncio.get_running_loop` (in the context of the call to
        :py:obj:`.virtual_chunked`).  If :py:param:`.loop` is not specified and
        there is no running event loop, it is an error for
        :py:param:`.read_function` or :py:param:`.write_function` to return a
        coroutine.

      rank: Constrains the rank of the TensorStore.  If there is an index transform, the
        rank constraint must match the rank of the *input* space.
      dtype: Constrains the data type of the TensorStore.  If a data type has already been
        set, it is an error to specify a different data type.
      domain: Constrains the domain of the TensorStore.  If there is an existing
        domain, the specified domain is merged with it as follows:

        1. The rank must match the existing rank.

        2. All bounds must match, except that a finite or explicit bound is permitted to
           match an infinite and implicit bound, and takes precedence.

        3. If both the new and existing domain specify non-empty labels for a dimension,
           the labels must be equal.  If only one of the domains specifies a non-empty
           label for a dimension, the non-empty label takes precedence.

        Note that if there is an index transform, the domain must match the *input*
        space, not the output space.
      shape: Constrains the shape and origin of the TensorStore.  Equivalent to specifying a
        :py:param:`domain` of :python:`ts.IndexDomain(shape=shape)`.

        .. note::

           This option also constrains the origin of all dimensions to be zero.
      chunk_layout: Constrains the chunk layout.  If there is an existing chunk layout constraint,
        the constraints are merged.  If the constraints are incompatible, an error
        is raised.
      dimension_units: Specifies the physical units of each dimension of the domain.

        The *physical unit* for a dimension is the physical quantity corresponding to a
        single index increment along each dimension.

        A value of :python:`None` indicates that the unit is unknown.  A dimension-less
        quantity can be indicated by a unit of :python:`""`.
      schema: Additional schema constraints to merge with existing constraints.
      context: Shared resource context.  Defaults to a new (unshared) context with default
        options, as returned by :py:meth:`tensorstore.Context`.  To share resources,
        such as cache pools, between multiple open TensorStores, you must specify a
        context.
      transaction: Transaction to use for opening/creating, and for subsequent operations.  By
        default, the open is non-transactional.

        .. note::

           To perform transactional operations using a TensorStore that was previously
           opened without a transaction, use :py:obj:`TensorStore.with_transaction`.



    Warning:

      Neither :py:param:`.read_function` nor :py:param:`.write_function` should
      block synchronously while waiting for another TensorStore operation; blocking
      on another operation that uses the same
      :json:schema:`Context.data_copy_concurrency` resource may result in deadlock.
      Instead, it is better to specify a :ref:`coroutine function<python:async def>`
      for :py:param:`.read_function` and :py:param:`.write_function` and use
      :ref:`await<python:await>` to wait for the result of other TensorStore
      operations.

    Group:
      Virtual views

    Caching
    -------

    By default, the computed content of chunks is not cached, and will be
    recomputed on every read.  To enable caching:

    - Specify a :py:obj:`~tensorstore.Context` that contains a
      :json:schema:`~Context.cache_pool` with a non-zero size limit, e.g.:
      :json:`{"cache_pool": {"total_bytes_limit": 100000000}}` for 100MB.

    - Additionally, if the data is not immutable, the :py:param:`read_function`
      should return a unique generation and a timestamp that is not
      :python:`float('inf')`.  When a cached chunk is re-read, the
      :py:param:`read_function` will be called with
      :py:obj:`~tensorstore.VirtualChunkedReadParameters.if_not_equal` specified.
      If the generation specified by
      :py:obj:`~tensorstore.VirtualChunkedReadParameters.if_not_equal` is still
      current, the :py:param:`read_function` may leave the output array unmodified
      and return a :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration` with
      an appropriate
      :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration.time` but
      :py:obj:`~tensorstore.KvStore.TimestampedStorageGeneration.generation` left
      unspecified.

    Pickle support
    --------------

    The returned :py:obj:`.TensorStore` supports pickling if, and only if, the
    :py:param:`.read_function` and :py:param:`.write_function` support pickling.

    .. note::

       The :py:mod:`pickle` module only supports global functions defined in named
       modules.  For broader function support, you may wish to use
       `cloudpickle <https://github.com/cloudpipe/cloudpickle>`__.

    .. warning::

       The specified :py:param:`.loop` is not preserved when the returned
       :py:obj:`.TensorStore` is pickled, since it is a property of the current
       thread.  Instead, when unpickled, the resultant :py:obj:`.TensorStore` will
       use the running event loop (as returned by
       :py:obj:`asyncio.get_running_loop`) of the thread used for unpickling, if
       there is one.

    Transaction support
    -------------------

    Transactional reads and writes are supported on virtual_chunked views.  A
    transactional write simply serves to buffer the write in memory until it is
    committed.  Transactional reads will observe prior writes made using the same
    transaction.  However, when the transaction commit is initiated, the
    :py:param:`.write_function` is called in exactly the same way as for a
    non-transactional write, and if more than one chunk is affected, the commit will
    be non-atomic.  If the transaction is atomic, it is an error to write to more
    than one chunk in the same transaction.

    You are also free to use transactional operations, e.g. operations on a
    :py:class:`.KvStore` or another :py:class:`.TensorStore`, within the
    :py:param:`.read_function` or :py:param:`.write_function`.

    - For read-write views, you should not attempt to use the same transaction
      within the :py:param:`.read_function` or :py:param:`.write_function` that is
      also used for read or write operations on the virtual view directly, because
      both :py:param:`.write_function` and :py:param:`.read_function` may be called
      after the commit starts, and any attempt to perform new operations using the
      same transaction once it is already being committed will fail; instead, any
      transactional operations performed within the :py:param:`.read_function` or
      :py:param:`.write_function` should use a different transaction.

    - For read-only views, it is possible to use the same transaction within the
      :py:param:`.read_function` as is also used for read operations on the virtual
      view directly, though this may not be particularly useful.

    Specifying a transaction directly when creating the virtual chunked view is no
    different than binding the transaction to an existing virtual chunked view.
    """
