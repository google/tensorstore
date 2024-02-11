.. _stack-driver:

``stack`` Driver
================

The ``stack`` driver specifies a TensorStore virtually overlays a sequence of
TensorStore *layers* within a common domain.

- If the domains of the :json:schema:`~driver/stack.layers` may overlap, the
  last layer that contains any given position within the domain takes
  precedence.  This last layer that contains the position is said to be the
  *backing* layer for that position.

- By choosing appropriate transforms for the layers, this driver may be used to
  virtually stack a sequence of TensorStores along a new dimension, or to
  concatenate a sequence of TensorStores along an existing dimension.  For these
  use cases, the layer domains do not overlap.

Python API
----------

Python APIs are provided for constructing a stack-driver-backed TensorStore
based on a sequence of :py:obj:`~tensorstore.Spec` or already-open
:py:obj:`~tensorstore.TensorStore` objects.

- :py:obj:`tensorstore.overlay` virtually overlays a sequence of layers.
- :py:obj:`tensorstore.stack` stacks a sequence of layers along a new dimension.
- :py:obj:`tensorstore.concat` concatenates a sequence of layers along an
  existing dimension.

Supported operations
--------------------

Opening the stack driver itself does not open any of the layers.  Instead,
layers are opened on demand as needed to satisfy read and write operations.

Reading or writing at a given position is supported, if any only if, the backing
layer supports reading or writing, respectively.  Reading or writing at a
position that has no backing layer results in an error.

.. json:schema:: driver/stack


Example
-------

.. admonition:: Example of stack driver
   :class: example

   >>> ts.open({
   ...     'driver':
   ...         'stack',
   ...     'layers': [
   ...         {
   ...             'driver': 'array',
   ...             'dtype': 'int32',
   ...             'array': [1, 2, 3, 4],
   ...         },
   ...         {
   ...             'driver': 'array',
   ...             'dtype': 'int32',
   ...             'array': [1, 2, 3, 4],
   ...             'transform': {
   ...                 'input_inclusive_min': [4],
   ...                 'input_exclusive_max': [8],
   ...                 'output': [{
   ...                     'input_dimension': 0,
   ...                     'offset': -4
   ...                 }]
   ...             },
   ...         },
   ...     ]
   ... }).result().schema.dtype
   dtype("int32")


TensorStore Schema
------------------

Data type
~~~~~~~~~

The :json:schema:`~Schema.dtype` must be specified for at least one layer, or in
the top-level :json:schema:`~driver/stack.schema`.  All specified data types
must match.

Domain
~~~~~~

The bounds of a stack driver-backed TensorStore, and of each layer, are fixed
when it is opened, based on the domains specified for each of the
:json:schema:`~driver/stack.layers`, and the :json:schema:`~Schema.domain`
specified for the stack driver, if any.

Any :ref:`implicit/resizeable bounds<implicit-bounds>` of layers become fixed
(explicit) bounds and will not be affected by subsequent resize operations.

By default, the bounds of the stack-driver-backed TensorStore are determined by
computing the hull of the effective domains of the layers, but any finite or
explicit bound specified on the top-level stack-driver
:json:schema:`~Schema.domain` takes precedence and overrides the bound
determined from the hull.

Note that it is valid for the domain to include positions that are not backed by
any layer, but any attempt to read or write such positions results in an error.

The :ref:`dimension labels<dimension-labels>` are merged and must be compatible.

Dimension units
~~~~~~~~~~~~~~~

For each dimension, the corresponding dimension unit is determined as follows:

1. If a unit for the dimension is specified directly in the top-level stack-driver
   :json:schema:`Schema.dimension_units`, the specified unit is assigned.

2. Otherwise, if there is agreement among all the layers that specify a unit for
   the dimension, the common unit is assigned.

3. Otherwise, the unit is unassigned (:json:`null`).

Fill value
~~~~~~~~~~

Fill values are not supported.

Codec
~~~~~

Codecs are not supported.

Chunk layout
~~~~~~~~~~~~

Chunk layouts are not supported.
