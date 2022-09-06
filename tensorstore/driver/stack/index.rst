``stack`` Driver
================

The ``stack`` driver specifies a TensorStore backed by a sequence of layered
drivers, where each underlying layer describes it's bounds via a transform.

- The TensorStore for a stack driver has staic bounds, which may be derived
  from the underlying :json:schema:`~driver/stack.layers`

- Reading is supported if the :json:schema:`~driver/stack.layers` TensorStore
  supports reading.

- Writing is supported if the :json:schema:`~driver/stack.layers` TensorStore
  supports writing.

The layers may overlap, with the subsequent layers overriding values from
earlier layers in the stack.

The underlying ``stack`` TensorStore :json:schema:`~driver/stack.layers` are
opened on demand by each operation.


.. json:schema:: driver/stack


Example
-----------

.. admonition:: Example of layer driver
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
-----------------------------

Only the following TensorStore schema constraints are useful to the ``stack``
driver. They will be propagated and will cause errors when incompatible with
the layer stack.

- :json:schema:`Schema.rank`  Sets the rank of the TensorStore and constrains
  the rank of each of the :json:schema:`~driver/stack.layers`

- :json:schema:`Schema.dtype`  Sets the dtype of the TensorStore and constrains
  the dtype of each of the :json:schema:`~driver/stack.layers`

- :json:schema:`Schema.domain`  Sets the domain of the TensorStore. The ``stack``
  driver computes an effective domain as the hull of each layer's effective
  domain, as derived from the layer schema and transform. For each dimension
  bound which is implicit and unbounded (as when the ``domain`` is omitted),
  the bounds for that dimension are taken from the effective domain.

- :json:schema:`Schema.dimension_units`  Sets the dimension units of the
  TensorStore. The ``stack`` driver will set any unset dimension units from the
  individual layers as long as they are in common.


Scheama options that are not allowed:

- :json:schema:`Schema.fill_value`
- :json:schema:`Schema.codec`
- :json:schema:`Schema.chunk_layout`
