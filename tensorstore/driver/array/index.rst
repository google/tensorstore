``array`` Driver
================

The ``array`` driver specifies a TensorStore backed by a contiguous in-memory
array.

When specified via JSON, this driver is useful for specifying small, constant
arrays.

When used from the C++ or Python API, this driver adapts an existing in-memory
array for use as a TensorStore.

.. json:schema:: driver/array
