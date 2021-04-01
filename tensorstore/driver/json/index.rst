``json`` Driver
================

The ``json`` driver provides read/write access to JSON values stored in any
supported :ref:`key_value_store`.  JSON values are accessed as rank-0 arrays
with :json:`"json"` data type.

.. note::
   
   Even if the JSON value is itself an array value, TensorStore still exposes it
   as a rank-0 array of a single JSON value.

Sub-values within a JSON file may be specified using JSON Pointer syntax
(:rfc:`6901`), and TensorStore guarantees consistency in the presence of
concurrent writes because writes are implemented as atomic read-modify-write
operations to individual JSON Pointers.  If two processes concurrently write to
non-overlapping pointers within the same JSON file, it is guaranteed that
neither write will be lost.

.. json:schema:: driver/json
