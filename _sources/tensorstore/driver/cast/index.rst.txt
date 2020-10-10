``cast`` Driver
===============

Virtual read/write view that performs element-wise data type conversion.

- Reading is supported if the :json-member:`base` TensorStore supports reading
  and conversion from the base data type to the view data type is supported.

- Writing is supported if the :json-member:`base` TensorStore supports writing
  and conversion from the view data type to the base data type is supported.

The top-level :json-member:`transform`, if any, is composed with the
:json-member:`transform`, if any, specified on :json-member:`base`.

.. json-schema:: schema.yml
   :title: JSON Schema
