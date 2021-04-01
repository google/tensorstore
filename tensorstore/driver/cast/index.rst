``cast`` Driver
===============

Virtual read/write view that performs element-wise data type conversion.

- Reading is supported if the :json:schema:`~driver/cast.base` TensorStore supports reading
  and conversion from the base data type to the view data type is supported.

- Writing is supported if the :json:schema:`~driver/cast.base` TensorStore supports
  writing and conversion from the view data type to the base data type is
  supported.

The top-level :json:schema:`~driver/cast.transform`, if any, is composed
with the :json:schema:`TensorStore.transform`, if any, specified on
:json:schema:`~driver/cast.base`.

.. json:schema:: driver/cast
