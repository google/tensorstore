#!/usr/bin/env python3
# Copyright 2023 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates DataTypeConversionTraits specializations."""

import io
import os

import update_generated_source_code

MAP = {
    'BOOL': ['bool_t'],
    'BYTE': ['char_t', 'byte_t'],
    'INT': [
        'int4_t',
        # 'uint4_t',  # TODO(summivox): b/295577703
        'int8_t',
        'uint8_t',
        'int16_t',
        'uint16_t',
        'int32_t',
        'uint32_t',
        'int64_t',
        'uint64_t',
    ],
    'FLOAT': [
        'float8_e4m3fn_t',
        'float8_e4m3fnuz_t',
        'float8_e4m3b11fnuz_t',
        'float8_e5m2_t',
        'float8_e5m2fnuz_t',
        'float16_t',
        'bfloat16_t',
        'float32_t',
        'float64_t',
    ],
    'COMPLEX': ['complex64_t', 'complex128_t'],
    'STRING': ['string_t', 'ustring_t'],
    'JSON': ['json_t'],
}

KEYS = MAP.keys()


TENSORSTORE_INTERNAL_INHERITED_CONVERT = [
    (
        'INT',
        'INT',
        'internal_data_type::IntegerIntegerDataTypeConversionTraits',
    ),
    (
        'INT',
        'FLOAT',
        'internal_data_type::IntegerFloatDataTypeConversionTraits',
    ),
    (
        'INT',
        'COMPLEX',
        'internal_data_type::NumericComplexDataTypeConversionTraits',
    ),
    (
        'INT',
        'JSON',
        'internal_data_type::IntegerJsonDataTypeConversionTraits',
    ),
    (
        'FLOAT',
        'FLOAT',
        'internal_data_type::FloatFloatDataTypeConversionTraits',
    ),
    (
        'FLOAT',
        'COMPLEX',
        'internal_data_type::NumericComplexDataTypeConversionTraits',
    ),
    (
        'FLOAT',
        'JSON',
        'internal_data_type::FloatJsonDataTypeConversionTraits',
    ),
    (
        'COMPLEX',
        'COMPLEX',
        'internal_data_type::ComplexComplexDataTypeConversionTraits',
    ),
]

TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS = {
    'DataTypeConversionFlags::kSupported | DataTypeConversionFlags::kSafeAndImplicit | DataTypeConversionFlags::kCanReinterpretCast': [
        ('char_t', ('byte_t',)),
        ('ustring_t', ('string_t',)),
    ],
    'DataTypeConversionFlags::kSupported | DataTypeConversionFlags::kSafeAndImplicit': [
        ('BOOL', ('INT', 'FLOAT', 'COMPLEX', 'JSON')),
        ('byte_t', ('char_t',)),
        ('ustring_t', ('JSON',)),
    ],
    'DataTypeConversionFlags::kSupported': [
        ('INT', ('BOOL', 'STRING')),
        ('FLOAT', ('BOOL', 'INT', 'STRING')),
        ('COMPLEX', ('INT', 'FLOAT', 'STRING')),
        ('JSON', ('BOOL', 'INT', 'FLOAT', 'STRING')),
        ('string_t', ('ustring_t', 'JSON')),
    ],
}


def data_type_conversion_h(out):
  """Generates the specializations for data_type_conversion.h."""

  out.write('\n')

  for args in TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS:
    for specialization in TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS[args]:
      f, ts = specialization
      assert isinstance(ts, tuple)
      for f0 in MAP.get(f, [f]):
        for t in ts:
          for t0 in MAP.get(t, [t]):
            out.write(
                'TENSORSTORE_INTERNAL_DEFINE_CONVERT_TRAITS(//\n'
                f'::tensorstore::dtypes::{f0}, ::tensorstore::dtypes::{t0},'
                f' {args})\n'
            )

  for specialization in TENSORSTORE_INTERNAL_INHERITED_CONVERT:
    f, t, args = specialization
    for f0 in MAP.get(f, [f]):
      for t0 in MAP.get(t, [t]):
        out.write(
            'TENSORSTORE_INTERNAL_INHERITED_CONVERT(//\n'
            f'::tensorstore::dtypes::{f0}, ::tensorstore::dtypes::{t0},'
            f' {args})\n'
        )

  out.write('\n')
  return out


DATA_TYPE_CC = [
    (
        'COMPLEX',
        'INT',
        'internal_data_type::ComplexNumericConvertDataType',
    ),
    (
        'COMPLEX',
        'FLOAT',
        'internal_data_type::ComplexNumericConvertDataType',
    ),
]


def data_type_cc(out):
  """Generates ConvertDataType specializations defined in data_type.cc."""

  out.write('\n')
  for specialization in DATA_TYPE_CC:
    f, t, parent = specialization
    for f0 in MAP.get(f, [f]):
      for t0 in MAP.get(t, [t]):
        out.write(
            'TENSORSTORE_INTERNAL_INHERITED_CONVERT(//\n'
            f'::tensorstore::dtypes::{f0}, ::tensorstore::dtypes::{t0},'
            f' {parent})\n'
        )
  out.write('\n')
  return out


def data_type_h(out):
  """Generates the specializations for data_type.h."""
  out.write("""
// Define a DataTypeId `x_t` corresponding to each C++ type `tensorstore::x_t`
// defined above.
enum class DataTypeId {
  custom = -1,
""")
  for k in KEYS:
    for t in MAP[k]:
      out.write(f'  {t},\n')

  out.write("""num_ids
};

inline constexpr size_t kNumDataTypeIds =
    static_cast<size_t>(DataTypeId::num_ids);

// TENSORSTORE_FOR_EACH_DATA_TYPE(X, ...) macros will instantiate
// X(datatype, ...) for each tensorstore data type.
""")

  for k in KEYS:
    out.write(f'#define TENSORSTORE_FOR_EACH_{k}_DATA_TYPE(X, ...) \\\n')
    for t in MAP[k]:
      out.write(f'X({t}, ##__VA_ARGS__) \\\n')
    out.write('  /**/\n\n')

  out.write('#define TENSORSTORE_FOR_EACH_DATA_TYPE(X, ...) \\\n')
  for k in KEYS:
    out.write(f'TENSORSTORE_FOR_EACH_{k}_DATA_TYPE(X, ##__VA_ARGS__) \\\n')
  out.write('  /**/\n\n')
  return out


def main():
  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'data_type.h'),
      script_name=os.path.basename(__file__),
      new_content=data_type_h(io.StringIO()).getvalue(),
  )

  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'data_type.cc'),
      script_name=os.path.basename(__file__),
      new_content=data_type_cc(io.StringIO()).getvalue(),
  )

  update_generated_source_code.update_generated_content(
      path=os.path.join(os.path.dirname(__file__), 'data_type_conversion.h'),
      script_name=os.path.basename(__file__),
      new_content=data_type_conversion_h(io.StringIO()).getvalue(),
  )


if __name__ == '__main__':
  main()
