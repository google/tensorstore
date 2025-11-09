// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::Index;

struct X {
  float a;
  int b;

  friend std::ostream& operator<<(std::ostream& os, const X& x) {
    os << "<" << x.a << ", " << x.b << ">";
    return os;
  }
};

template <typename Array>
void PrintCSVArray(Array&& data) {
  if (data.rank() == 0) {
    std::cout << data << std::endl;
    return;
  }

  // Iterate over the shape of the data array, which gives us one
  // reference for every element.
  //
  // The builtin streaming operator outputs data in C++ array initialization
  // syntax: {{0, 0}, {1, 0}}, but this routine prefers CSV-formatted output.
  //
  // The output of this function is equivalent to:
  //
  // for (int x = 0; x < data.shape()[0]; x++)
  //  for (int y = 0; y < data.shape()[1]; y++) {
  //     ...
  //       std::cout << data[x][y][...] << "\t";
  //  }
  //
  const auto max = data.shape()[data.rank() - 1] - 1;
  auto element_rep = data.dtype();

  // FIXME: We can't use operator() to get a value reference since that doesn't
  // work for tensorstore::ArrayView<const void, N>. However in the case of
  // printing, rank-0 arrays have been overloaded to print correctly, and so we
  // can do this:
  std::string s;
  tensorstore::IterateOverIndexRange(  //
      data.shape(), [&](tensorstore::span<const Index> idx) {
        element_rep->append_to_string(&s, data[idx].pointer());
        if (*idx.rbegin() == max) {
          std::cout << s << std::endl;
          s.clear();
        } else {
          s.append("\t");
        }
      });
  std::cout << s << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  // How do we create arrays? Let's try this multiple different ways.
  const int two_dim[5][5] =  //
      {{0, 0, 0, 0, 0},      //
       {0, 72, 53, 60, 0},   //
       {0, 76, 56, 65, 0},   //
       {0, 88, 78, 82, 0},   //
       {0, 0, 0, 0, 0}};

  const float three_dim[3][3][3] = {{
                                        {1, 2, 3},  //
                                        {4, 5, 6},  //
                                        {7, 8, 9}   //
                                    },
                                    {
                                        {10, 11, 12},  //
                                        {13, 14, 15},  //
                                        {16, 17, 18}   //
                                    },
                                    {
                                        {19, 20, 21},  //
                                        {22, 23, 24},  //
                                        {25, 26, 27}   //
                                    }};

  // tensorstore::ArrayView can be created from native C++ arrays of arbitrary
  // dimensions. The number of dimensions in an array is the rank of the array.

  {
    std::cout << std::endl << "Native C++ arrays (rank 1)" << std::endl;

    // This creates a reference to an array of rank 1.
    int one_dim[5] = {1, 2, 3, 4, 5};
    auto array_ref = tensorstore::MakeArrayView(one_dim);
    PrintCSVArray(array_ref);

    // tensorstore::ArrayView does not own the underlying data, so it can be
    // modified, and the underlying storage must outlive the ArrayView.

    // The most efficient way to index a tensorstore::ArrayView is using
    // operator(), which is similar to fortran syntax:
    array_ref(2) = 0;
    PrintCSVArray(array_ref);

    // FIXME: tensorstore::MakeArrayView does not work with tensorstore::span
    /*
    PrintCSVArray(tensorstore::MakeArrayView(tensorstore::span(one_dim)));
    */
  }

  {
    std::cout << std::endl << "Native C++ arrays (rank 2)" << std::endl;

    PrintCSVArray(tensorstore::MakeArrayView(two_dim));
  }

  // tensorstore::Array and tensorstore::ArrayView can be of any primitive type.
  {
    std::cout << std::endl << "Native C++ arrays (double)" << std::endl;

    const double two_dim_double[5][5] =  //
        {{0, 0, 0, 0, 0},                //
         {0, .72, .53, .60, 0},          //
         {0, .76, .56, .65, 0},          //
         {0, .88, .78, .82, 0},          //
         {0, 0, 0, 0, 0}};

    auto array_ref = tensorstore::MakeArrayView(two_dim_double);
    PrintCSVArray(array_ref);

    // The rank and shape of an array can be inspected:
    std::cout << "rank=" << array_ref.rank() << " shape=" << array_ref.shape();
  }

  // FIXME: We cannot create an array ref from a C++ std::array type.
  /*
  std::array<std::array<int, 3>, 3> arr = {{{5, 8, 2}, {8, 3, 1}, {5, 3, 9}}};
  PrintCSVArray(tensorstore::MakeArrayView(arr));
  */

  // tensorstore also includes the concept of rank-0 arrays, or Scalars.
  // these are created from a single value.
  {
    std::cout << std::endl << "Scalar Array" << std::endl;
    PrintCSVArray(tensorstore::MakeScalarArray(123).array_view());
  }

  // In addition to the tensorstore::ArrayView types, tensorstore can allocate
  // arrays dynamically of specified types, rank, and dimensions.
  {
    std::cout << std::endl << "Allocated Array" << std::endl;
    auto allocated = tensorstore::AllocateArray<int>({5, 5});
    if (tensorstore::ArraysHaveSameShapes(tensorstore::MakeArrayView(two_dim),
                                          allocated)) {
      // Copy uses the same order as std::copy, rather than the reversed
      // order from std::memcpy.
      tensorstore::CopyArray(tensorstore::MakeArrayView(two_dim), allocated);
    }
    PrintCSVArray(allocated.array_view());
    std::cout << std::endl;

    // FIXME: It seems like we might want to require this conversion to be
    // explicit rather than silently allowing the conversion.
    tensorstore::ArrayView<void> untyped = allocated;
    PrintCSVArray(untyped.array_view());
  }

  // Arrays can be of other types as well:
  {
    std::cout << std::endl << "Array<X>" << std::endl;

    // Uninitialized...
    auto allocated = tensorstore::AllocateArray<X>({3, 3});

    // Explicitly initialized...
    //
    // FIXME: std::vector<> and other containers allow using a constant value to
    // initialize the array.
    //
    // FIXME: it might be nice to be able to use std::fill() on
    // tensorstore::ArrayView.
    tensorstore::InitializeArray(allocated);

    PrintCSVArray(allocated.array_view());
  }

  // We can allocate arrays and then cast them as untyped.
  {
    std::cout << std::endl << "Dynamic Array" << std::endl;

    // NOTE: To get the untyped result from AllocateArray, we have to cast off
    // the type from dtype_v, otherwise the templates will deduce the result
    // type to be Array<int, 2>.
    auto untyped = tensorstore::AllocateArray(
        {5, 5}, tensorstore::c_order, tensorstore::default_init,
        static_cast<tensorstore::DataType>(tensorstore::dtype_v<int>));

    if (tensorstore::ArraysHaveSameShapes(tensorstore::MakeArrayView(two_dim),
                                          untyped)) {
      tensorstore::CopyArray(tensorstore::MakeArrayView(two_dim), untyped);
    }
    PrintCSVArray(untyped.array_view());

    // FIXME: How do we reinterpret_cast ArrayView<int> to ArrayView<short>?
  }

  // FIXME: Add example of dynamic-rank arrays as well.

  // Array with filter operations.
  {
    std::cout << std::endl << "Transformed Arrays" << std::endl;

    auto array_ref = tensorstore::MakeArrayView(three_dim);
    PrintCSVArray(array_ref);

    // IndexArraySlice requires an array to store the indices,
    // thus the raw variant is currently not allowed.
    //
    // ::tensorstore::Dims(2).IndexArraySlice({2, 0})

    std::vector<tensorstore::Index> desired_channels = {2, 0};
    auto indices = tensorstore::UnownedToShared(
        tensorstore::MakeArrayView(desired_channels));

    auto channels = array_ref |
                    ::tensorstore::Dims(2).IndexArraySlice(indices) |
                    ::tensorstore::Materialize();

    PrintCSVArray(channels.value());
  }

  // FIXME: How do I cast a dense rank-2 array to a rank-1 array?
  // There is no mechanism to reshape arrays, though a transform
  // could make views of arrays.

  return 0;
}
