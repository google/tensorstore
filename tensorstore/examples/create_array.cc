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
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"

using tensorstore::Index;

namespace {

struct X {
  float a;
  int b;

  friend std::ostream& operator<<(std::ostream& os, const X& x) {
    os << "<" << x.a << ", " << x.b << ">";
    return os;
  }
};

template <typename T, tensorstore::DimensionIndex N>
void PrintCSVArray(tensorstore::ArrayView<T, N> data) {
  if (data.rank() == 0) {
    std::cout << data;
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
  auto element_rep = data.data_type();

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
  std::cout << s;
}
}  // namespace

int main(int argc, char** argv) {
  // How do we create arrays? Let's try this multiple different ways.

  std::cout << std::endl << "Native C++ arrays" << std::endl;

  // tensorstore::ArrayView can be created from native C++ arrays of arbitrary
  // dimensions. The number of dimensions in an array is the rank of the array.

  {
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

    std::cout << std::endl;
  }

  // tensorstore::Array and tensorstore::ArrayView can be of any primitive type.
  {
    const double two_dim[5][5] =  //
        {{0, 0, 0, 0, 0},         //
         {0, .72, .53, .60, 0},   //
         {0, .76, .56, .65, 0},   //
         {0, .88, .78, .82, 0},   //
         {0, 0, 0, 0, 0}};

    auto array_ref = tensorstore::MakeArrayView(two_dim);
    PrintCSVArray(array_ref);

    // The rank and shape of an array can be inspected:
    std::cout << "rank=" << array_ref.rank() << " shape=" << array_ref.shape();
    std::cout << std::endl;
  }

  // image is an array of rank 2.
  const int image[5][5] =   //
      {{0, 0, 0, 0, 0},     //
       {0, 72, 53, 60, 0},  //
       {0, 76, 56, 65, 0},  //
       {0, 88, 78, 82, 0},  //
       {0, 0, 0, 0, 0}};
  PrintCSVArray(tensorstore::MakeArrayView(image));
  std::cout << std::endl;

  // FIXME: We cannot create an array ref from a C++ std::array type.
  /*
  std::array<std::array<int, 3>, 3> arr = {{{5, 8, 2}, {8, 3, 1}, {5, 3, 9}}};
  PrintCSVArray(tensorstore::MakeArrayView(image));
  */

  // tensorstore also includes the concept of rank-0 arrays, or Scalars.
  // these are created from a single value.
  {
    std::cout << "Scalar Array" << std::endl;
    PrintCSVArray(tensorstore::MakeScalarArray(123).array_view());
    std::cout << std::endl;
  }

  // In addition to the tensorstore::ArrayView types, tensorstore can allocate
  // arrays dynamically of specified types, rank, and dimensions.
  {
    std::cout << "Allocated Array" << std::endl;
    auto allocated = tensorstore::AllocateArray<int>({5, 5});
    if (tensorstore::ArraysHaveSameShapes(tensorstore::MakeArrayView(image),
                                          allocated)) {
      // Copy uses the same order as std::copy, rather than the reversed
      // order from std::memcpy.
      tensorstore::CopyArray(tensorstore::MakeArrayView(image), allocated);
    }
    PrintCSVArray(allocated.array_view());
    std::cout << std::endl;

    // FIXME: It seems like we might want to require this conversion to be
    // explicit rather than silently allowing the conversion.
    tensorstore::ArrayView<void> untyped = allocated;
    PrintCSVArray(untyped.array_view());
    std::cout << std::endl;
  }

  // Arrays can be of other types as well:
  {
    std::cout << "Array<X>" << std::endl;

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
    std::cout << std::endl;
  }

  // We can allocate arrays and then cast them as untyped.
  {
    std::cout << "Dynamic Array" << std::endl;

    // NOTE: To get the untyped result from AllocateArray, we have to cast off
    // the type from DataTypeOf, otherwise the templates will deduce the result
    // type to be Array<int, 2>.
    auto untyped = tensorstore::AllocateArray(
        {5, 5}, tensorstore::c_order, tensorstore::default_init,
        static_cast<tensorstore::DataType>(tensorstore::DataTypeOf<int>()));

    if (tensorstore::ArraysHaveSameShapes(tensorstore::MakeArrayView(image),
                                          untyped)) {
      tensorstore::CopyArray(tensorstore::MakeArrayView(image), untyped);
    }
    PrintCSVArray(untyped.array_view());
    std::cout << std::endl;

    // FIXME: How do we reinterpret_cast ArrayView<int> to ArrayView<short>?
  }

  // FIXME: Add example of dynamic-rank arrays as well.

  // FIXME: How do I cast a dense rank-2 array to a rank-1 array?
  // There is no mechanism to reshape arrays, though a transform
  // could make views of arrays.
}
