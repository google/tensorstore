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

// Extracts a slice of a volumetric dataset, outputtting it as a 2d jpeg image.
//
// extract_slice --output_file=/tmp/foo.jpg --input_spec=...

#include <stdint.h>

#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorstore/tensorstore.h"
#include "tensorstore/context.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/compression/blosc.h"
//#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/kvstore/memory/memory_key_value_store.h"
#include "tensorstore/open.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

using ::tensorstore::Context;
using ::tensorstore::StrCat;
using tensorstore::Index;

template <typename Array>
void PrintCSVArray(Array&& data) {
  if (data.rank() == 0) {
    std::cout << data << std::endl;
    return;
  }
  size_t sum = 0;
  for (int x = 0; x < data.shape()[0]; x++)
    for (int y = 0; y < data.shape()[1]; y++) {
      sum += data(x,y);
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
  // std::string s;

  // tensorstore::IterateOverIndexRange(  //
  //     data.shape(), [&](tensorstore::span<const Index> idx) {
        
  //       element_rep->append_to_string(&s, data[idx].pointer());
  //       if (*idx.rbegin() == max) {
  //         std::cout << s << std::endl;

  //         s.clear();
  //       } else {
  //         s.append("\t");
  //       }
  //     });
  // std::cout << s << std::endl;
  std::cout << "sum is " << sum << std::endl;
}


void read_ometiff_data()
{
  tensorstore::Context context = Context::Default();
  TENSORSTORE_CHECK_OK_AND_ASSIGN(auto store, tensorstore::Open({{"driver", "ometiff"},
                            {"cache_pool", {{"total_bytes_limit", 10000000}}},
                            {"kvstore", {{"driver", "tiff"},
                                         {"path", "/mnt/hdd8/axle/data/bfio_test_images/r001_c001_z000.ome.tif"}}
                            }},
                            context,
                            tensorstore::OpenMode::open,
                            //tensorstore::RecheckCached{true},
                            //tensorstore::RecheckCachedData{false},
                            tensorstore::ReadWriteMode::read).result());



 
  auto array = tensorstore::AllocateArray<tensorstore::uint16_t>({10, 10});
  auto array2 = tensorstore::AllocateArray<tensorstore::uint16_t>({10, 10});
  for (int i=0; i<100; i++){
    tensorstore::Read(store | 
                tensorstore::AllDims().TranslateTo(0) |
                tensorstore::Dims(0).ClosedInterval(0,9) |
                tensorstore::Dims(1).ClosedInterval(0,9) ,
                array).value();
    PrintCSVArray(array);
    tensorstore::Read(store | 
                tensorstore::AllDims().TranslateTo(0) |
                tensorstore::Dims(0).ClosedInterval(1024,1033) |
                tensorstore::Dims(1).ClosedInterval(1024,1033) ,
                array).value();
    PrintCSVArray(array);

  }
  // tensorstore::Read(store | 
  //               tensorstore::AllDims().TranslateTo(0) |
  //               tensorstore::Dims(0).ClosedInterval(0,9) |
  //               tensorstore::Dims(1).ClosedInterval(0,9) ,
  //               array).value();

  // PrintCSVArray(array);


  // tensorstore::Read(store | 
  //               tensorstore::AllDims().TranslateTo(0) |
  //               tensorstore::Dims(0).ClosedInterval(0,9) |
  //               tensorstore::Dims(1).ClosedInterval(0,9) ,
  //               array2).value();
}




int main(int argc, char** argv) {
  read_ometiff_data();
 return 0;
}