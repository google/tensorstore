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

#include <algorithm>
#include <iostream>

#include "absl/functional/function_ref.h"
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/status.h"

using ::tensorstore::Index;

// ApplyKernel applies the convolution kernel `kernel` to the
// input image `in`.
//
// This is not intended to be an efficient convolution implementation,
// but merely a demonstration of the tensorstore::Array api.
tensorstore::SharedArray<int, 2> ApplyKernel(
    const tensorstore::ArrayView<const int, 2> in,
    const tensorstore::ArrayView<const double, 2> kernel) {
  // Compute bounds for the offset.
  // FIXME: It's akward that we cannot do this:
  //   std::array<Index, 2> k(kernel.shape());
  //
  std::array<Index, 2> k;
  for (size_t i = 0; i < 2; i++) {
    k[i] = kernel.shape()[i] / 2;
  }

  // Our output has the same shape as our input (zero-based).
  const auto shape = in.shape();
  auto dest = tensorstore::AllocateArray<int>(shape);

  // The mean filter iterates over a transformed array, sums the
  // values, and then stores the mean into dest.
  tensorstore::IterateOverIndexRange(
      in.shape(), [&](tensorstore::span<const Index, 2> outer) {
        double acc = 0;

        // FIXME: Maybe we should have an IterateOverIndexRangeWithOrigin,
        // which will allow specifying an origin as well as the shape.
        tensorstore::IterateOverIndexRange(
            kernel.shape(), [&](tensorstore::span<const Index, 2> inner) {
              std::array<Index, 2> s{outer[0] + inner[0] - k[0],
                                     outer[1] + inner[1] - k[1]};

              // FIXME: How do we check whether the indices are valid?
              //
              // There is no mechanism to check whether a span<Index> is a
              // a valid position for any given array.
              if (s[0] >= 0 && s[0] < shape[0] &&  //
                  s[1] >= 0 && s[1] < shape[1]) {
                // In bounds, add the current value.
                //
                // NOTE: There are several ways to get this value, but the most
                // intuitive is not available.
                //   sum += in[s]
                //
                // These unintuitive alternatives are:
                //   sum += in(s)
                //   sum += in[s]()
                //   sum += in[s[0]][s[1]]()
                //
                acc += in(s)*kernel(inner);
              }
            });

        // Again, the most intutive way to write an array value
        // is not permitted:
        //
        //   dest[indices] = (sum / count);  // error: no viable overloaded '='
        //
        // The unintuitive ways are available:
        //   dest(indices) = (sum / count);
        //   dest[indices]() = (sum / count);
        //   *(dest[indices].pointer()) = (sum / count);
        //
        dest(outer) = acc;
      });

  return dest;
}

using AffineWarpGridFunction = absl::FunctionRef<void(
    tensorstore::BoxView<2>,  // mapping box for current block
    size_t,                   // y-strides of block data.
    double*  // block data, a y-major pointer to an array of x,y coordinates
    )>;

// AffineWarpGrid computes the grid mapping input pixels in the
// range [0..maxx,0..maxy] to an output space. For mapping from
// an output space to an image, see AffineWarpInverseGrid.
//
//  [ x']   [  m00  m01  m02  ] [ x ]   [ m00x + m01y + m02 ]
//  [ y'] = [  m10  m11  m12  ] [ y ] = [ m10x + m11y + m12 ]
//
// Since x, y typically have a range of 0..n, we can simplify the
// grid construction by precomputing the m00x, m10x, m01y, m11y
//
// Real use cases should use opencv or similar to achieve this.
void AffineWarpGrid(size_t xmax, size_t ymax,
                    tensorstore::span<const double, 6> M,
                    AffineWarpGridFunction fn) {
  constexpr size_t kBlockSize = 32;
  double mx[kBlockSize * 2];
  double yx[kBlockSize * kBlockSize * 2];

  size_t bxstep = std::min(kBlockSize, xmax);
  size_t bystep = std::min(kBlockSize, ymax);

  // Loop over the input range from [0,0],[xmax,ymax], subdivided into
  // simple blocks.
  for (size_t y = 0; y < ymax; y += bystep) {
    for (size_t x = 0; x < xmax; x += bxstep) {
      size_t bx = std::min(bxstep, xmax - x);
      size_t by = std::min(bystep, ymax - y);

      for (size_t x1 = 0; x1 < bx; x1++) {
        mx[x1 * 2] = M[0] * (x1 + x);      // m00x
        mx[x1 * 2 + 1] = M[3] * (x1 + x);  // m10x
      }

      for (size_t y1 = 0; y1 < by; y1++) {
        double m01y = M[1] * (y1 + y);
        double m11y = M[4] * (y1 + y);

        auto* ptr = &yx[kBlockSize * 2 * y1];
        for (size_t x1 = 0; x1 < bx; x1++) {
          *(ptr++) = mx[x1 * 2] + m01y + M[2];      // x
          *(ptr++) = mx[x1 * 2 + 1] + m11y + M[5];  // y
        }
      }

      fn(tensorstore::BoxView({static_cast<Index>(x), static_cast<Index>(y)},
                              {static_cast<Index>(bx), static_cast<Index>(by)}),
         kBlockSize, yx);
    }
  }
}

// AffineWarpInverseGrid computes the inverse mapping from AffineWarpGrid,
// so it can be used to map from a destination image to a souce image.
void AffineWarpInverseGrid(size_t xmax, size_t ymax,
                           tensorstore::span<const double, 6> M,
                           AffineWarpGridFunction fn) {
  double inv[6];
  memcpy(inv, M.data(), sizeof(inv));

  // Invert the matrix to apply it to the output.
  double d = inv[0] * inv[4] - inv[1] * inv[3];
  d = (d != 0) ? 1.0 / d : 0;
  double a11 = inv[4] * d;
  double a22 = inv[0] * d;
  inv[0] = a11;
  inv[1] *= -d;
  inv[3] *= -d;
  inv[4] = a22;
  double b1 = -inv[0] * inv[2] - inv[1] * inv[5];
  double b2 = -inv[3] * inv[2] - inv[4] * inv[5];
  inv[2] = b1;
  inv[5] = b2;

  return AffineWarpGrid(xmax, ymax, inv, std::move(fn));
}

//  [ x']   [  m00  m01  m02  ] [ x ]   [ m00x + m01y + m02 ]
//  [ y'] = [  m10  m11  m12  ] [ y ] = [ m10x + m11y + m12 ]
inline std::pair<double, double> AffinePoint(
    size_t x, size_t y, tensorstore::span<const double, 6> M) {
  return {M[0] * x + M[1] * y + M[2], M[3] * x + M[4] * y + M[5]};
}

template <typename T>
T clamp(T x, T l, T h) {
  return (x < l) ? l : (x >= h) ? h : x;
}

tensorstore::SharedArray<int, 2> AffineWarp(
    const tensorstore::ArrayView<const int, 2> in,
    tensorstore::span<const double, 6> M) {
  [[maybe_unused]] const auto origin =
      AffinePoint(in.origin()[0], in.origin()[1], M);
  assert(origin.first == 0);
  assert(origin.second == 0);
  const auto shape = AffinePoint(in.shape()[0], in.shape()[1], M);

  auto output = tensorstore::AllocateArray<int>(
      {static_cast<Index>(shape.first), static_cast<Index>(shape.second)},
      tensorstore::ContiguousLayoutOrder::c, tensorstore::value_init);

  // Use the inverse grid to generate a mapping from our output to
  // our input; otherwise we'd use a forward mapping.
  AffineWarpInverseGrid(
      output.shape()[0], output.shape()[1], M,
      [&](tensorstore::BoxView<2> box, size_t stride, double* data) {
        auto o = box.origin();
        auto s = box.shape();

        Index lx = in.origin()[0];
        Index ly = in.origin()[1];
        Index ux = in.origin()[0] + in.shape()[0] - 1;
        Index uy = in.origin()[1] + in.shape()[1] - 1;
        for (Index y1 = 0; y1 < s[1]; y1++) {
          auto* ptr = &data[stride * 2 * y1];
          for (Index x1 = 0; x1 < s[0]; x1++) {
            // Use a simple NN function to compute the source pixel.
            double x = *ptr++;
            double y = *ptr++;
            auto ix = clamp(static_cast<Index>(x + 0.5), lx, ux);
            auto iy = clamp(static_cast<Index>(y + 0.5), ly, uy);
            output(o[0] + x1, o[1] + y1) = in(ix, iy);
          }
        }
      });

  return output;
}

template <typename T, tensorstore::DimensionIndex N>
void PrintCSVArray(tensorstore::ArrayView<T, N> data) {
  // Iterate over the shape of the data array, which gives us one
  // reference for every element.
  //
  // There is a streaming operator already, but the output is
  // this is equvalent to:
  // for (int x = 0; x < data.shape()[0]; x++)
  //  for (int y = 0; y < data.shape()[1]; y++) {
  //     ... body ...
  //  }
  //
  // The builtin streaming operator outputs data in C++ array initialization
  // syntax: {{0, 0}, {1, 0}}, but we want to dump a CSV-formatted
  // array.
  const auto max = *data.shape().rbegin() - 1;

  tensorstore::IterateOverIndexRange(
      data.shape(), [&](tensorstore::span<const Index, 2> idx) {
        // FIXME: It's somewhat surprising that when we stream out a rank-0
        // array acquired through data[] it behaves like the native type, but
        // when we try to use the value from data[], it does not.
        std::cout << data[idx];
        if (*idx.rbegin() == max) {
          std::cout << std::endl;
        } else {
          std::cout << "\t";
        }
      });
}

int main(int argc, char** argv) {
  tensorstore::SharedArray<int, 2> result;

  // Construct an arbitrary 5x5 image.
  const int image[5][5] =   //
      {{0, 0, 0, 0, 0},     //
       {0, 72, 53, 60, 0},  //
       {0, 76, 56, 65, 0},  //
       {0, 88, 78, 82, 0},  //
       {0, 0, 0, 0, 0}};

  // identity kernel
  const double identity_k[3][3] =  //
      {{0, 0, 0},                  //
       {0, 1, 0},                  //
       {0, 0, 0}};

  result = ApplyKernel(tensorstore::MakeArrayView(image),
                       tensorstore::MakeArrayView(identity_k));

  std::cout << result << std::endl << std::endl;

  // edge detection
  const double edge_k[3][3] = {{-1, -1, -1}, {-1, 8, -1}, {-1, -1, -1}};

  result = ApplyKernel(tensorstore::MakeArrayView(image),
                       tensorstore::MakeArrayView(edge_k));

  // FIXME: It would be nice if there was an implicit conversion between
  // tensorstore::SharedArray<T, N> and tensorstore::ArrayView<T, N>.
  //
  // FIXME: tensorstore::MakeArrayView(SharedArray<T, N>{}) fails.
  PrintCSVArray(result.array_view());
  std::cout << std::endl;

  // Scale by 2 Kernel.
  const double M[6] = {
      2., 0,  0,  //
      0,  2., 0,  //
  };

  auto new_result = AffineWarp(tensorstore::MakeArrayView(image), M);
  PrintCSVArray(new_result.array_view());
  std::cout << std::endl;
}
