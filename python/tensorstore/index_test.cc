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

#include "python/tensorstore/index.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/index.h"

namespace {

TEST(OptionallyImplicitIndexReprTest, Basic) {
  using tensorstore::kImplicit;
  using tensorstore::internal_python::OptionallyImplicitIndexRepr;

  EXPECT_EQ("None", OptionallyImplicitIndexRepr(kImplicit));
  EXPECT_EQ("3", OptionallyImplicitIndexRepr(3));
  EXPECT_EQ("-3", OptionallyImplicitIndexRepr(-3));
}

TEST(IndexVectorReprTest, Basic) {
  using tensorstore::Index;
  using tensorstore::kImplicit;
  using tensorstore::internal_python::IndexVectorRepr;

  for (bool subscript : {false, true}) {
    EXPECT_EQ("None", IndexVectorRepr(kImplicit, /*implicit=*/true, subscript));
    for (bool implicit : {false, true}) {
      EXPECT_EQ("1", IndexVectorRepr(1, implicit, subscript));
      EXPECT_EQ("-1", IndexVectorRepr(-1, implicit, subscript));
    }
  }

  for (bool implicit : {false, true}) {
    EXPECT_EQ("[1,2,3]", IndexVectorRepr(std::vector<Index>{1, 2, 3}, implicit,
                                         /*subscript=*/false));
    EXPECT_EQ("1,2,3", IndexVectorRepr(std::vector<Index>{1, 2, 3}, implicit,
                                       /*subscript=*/true));
    EXPECT_EQ("[]", IndexVectorRepr(std::vector<Index>{}, implicit,
                                    /*subscript=*/false));
    EXPECT_EQ("()", IndexVectorRepr(std::vector<Index>{}, implicit,
                                    /*subscript=*/true));
  }
  EXPECT_EQ("[1,2,None]", IndexVectorRepr(std::vector<Index>{1, 2, kImplicit},
                                          /*implicit=*/true,
                                          /*subscript=*/false));
  EXPECT_EQ("1,2,None", IndexVectorRepr(std::vector<Index>{1, 2, kImplicit},
                                        /*implicit=*/true,
                                        /*subscript=*/true));
}

TEST(ToIndexVectorOrScalarContainerTest, Basic) {
  using tensorstore::Index;
  using tensorstore::kImplicit;
  using tensorstore::internal_python::IndexVectorOrScalarContainer;
  using tensorstore::internal_python::OptionallyImplicitIndex;
  using tensorstore::internal_python::ToIndexVectorOrScalarContainer;

  EXPECT_EQ(
      IndexVectorOrScalarContainer{Index{3}},
      ToIndexVectorOrScalarContainer(OptionallyImplicitIndex{3}, kImplicit));
  EXPECT_EQ(IndexVectorOrScalarContainer{3},
            ToIndexVectorOrScalarContainer(OptionallyImplicitIndex{3}, 4));
  EXPECT_EQ(
      IndexVectorOrScalarContainer{Index{3}},
      ToIndexVectorOrScalarContainer(OptionallyImplicitIndex{kImplicit}, 3));
  EXPECT_EQ(IndexVectorOrScalarContainer{kImplicit},
            ToIndexVectorOrScalarContainer(OptionallyImplicitIndex{kImplicit},
                                           kImplicit));
  EXPECT_EQ(IndexVectorOrScalarContainer{std::vector<Index>({1, 2, 3})},
            ToIndexVectorOrScalarContainer(
                std::vector<OptionallyImplicitIndex>{
                    OptionallyImplicitIndex{1},
                    OptionallyImplicitIndex{2},
                    OptionallyImplicitIndex{3},
                },
                kImplicit));
  EXPECT_EQ(IndexVectorOrScalarContainer{std::vector<Index>({1, 2, kImplicit})},
            ToIndexVectorOrScalarContainer(
                std::vector<OptionallyImplicitIndex>{
                    OptionallyImplicitIndex{1},
                    OptionallyImplicitIndex{2},
                    OptionallyImplicitIndex{kImplicit},
                },
                kImplicit));
  EXPECT_EQ(IndexVectorOrScalarContainer{std::vector<Index>({1, 2, 3})},
            ToIndexVectorOrScalarContainer(
                std::vector<OptionallyImplicitIndex>{
                    OptionallyImplicitIndex{1},
                    OptionallyImplicitIndex{2},
                    OptionallyImplicitIndex{kImplicit},
                },
                3));
}

}  // namespace
