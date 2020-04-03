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

#ifndef TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
#define TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_

namespace tensorstore {

class SourceLocation {
 public:
  SourceLocation(const char* file_name, int line)
      : file_name_(file_name), line_(line) {}

  const char* file_name() const { return file_name_; }
  int line() const { return line_; }

 private:
  const char* file_name_;
  int line_;
};

#define TENSORSTORE_LOC (::tensorstore::SourceLocation(__FILE__, __LINE__))
#define TENSORSTORE_LOC_CURRENT_DEFAULT_ARG = TENSORSTORE_LOC

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_SOURCE_LOCATION_H_
