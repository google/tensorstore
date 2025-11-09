#include <iostream>
#include "tensorstore/tensorstore.h"

// This simple program just verifies the binary links and can run.
// Linking against the netCDF driver target ensures its static registration runs.
int main() {
  std::cout << "netCDF driver linked. Smoke OK." << std::endl;
  return 0;
}
