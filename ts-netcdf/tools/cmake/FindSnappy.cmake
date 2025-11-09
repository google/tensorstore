

# SNAPPY_INCLUDE_DIR - where to find snappy-c.h, etc.
# SNAPPY_LIBRARY - List of libraries when using snappy.
# SNAPPY_FOUND - True if snappy found.
find_path(SNAPPY_INCLUDE_DIR
  NAMES snappy-c.h)

find_library(SNAPPY_LIBRARY
  NAMES snappy)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(snappy DEFAULT_MSG SNAPPY_LIBRARY SNAPPY_INCLUDE_DIR)

if (DEFINED SNAPPY_LIBRARY)
  set(SNAPPY_LIBRARIES ${SNAPPY_LIBRARY})
endif()

mark_as_advanced(
  SNAPPY_LIBRARY
  SNAPPY_LIBRARIES
  SNAPPY_INCLUDE_DIR)
