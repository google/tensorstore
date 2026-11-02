
# cc_library(@rules_nasm_arm64_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# cc_library(@rules_nasm_arm64_test_repo//:cc_library_alwayslink)
add_library(CMakeProject_cc_library_alwayslink.alwayslink)
set_property(TARGET CMakeProject_cc_library_alwayslink.alwayslink PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_library_alwayslink.alwayslink PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_cc_library_alwayslink.alwayslink PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_cc_library_alwayslink.alwayslink PUBLIC cxx_std_17)
target_sources(CMakeProject_cc_library_alwayslink.alwayslink PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")

add_library(CMakeProject_cc_library_alwayslink INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries(CMakeProject_cc_library_alwayslink INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,CMakeProject_cc_library_alwayslink.alwayslink>")
else ()
  target_link_libraries(CMakeProject_cc_library_alwayslink INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,CMakeProject_cc_library_alwayslink.alwayslink>")
endif()
add_library(CMakeProject::cc_library_alwayslink ALIAS CMakeProject_cc_library_alwayslink)

# cc_library(@rules_nasm_arm64_test_repo//:cc_library_both)
add_library(CMakeProject_cc_library_both.alwayslink STATIC)
set_property(TARGET CMakeProject_cc_library_both.alwayslink PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_cc_library_both.alwayslink PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_cc_library_both.alwayslink PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_cc_library_both.alwayslink PUBLIC cxx_std_17)
target_sources(CMakeProject_cc_library_both.alwayslink PRIVATE
        "${PROJECT_BINARY_DIR}/bazel_to_cmake_empty_source.cc")

add_library(CMakeProject_cc_library_both INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries(CMakeProject_cc_library_both INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,CMakeProject_cc_library_both.alwayslink>")
else ()
  target_link_libraries(CMakeProject_cc_library_both INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,CMakeProject_cc_library_both.alwayslink>")
endif()
add_library(CMakeProject::cc_library_both ALIAS CMakeProject_cc_library_both)

# cc_library(@rules_nasm_arm64_test_repo//:cc_library_linkstatic)
add_library(CMakeProject_cc_library_linkstatic INTERFACE)
target_link_libraries(CMakeProject_cc_library_linkstatic INTERFACE
        "Threads::Threads"
        "m")
target_compile_features(CMakeProject_cc_library_linkstatic INTERFACE cxx_std_17)
add_library(CMakeProject::cc_library_linkstatic ALIAS CMakeProject_cc_library_linkstatic)
