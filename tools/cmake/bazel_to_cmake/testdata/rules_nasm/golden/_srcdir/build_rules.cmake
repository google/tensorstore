
get_filename_component(_nasm_compiler_barename "${CMAKE_ASM_NASM_COMPILER}" NAME)
if (_nasm_compiler_barename STREQUAL "yasm")
  message(WARNING "CMake found YASM assembler. Please install 'nasm' instead.")
endif()
unset(_nasm_compiler_barename)


# nasm_library(@rules_nasm_test_repo//:asm_library)
add_custom_command(
  OUTPUT "${TEST_BINDIR}/_nasm/asm_library/38192f0a/a.asm.o"
  DEPENDS "${TEST_SRCDIR}/a.asm"
  COMMAND ${CMAKE_ASM_NASM_COMPILER}
          -f ${CMAKE_ASM_NASM_OBJECT_FORMAT}
          ${CMAKE_ASM_NASM_FLAGS}
          "-I${TEST_SRCDIR}" "-I${TEST_BINDIR}" "-I${TEST_SRCDIR}/include" "-I${TEST_BINDIR}/include" "-w+all" "-D__x86_64__" "-DELF" "-DPIC"
          -o "${TEST_BINDIR}/_nasm/asm_library/38192f0a/a.asm.o"
          "${TEST_SRCDIR}/a.asm"
  COMMENT "Assembling NASM source ${TEST_SRCDIR}/a.asm"
)
set_source_files_properties("${TEST_BINDIR}/_nasm/asm_library/38192f0a/a.asm.o"
  PROPERTIES GENERATED TRUE)

# cc_library(@rules_nasm_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_BINARY_DIR}/_nasm/asm_library/38192f0a/a.asm.o"
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)

# nasm_cc_library(@rules_nasm_test_repo//:cc_library)
add_library(CMakeProject_cc_library)
target_sources(CMakeProject_cc_library PRIVATE
    "${TEST_SRCDIR}/a.asm")
target_include_directories(CMakeProject_cc_library PRIVATE
    "${TEST_BINDIR}"
    "${TEST_BINDIR}/include"
    "${TEST_SRCDIR}"
    "${TEST_SRCDIR}/include")
set_source_files_properties(
    "${TEST_SRCDIR}/a.asm"
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS "-s;-w-macro-params-legacy;-w-orphan-labels")

add_library(CMakeProject_cc_library.alwayslink INTERFACE)
if (BUILD_SHARED_LIBS)
  target_link_libraries(CMakeProject_cc_library.alwayslink INTERFACE "$<LINK_LIBRARY:bazel_to_cmake_needed_library,CMakeProject_cc_library>")
else ()
  target_link_libraries(CMakeProject_cc_library.alwayslink INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,CMakeProject_cc_library>")
endif()
add_library(CMakeProject::cc_library ALIAS CMakeProject_cc_library.alwayslink)
