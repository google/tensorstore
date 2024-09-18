
get_filename_component(_nasm_compiler_barename "${CMAKE_ASM_NASM_COMPILER}" NAME)
if (_nasm_compiler_barename STREQUAL "yasm")
  message(WARNING "CMake found YASM assembler. Please install 'nasm' instead.")
endif()
unset(_nasm_compiler_barename)


# nasm_library(@rules_nasm_test_repo//:asm_library)
add_library(CMakeProject_asm_library)
target_sources(CMakeProject_asm_library PRIVATE "${TEST_SRCDIR}/a.asm")
target_include_directories(CMakeProject_asm_library PRIVATE "${TEST_SRCDIR}"
    "${TEST_SRCDIR}/include")
set_source_files_properties(
    "${TEST_SRCDIR}/a.asm"
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS "-w+all;-D__x86_64__;-felf64;-DELF;-DPIC")
add_library(CMakeProject::asm_library ALIAS CMakeProject_asm_library)

# cc_library(@rules_nasm_test_repo//:a)
add_library(CMakeProject_a)
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::asm_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_SOURCE_DIR}/a.cc")
add_library(CMakeProject::a ALIAS CMakeProject_a)
