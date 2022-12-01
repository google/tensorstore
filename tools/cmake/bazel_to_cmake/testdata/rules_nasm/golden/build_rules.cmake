add_library(CMakeProject_asm_library)
target_sources(CMakeProject_asm_library PRIVATE "${TEST_DIRECTORY}/a.asm")
target_include_directories(CMakeProject_asm_library PRIVATE "${TEST_DIRECTORY}" "${TEST_DIRECTORY}/include")
set_source_files_properties(
    "${TEST_DIRECTORY}/a.asm"
    PROPERTIES
      LANGUAGE ASM_NASM
      COMPILE_OPTIONS "-w+all;-D__x86_64__;-felf64;-DELF;-DPIC")
add_library(CMakeProject::asm_library ALIAS CMakeProject_asm_library)


add_library(CMakeProject_a)
target_sources(CMakeProject_a PRIVATE
        "${TEST_DIRECTORY}/a.cc")
set_property(TARGET CMakeProject_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::asm_library"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
add_library(CMakeProject::a ALIAS CMakeProject_a)
