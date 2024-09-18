
# genrule(@cc_includes_test_repo//parent:c_inc)
file(MAKE_DIRECTORY "${TEST_BINDIR}/parent/child")
add_custom_command(
  OUTPUT
    "${TEST_BINDIR}/parent/child/c.inc"
  
  COMMAND echo "// c.inc" > "${TEST_BINDIR}/parent/child/c.inc"
  VERBATIM
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_custom_target(genrule__CMakeProject_parent_c_inc DEPENDS
    "${TEST_BINDIR}/parent/child/c.inc")
add_library(CMakeProject_parent_c_inc INTERFACE)
target_sources(CMakeProject_parent_c_inc INTERFACE
    "${TEST_BINDIR}/parent/child/c.inc")
target_include_directories(CMakeProject_parent_c_inc INTERFACE
    "${PROJECT_BINARY_DIR}")
add_dependencies(CMakeProject_parent_c_inc genrule__CMakeProject_parent_c_inc)

# cc_library(@cc_includes_test_repo//parent:a)
add_library(CMakeProject_parent_a)
set_property(TARGET CMakeProject_parent_a PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a PUBLIC cxx_std_17)
add_dependencies(CMakeProject_parent_a "CMakeProject_parent_c_inc")
target_sources(CMakeProject_parent_a PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a ALIAS CMakeProject_parent_a)

# cc_library(@cc_includes_test_repo//parent:a_dot)
add_library(CMakeProject_parent_a_dot)
set_property(TARGET CMakeProject_parent_a_dot PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_dot PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_dot PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/parent>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/parent>")
target_include_directories(CMakeProject_parent_a_dot PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_dot PUBLIC cxx_std_17)
add_dependencies(CMakeProject_parent_a_dot "CMakeProject_parent_c_inc")
target_sources(CMakeProject_parent_a_dot PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_dot ALIAS CMakeProject_parent_a_dot)

# cc_library(@cc_includes_test_repo//parent:a_child)
add_library(CMakeProject_parent_a_child)
set_property(TARGET CMakeProject_parent_a_child PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_child PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_child PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/parent/child>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/parent/child>")
target_include_directories(CMakeProject_parent_a_child PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_child PUBLIC cxx_std_17)
target_sources(CMakeProject_parent_a_child PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_child ALIAS CMakeProject_parent_a_child)

# cc_library(@cc_includes_test_repo//parent:a_parent)
add_library(CMakeProject_parent_a_parent)
set_property(TARGET CMakeProject_parent_a_parent PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_parent PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_parent PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/parent>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/parent>")
target_include_directories(CMakeProject_parent_a_parent PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_parent PUBLIC cxx_std_17)
add_dependencies(CMakeProject_parent_a_parent "CMakeProject_parent_c_inc")
target_sources(CMakeProject_parent_a_parent PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_parent ALIAS CMakeProject_parent_a_parent)

# cc_library(@cc_includes_test_repo//parent:a_strip_child)
add_library(CMakeProject_parent_a_strip_child)
set_property(TARGET CMakeProject_parent_a_strip_child PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_strip_child PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_strip_child PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/parent/child>")
target_include_directories(CMakeProject_parent_a_strip_child PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_strip_child PUBLIC cxx_std_17)
target_sources(CMakeProject_parent_a_strip_child PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_strip_child ALIAS CMakeProject_parent_a_strip_child)

# cc_library(@cc_includes_test_repo//parent:a_strip_parent)
add_library(CMakeProject_parent_a_strip_parent)
set_property(TARGET CMakeProject_parent_a_strip_parent PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_strip_parent PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_strip_parent PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_strip_parent PUBLIC cxx_std_17)
add_dependencies(CMakeProject_parent_a_strip_parent "CMakeProject_parent_c_inc")
target_sources(CMakeProject_parent_a_strip_parent PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_strip_parent ALIAS CMakeProject_parent_a_strip_parent)

# cc_library(@cc_includes_test_repo//parent:a_prefix_child)
add_library(CMakeProject_parent_a_prefix_child)
set_property(TARGET CMakeProject_parent_a_prefix_child PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_parent_a_prefix_child PUBLIC
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_parent_a_prefix_child PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_parent_a_prefix_child PUBLIC cxx_std_17)
target_sources(CMakeProject_parent_a_prefix_child PRIVATE
        "${PROJECT_SOURCE_DIR}/parent/child/a.cc")
add_library(CMakeProject::parent_a_prefix_child ALIAS CMakeProject_parent_a_prefix_child)

# cc_binary(@cc_includes_test_repo//:a)
add_executable(CMakeProject_a "")
add_executable(CMakeProject::a ALIAS CMakeProject_a)
target_link_libraries(CMakeProject_a PUBLIC
        "CMakeProject::parent_a"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a PUBLIC cxx_std_17)
target_sources(CMakeProject_a PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_dot)
add_executable(CMakeProject_a_dot "")
add_executable(CMakeProject::a_dot ALIAS CMakeProject_a_dot)
target_link_libraries(CMakeProject_a_dot PUBLIC
        "CMakeProject::parent_a_dot"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_dot PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_dot PUBLIC cxx_std_17)
target_sources(CMakeProject_a_dot PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_child)
add_executable(CMakeProject_a_child "")
add_executable(CMakeProject::a_child ALIAS CMakeProject_a_child)
target_link_libraries(CMakeProject_a_child PUBLIC
        "CMakeProject::parent_a_child"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_child PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_child PUBLIC cxx_std_17)
target_sources(CMakeProject_a_child PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_parent)
add_executable(CMakeProject_a_parent "")
add_executable(CMakeProject::a_parent ALIAS CMakeProject_a_parent)
target_link_libraries(CMakeProject_a_parent PUBLIC
        "CMakeProject::parent_a_parent"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_parent PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_parent PUBLIC cxx_std_17)
target_sources(CMakeProject_a_parent PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_strip_child)
add_executable(CMakeProject_a_strip_child "")
add_executable(CMakeProject::a_strip_child ALIAS CMakeProject_a_strip_child)
target_link_libraries(CMakeProject_a_strip_child PUBLIC
        "CMakeProject::parent_a_strip_child"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_strip_child PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_strip_child PUBLIC cxx_std_17)
target_sources(CMakeProject_a_strip_child PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_strip_parent)
add_executable(CMakeProject_a_strip_parent "")
add_executable(CMakeProject::a_strip_parent ALIAS CMakeProject_a_strip_parent)
target_link_libraries(CMakeProject_a_strip_parent PUBLIC
        "CMakeProject::parent_a_strip_parent"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_strip_parent PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_strip_parent PUBLIC cxx_std_17)
target_sources(CMakeProject_a_strip_parent PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")

# cc_binary(@cc_includes_test_repo//:a_prefix_child)
add_executable(CMakeProject_a_prefix_child "")
add_executable(CMakeProject::a_prefix_child ALIAS CMakeProject_a_prefix_child)
target_link_libraries(CMakeProject_a_prefix_child PUBLIC
        "CMakeProject::parent_a_prefix_child"
        "Threads::Threads"
        "m")
target_include_directories(CMakeProject_a_prefix_child PRIVATE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")
target_compile_features(CMakeProject_a_prefix_child PUBLIC cxx_std_17)
target_sources(CMakeProject_a_prefix_child PRIVATE
        "${PROJECT_SOURCE_DIR}/b.cc")
