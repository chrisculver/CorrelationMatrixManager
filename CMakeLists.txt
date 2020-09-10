cmake_minimum_required(VERSION 3.14)
project("Correlation Matrix Manager")

SET(GCC_COMPILE_FLAGS "-std=c++17 -O3")

include_directories(INCLUDE $ENV{HOME}/Code/Libraries/spdlog/include/)
file(GLOB SOURCES "SRC/*.cpp" "SRC/UTILS/*.cpp" "SRC/OPS/*.cpp" "SRC/DIAGS/*.cpp" main.cpp)

add_executable(compute_correlation_matrix ${SOURCES})

#include GWU-QCD library
find_library(SPDLOG libspdlog.a $ENV{HOME}/Code/Libraries/spdlog/build/)
target_link_libraries(compute_correlation_matrix ${SPDLOG})


set_target_properties(
  compute_correlation_matrix PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "${WARNING_FLAGS}"
 )

# Set up the documentation
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMAKE" ${CMAKE_MODULE_PATH})
add_subdirectory("DOCS")
