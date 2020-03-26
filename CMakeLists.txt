cmake_minimum_required(VERSION 3.14)
project("Correlation Matrix Manager")

SET(GCC_COMPILE_FLAGS "-std=c++17 -O3")

include_directories(INCLUDE $ENV{HOME}/Code/Libraries/gwu-qcd/include)
file(GLOB SOURCES "SRC/*.cpp" main.cpp)

add_executable(compute_correlation_matrix ${SOURCES})

#include GWU-QCD library
find_library(GWUQCDlib libqcd.a $ENV{HOME}/Code/Libraries/gwu-qcd/lib/)
target_link_libraries(compute_correlation_matrix ${GWUQCDlib})

set_target_properties(
  compute_correlation_matrix PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  COMPILE_FLAGS "${WARNING_FLAGS}"
 )

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMAKE" ${CMAKE_MODULE_PATH})

add_subdirectory("DOCS")
