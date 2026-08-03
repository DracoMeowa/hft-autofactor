# hft-autofactor — root build file.
#
# Determinism contract (docs/architecture.md): -O2, no fast-math, no FMA
# contraction, so factor doubles are bitwise reproducible across builds.
cmake_minimum_required(VERSION 3.16)
project(hft_autofactor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Determinism flags. Applied globally so tests and the engine agree bit-for-bit.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O2 -fno-fast-math -ffp-contract=off)
elseif(MSVC)
  add_compile_options(/O2 /fp:precise)
endif()

find_package(ZLIB REQUIRED)

enable_testing()
add_subdirectory(cpp)
