cmake_minimum_required(VERSION 3.5.0)
project(Neural_Network VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(Neural_Network main.cpp)
target_compile_features(Neural_Network PUBLIC cxx_std_23)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
target_compile_options(Neural_Network PUBLIC -O3 -mavx2 -mfma -msse4.1 -msse4.2)
endif()
