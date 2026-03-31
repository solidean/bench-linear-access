cmake_minimum_required(VERSION 3.25)
project(bench-linear-access CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench-linear-access src/main.cc)
target_include_directories(bench-linear-access PRIVATE src)
