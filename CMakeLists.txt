cmake_minimum_required(VERSION 3.25)
project(bench-linear-access CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put all build products into predictable folders inside the build dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable(bench-linear-access src/main.cc)
target_include_directories(bench-linear-access PRIVATE src)

if(MSVC)
    target_compile_options(bench-linear-access PRIVATE /arch:AVX2)
else()
    target_compile_options(bench-linear-access PRIVATE -mavx2)
endif()
