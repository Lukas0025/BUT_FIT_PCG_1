#
# @file      CMakeLists.txt
#
# @author    David Bayer \n
#            Faculty of Information Technology \n
#            Brno University of Technology \n
#            ibayer@fit.vutbr.cz
#
# @brief     PCG Assignment 1
#
# @version   2023
#
# @date      04 October   2023, 09:00 (created) \n
#

cmake_minimum_required(VERSION 3.8)

project(Assignment1 LANGUAGES C CXX CUDA)

set(COMPILE_CPU_VERSION              OFF)

set(CMAKE_CXX_STANDARD               17)
set(CMAKE_CXX_STANDARD_REQUIRED      ON)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native -flto")
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "Intel|IntelLLVM")
  if(${WIN32})
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /Od /debug:full")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=full /fast /QxHost")
  else()
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -debug full")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto=full -fast -xHost")
  endif()
elseif(${CMAKE_CXX_COMPILER_ID} MATCHES "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_DEBUG} /Ox /Mt /fp:fast /GL")
  set(CMAKE_CXX_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS_RELEASE} /Od /Mtd")
endif()

set(CMAKE_CUDA_STANDARD              17)
set(CMAKE_CUDA_STANDARD_REQUIRED     ON)
set(CMAKE_CUDA_ARCHITECTURES         61;70;75;80)
set(CMAKE_CUDA_HOST_COMPILER         ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_FLAGS_DEBUG           "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE         "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  find_package(OpenMP)
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} Commons)
link_directories(${CUDAToolkit_LIBRARY_DIR} ${HDF5_LIBRARY_DIRS})

# Gen
add_executable(gen Commons/gen.cpp)
target_link_libraries(gen HDF5::HDF5)

# Cpu
add_executable(nbodyCpu Cpu/main.cpp Cpu/nbody.cpp Commons/h5Helper.cpp)
if (OpenMP_CXX_FOUND)
  target_link_libraries(nbodyCpu HDF5::HDF5 OpenMP::OpenMP_CXX)
else()
  target_link_libraries(nbodyCpu HDF5::HDF5)
endif()

# Step0
add_executable(nbody0 Step0/main.cu Step0/nbody.cu Commons/h5Helper.cpp)
target_link_libraries(nbody0 HDF5::HDF5)

# Step1
add_executable(nbody1 Step1/main.cu Step1/nbody.cu Commons/h5Helper.cpp)
target_link_libraries(nbody1 HDF5::HDF5)

# Step2
add_executable(nbody2 Step2/main.cu Step2/nbody.cu Commons/h5Helper.cpp)
target_link_libraries(nbody2 HDF5::HDF5)

# Step3
add_executable(nbody3 Step3/main.cu Step3/nbody.cu Commons/h5Helper.cpp)
target_link_libraries(nbody3 HDF5::HDF5)

# Step4
add_executable(nbody4 Step4/main.cu Step4/nbody.cu Commons/h5Helper.cpp)
target_link_libraries(nbody4 HDF5::HDF5)
