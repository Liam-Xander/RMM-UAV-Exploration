cmake_minimum_required(VERSION 2.8.3)
project(plan_env)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
# Existing cuda flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O3 --expt-relaxed-constexpr")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  cv_bridge
  message_filters
)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)


find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "########## Found openmp ##########") 
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${OPENMP_C_FLAGS})
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OPENMP_CXX_FLAGS})
else()
    message(FATAL_ERROR "Openmp not found!")
endif()


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_env
 CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

# CUDA include directory
include_directories(${CUDA_INCLUDE_DIRS})

include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})


# Set compilation options
set(CUDA_CUBLAS_LIBRARIES "/usr/local/cuda-11.6/lib64/libcublas.so")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wall")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 -O3")

# RTX4060 GPU
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")
# RTX4080 Super GPU
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_86")
# 1650 GPU
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_75")

# Compile CUDA library, containing extract_surface_points.cu file
cuda_add_library(cuda_kernels
  src/random_mapping_cuda.cu 
)

# Link cuBLAS library
target_link_libraries(cuda_kernels
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
)

add_library( plan_env 
    src/sdf_map.cpp 
    src/obj_predictor.cpp 
    src/edt_environment.cpp 
    src/raycast.cpp
    src/map_ros.cpp
    src/random_mapping_method.cpp
    )
target_link_libraries( plan_env
    cuda_kernels
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    )  

add_executable(obj_generator
    src/obj_generator.cpp 
    )
    
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(obj_generator 
    plan_env
    ${catkin_LIBRARIES})
