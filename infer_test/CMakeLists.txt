cmake_minimum_required(VERSION 3.10)
#set(CMAKE_C_COMPILER /usr/bin/gcc)
#set(CMAKE_CXX_COMPILER /usr/bin/g++)
project(headinferproject)
# set(CMAKE_CXX_STANDARD 17 )
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(SRC src/infer.cu src/rail.cpp src/yolo.cu)
enable_language(CUDA)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS} /usr/local/cuda-11.4/include)

add_library(headinfer SHARED ${SRC})
target_link_libraries(headinfer nvinfer cudart ${OpenCV_LIBS} stdc++fs)

add_executable(a ${SRC} src/main.cpp)
target_link_libraries(a nvinfer cudart ${OpenCV_LIBS} stdc++fs)
