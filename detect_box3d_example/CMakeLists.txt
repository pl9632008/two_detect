cmake_minimum_required(VERSION 3.10)
project(test)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
)
include_directories(${catkin_INCLUDE_DIRS})
add_executable(a src/main.cpp)
target_link_libraries(a ${catkin_LIBRARIES})
