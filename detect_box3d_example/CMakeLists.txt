cmake_minimum_required(VERSION 3.10)
project(test)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} include)

add_executable(a src/main.cpp)
target_link_libraries(a ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(projectcloud src/projectCloud.cpp)
target_link_libraries(projectcloud ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})



message("---catkin_INCLUDE_DIRS---")
message(${catkin_INCLUDE_DIRS})

message("---catkin_LIBRARIES---")
message(${catkin_LIBRARIES})
