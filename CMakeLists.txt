cmake_minimum_required(VERSION 3.10)
project(apriltag_demo)

find_package(OpenCV REQUIRED)

add_executable(apriltag_demo main.cpp)
target_link_libraries(apriltag_demo ${OpenCV_LIBS})
target_include_directories(apriltag_demo PRIVATE ${OpenCV_INCLUDE_DIRS})
