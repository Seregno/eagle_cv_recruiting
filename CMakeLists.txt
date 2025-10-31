cmake_minimum_required(VERSION 3.10)
project(cone_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(cone_detector src/main.cpp)
target_link_libraries(cone_detector ${OpenCV_LIBS})

add_executable(color_detector src/color_detector.cpp)
target_link_libraries(color_detector ${OpenCV_LIBS})
