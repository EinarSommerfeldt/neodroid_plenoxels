cmake_minimum_required(VERSION 3.10)

project(mcubes)

set (CMAKE_CXX_STANDARD 17)
set(OpenCV_DIR C:/Users/einarjso/opencv/build)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp src/mcubes.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE "C:/Users/einarjso/eigen-3.4.0")