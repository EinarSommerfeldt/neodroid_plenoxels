cmake_minimum_required(VERSION 3.10)

project(dist_loss)

set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR C:/Users/einar/opencv/build/)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE "C:/Users/einar/opencv/build/include")
target_include_directories(${PROJECT_NAME} PRIVATE "C:/Users/einar/eigen-3.4.0")
target_include_directories(${PROJECT_NAME} PRIVATE "C:/Users/einar/eigen-3.4.0/unsupported")