cmake_minimum_required(VERSION 3.0)

project(ConeDetection)

add_executable(ConeDetection "./src/main.cpp")

SET(CMAKE_PREFIX_PATH .;./opencv/build;)

FIND_PACKAGE(OpenCV REQUIRED world)
TARGET_LINK_LIBRARIES(ConeDetection ${OpenCV_LIBS})
include_directories(OpenCV_INCLUDE_DIRS)
# add_library(opencv SHARED IMPORTED)
# set_target_properties(opencv PROPERTIES 
#     IMPORTED_IMPLIB ${OPENCV_PATH}/build/x64/vc16/lib/opencv_world470.lib
#     IMPORTED_LOCATION ${OPENCV_PATH}/build/x64/vc16/bin/opencv_world470.dll
# )


