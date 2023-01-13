cmake_minimum_required(VERSION 3.0)

project(ConeDetection)

SET(CMAKE_PREFIX_PATH .;./opencv/build;./apriltag-3.2.0/build;)

include_directories(OpenCV_INCLUDE_DIRS)

#packages 
FIND_PACKAGE(OpenCV REQUIRED)
find_package(apriltag REQUIRED)

#executables 
add_executable(ConeDetection "./src/main.cpp")
TARGET_LINK_LIBRARIES(ConeDetection ${OpenCV_LIBS})

add_executable(AprilTagDetection "./src/apriltag.cpp")
target_link_libraries(AprilTagDetection ${OpenCV_LIBS})
target_link_libraries(AprilTagDetection apriltag)

# add_library(opencv SHARED IMPORTED)
# set_target_properties(opencv PROPERTIES 
#     IMPORTED_IMPLIB ${OPENCV_PATH}/build/x64/vc16/lib/opencv_world470.lib
#     IMPORTED_LOCATION ${OPENCV_PATH}/build/x64/vc16/bin/opencv_world470.dll
# )


