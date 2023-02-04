cmake_minimum_required(VERSION 3.0)
#windows being dumb about linking 
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

project(ConeDetection)

SET(CMAKE_PREFIX_PATH .;./opencv/build;./apriltag-3.2.0/build;)

include_directories(OpenCV_INCLUDE_DIRS)

#packages 
FIND_PACKAGE(OpenCV REQUIRED)
# find_package(apriltag REQUIRED)

# libriaries

## apriltag
# SET(APRILTAG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/apriltag)
# SET(APRILTAG_SOURCES 
#     ${APRILTAG_DIR}/cpp/apriltag.cpp
#     ${APRILTAG_DIR}/cpp/detector.cpp

#     ${APRILTAG_DIR}/public/dependencies.h

#     ${APRILTAG_DIR}/public/apriltag.h
#     ${APRILTAG_DIR}/public/detector.h
# ) 
# SET(APRILTAG_PUBLIC_DIR ${APRILTAG_DIR}/public)
# SET(APRILTAG_PRIVATE_DIR ${APRILTAG_DIR}/private)

# add_library(Apriltag SHARED ${APRILTAG_SOURCES})

# #include directories 
# target_include_directories(Apriltag PRIVATE ${APRILTAG_PRIVATE_DIR})
# target_include_directories(Apriltag PUBLIC $<BUILD_INTERFACE:${APRILTAG_PUBLIC_DIR}>)
# target_link_libraries(Apriltag ${OpenCV_LIBS})

#executables 
add_executable(ConeDetection "./src/main.cpp")
TARGET_LINK_LIBRARIES(ConeDetection ${OpenCV_LIBS})

add_executable(Cuda "./src/cuda.cpp")
target_link_libraries(Cuda ${OpenCV_LIBS})

# add_executable(AprilTagDetection "./src/apriltag.cpp")
# target_link_libraries(AprilTagDetection ${OpenCV_LIBS})
# target_link_libraries(AprilTagDetection Apriltag)

# add_library(opencv SHARED IMPORTED)
# set_target_properties(opencv PROPERTIES 
#     IMPORTED_IMPLIB ${OPENCV_PATH}/build/x64/vc16/lib/opencv_world470.lib
#     IMPORTED_LOCATION ${OPENCV_PATH}/build/x64/vc16/bin/opencv_world470.dll
# )


