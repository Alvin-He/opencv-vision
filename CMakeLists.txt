cmake_minimum_required(VERSION 3.0)
#windows being dumb about linking 
SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)

project(ConeDetection)

SET(CMAKE_PREFIX_PATH .;./opencv/build;./apriltag-3.2.0/build;)

include_directories(OpenCV_INCLUDE_DIRS)

#packages 
FIND_PACKAGE(OpenCV REQUIRED)
# find_package(apriltag REQUIRED)

#executables 
add_executable(ConeDetection "./src/main.cpp")
TARGET_LINK_LIBRARIES(ConeDetection ${OpenCV_LIBS})

add_executable(Cuda "./src/cuda.cpp")
target_link_libraries(Cuda ${OpenCV_LIBS})

