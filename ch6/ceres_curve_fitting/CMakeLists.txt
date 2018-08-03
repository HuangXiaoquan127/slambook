cmake_minimum_required(VERSION 2.8)
project(ceres_curve_fitting)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})
#include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(curve_fitting main.cpp)
target_link_libraries(curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS})
#target_link_libraries(curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})



message("@@@@ 
OpenCV_DIRS = ${OpenCV_DIRS}
OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}
CERES_INCLUDE_DIRS = ${CERES_INCLUDE_DIRS}
CERES_DIRS = ${CERES_DIRS}

CERES_LIBRARIES = ${CERES_LIBRARIES}
CERES_LIBS = ${CERES_LIBS}
OpenCV_LIBS = ${OpenCV_LIBS}
OpenCV_LIBRARIES = ${OpenCV_LIBRARIES} 
@@@@")
