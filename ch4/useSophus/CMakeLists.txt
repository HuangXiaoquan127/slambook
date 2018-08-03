cmake_minimum_required(VERSION 2.8)
project(useSophus)

set(CMAKE_BUILD_TYPE "Debug")

#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/camke_modules)
find_package(Sophus REQUIRED)
include_directories( ${Sophus_INCLUDE_DIRS} )
add_executable(useSophus useSophus.cpp)#需要放在目标链接之前，否则链接时找不到执行文件；
target_link_libraries( useSophus ${Sophus_LIBRARIES} )

#find_package(Eigen3 REQUIRED)
#include_directories( ${Eigen3_INCLUDE_DIRS} )
#include_directories( ${Eigen3_INCLUDE_DIR} )
#message(${EIGEN3_INCLUDE_DIRS})
#message(${EIGEN3_INCLUDE_DIR})
include_directories( "/usr/include/eigen3" )

message(!!! ${Sophus_DIR} !!!)
message(${Sophus_INCLUDE_DIRS})
message(${Sophus_LIBRARIES})
message(${Sophus_CONFIG})
