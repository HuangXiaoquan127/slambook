#声明要求的cmake 最低版本
cmake_minimum_required( VERSION 2.8 )

#声明一个cmake 工程
project( HelloSLAM )

#添加一个可执行程序
#语法： add_executable(程序名源代码文件)
include_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable( helloSLAM helloSLAM.cpp )
#add_library( hello ${PROJECT_SOURCE_DIR}/lib/libHelloSLAM.cpp )
#add_library(hello_shared SHARED ${PROJECT_SOURCE_DIR}/lib/libHelloSLAM.cpp)
#add_executable(useHello useHello.cpp)
#target_link_libraries(useHello ${PROJECT_SOURCE_DIR}/build/lib/libHelloSLAM.so)
set(CMAKE_BUILD_TYPE "Debug")


#ADD_SUBDIRECTORY(lib)
