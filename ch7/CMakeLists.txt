cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#find_package(OpenCV 3.1 REQUIRED)
find_package(OpenCV 3.1.0 EXACT)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

add_executable(feature_extraction feature_extraction.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS} ${OpenCV_LIB_SUFFIX})

add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})

add_executable(triangulation triangulation.cpp)
target_link_libraries(triangulation ${OpenCV_LIBS})


add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d2d 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
add_executable(pose_estimation_3d2d_exer6 pose_estimation_3d2d_exer6.cpp)
target_link_libraries(pose_estimation_3d2d_exer6
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
add_executable(pose_estimation_3d2d_exer6_2 pose_estimation_3d2d_exer6_2.cpp)
target_link_libraries(pose_estimation_3d2d_exer6_2
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
add_executable(pose_estimation_3d2d_orig pose_estimation_3d2d_orig.cpp)
target_link_libraries(pose_estimation_3d2d_orig
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
    

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
#add_executable(pose_estimation_3d3d pose_estimation_3d3d_orig.cpp)
target_link_libraries(pose_estimation_3d3d
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
add_executable(pose_estimation_3d3d_orig pose_estimation_3d3d_orig.cpp)
target_link_libraries(pose_estimation_3d3d_orig
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
add_executable(pose_estimation_3d3d_exer7 pose_estimation_3d3d_exer7.cpp)
target_link_libraries(pose_estimation_3d3d_exer7
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension)
    
