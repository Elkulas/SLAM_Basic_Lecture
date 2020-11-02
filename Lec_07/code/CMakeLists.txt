cmake_minimum_required(VERSION 3.0.0)
project(directBA VERSION 0.1.0)

include(CTest)
enable_testing()

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

set(OpenCV_DIR "/home/jjj/CodeSrc/opencv-3.4.8/build")
find_package(OpenCV 3.0 QUIET)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "Eigen3 include path: ${EIGEN3_INCLUDE_DIR}")

find_package(G2O REQUIRED)
message(STATUS "g2o include path: ${G2O_INCLUDE_DIR}")
message(STATUS "G2O lib path: ${G2O_LIB_DIRS}")

find_package(CSparse REQUIRED )
message(STATUS "CSPARSE include path: ${CSPARSE_INCLUDE_DIR}")

find_package(Pangolin REQUIRED)
message(STATUS "Pangolin include path: ${Pangolin_INCLUDE_DIRS}")

Set(Sophus_DIR "/home/jjj/CodeSrc/Sophus_No_Template/build")
find_package(Sophus REQUIRED)
message(STATUS "Sophus include path: ${Sophus_INCLUDE_DIRS}")
message(STATUS "Sophus libraries path: ${Sophus_LIBRARY_DIR}")

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear 
    g2o_types_icp g2o_types_slam2d g2o_types_sba g2o_types_slam3d g2o_core g2o_interface 
    g2o_solver_csparse g2o_solver_structure_only g2o_csparse_extension g2o_opengl_helper g2o_solver_dense 
    g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

include_directories(  
    ${G2O_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_executable(directBA src/directBA.cpp)
target_link_libraries(directBA
   ${CSPARSE_LIBRARY}
   ${G2O_LIBS}
   ${Pangolin_LIBRARIES}
   ${Sophus_LIBRARIES}
   ${OpenCV_LIBS}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)