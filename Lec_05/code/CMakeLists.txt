cmake_minimum_required(VERSION 3.0.0)
project(lec5 VERSION 0.1.0)

include(CTest)
enable_testing()

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


find_package( OpenCV 3.1 REQUIRED )
message(STATUS "OpenCV include path:::${OpenCV_INCLUDE_DIRS}")

Set(Sophus_DIR "/home/jjj/CodeSrc/Sophus_No_Template/build")
find_package(Sophus REQUIRED)
message(STATUS "Sophus include path: ${Sophus_INCLUDE_DIRS}")
message(STATUS "Sophus librares path: ${Sophus_LIBRARY_DIR}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "/usr/include/eigen3/"
    ${Sophus_INCLUDE_DIRS}
)

add_executable( computeORB src/computeORB.cpp )
target_link_libraries( computeORB ${OpenCV_LIBS} )

add_executable( E2Rt src/E2Rt.cpp )
target_link_libraries( E2Rt 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
 )

 add_executable( GN-BA src/GN-BA.cpp )
 target_link_libraries( GN-BA 
    ${OpenCV_LIBS}
    ${Sophus_LIBRARIES}
 )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
