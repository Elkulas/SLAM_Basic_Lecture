cmake_minimum_required(VERSION 3.0.0)
project(draw VERSION 0.1.0)

include(CTest)
enable_testing()

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package(Pangolin REQUIRED)
message(STATUS "Pangolin include path: ${Pangolin_INCLUDE_DIRS}")

Set(Sophus_DIR "/home/jjj/CodeSrc/Sophus_No_Template/build")
find_package(Sophus REQUIRED)
message(STATUS "Sophus include path: ${Sophus_INCLUDE_DIRS}")
message(STATUS "Sophus libraries path: ${Sophus_LIBRARY_DIR}")

include_directories(
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_executable(draw draw_trajectory.cpp)

target_link_libraries(
    draw
    ${Pangolin_LIBRARIES}
    ${Sophus_LIBRARIES}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
