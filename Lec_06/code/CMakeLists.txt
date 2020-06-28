cmake_minimum_required(VERSION 3.0.0)
project(lec6 VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(OpenCV_DIR "/home/jjj/CodeSrc/opencv-3.4.8/build")
find_package(OpenCV 3.0 QUIET)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "Eigen3 include path: ${EIGEN3_INCLUDE_DIR}")

find_package(Pangolin REQUIRED)
message(STATUS "Pangolin include path: ${Pangolin_INCLUDE_DIRS}")

Set(Sophus_DIR "/home/jjj/CodeSrc/Sophus_No_Template/build")
find_package(Sophus REQUIRED)
message(STATUS "Sophus include path: ${Sophus_INCLUDE_DIRS}")
message(STATUS "Sophus libraries path: ${Sophus_LIBRARY_DIR}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_executable(direct_method src/direct_method.cpp)
target_link_libraries( direct_method
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${Sophus_LIBRARIES}
)

add_executable(optical_flow src/optical_flow.cpp)
target_link_libraries( optical_flow
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}  
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
