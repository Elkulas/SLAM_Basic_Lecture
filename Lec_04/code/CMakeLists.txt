cmake_minimum_required(VERSION 3.0.0)
project(lec4)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

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

include_directories(
${OpenCV_INCLUDE_DIRS}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)


add_executable(undistort_image
src/undistort_image.cpp)
target_link_libraries(undistort_image
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)

add_executable(disparity
src/disparity.cpp)
target_link_libraries(disparity
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)

add_executable(gaussnewton
src/gaussnewton.cpp)
target_link_libraries(gaussnewton
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)


include(CPack)
