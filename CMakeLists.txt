cmake_minimum_required(VERSION 2.8.3)
project(MotionShot)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)   # Release, Debug
endif()
message("===> CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

IF((CMAKE_BUILD_TYPE MATCHES "Debug") OR (CMAKE_BUILD_TYPE MATCHES "DEBUG") OR (CMAKE_BUILD_TYPE MATCHES "debug"))
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

# -march=native 参数不可用, 会使g2o出错.
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -ggdb") # -g
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall -march=native")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations -Wno-unused-variable")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

# Boost
find_package(Boost REQUIRED COMPONENTS filesystem)
include_directories(${Boost_INCLUDE_DIRS})
if(Boost_FOUND)
    message("===> Boost Information: Version: ${Boost_VERSION}" )
    message("===> Boost Information: Include Dir: ${Boost_INCLUDE_DIRS}" )
    message("===> Boost Information: Libs Dir: ${Boost_LIBRARIES}" )
else()
    message("===> Boost Not Found!!!" )
endif()

## OpenCV
set(OpenCV_DIR /home/vance/opt/opencv-4.2.0/build)
find_package(OpenCV 4)
if (OpenCV_FOUND)
    add_definitions(-DUSE_OPENCV4)
else()
    set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
    find_package(OpenCV 3 REQUIRED)
    add_definitions(-DUSE_OPENCV3)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND)
    message("===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/MotionShot
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/MotionDetector
    ${PROJECT_SOURCE_DIR}/src/MotionTracker
    ${PROJECT_SOURCE_DIR}/src/MotionShoter
    ${PROJECT_SOURCE_DIR}/src/ImageStitching
    ${PROJECT_SOURCE_DIR}/Thirdparty/aip-cpp-sdk-0.8.5
    /usr/include/eigen3 # for optical flow
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# Lib
#aux_source_directory(${PROJECT_SOURCE_DIR}/src/MotionDetector DETECTOR_SRC_FILES)
#aux_source_directory(${PROJECT_SOURCE_DIR}/src/MotionTracker  TRACKER_SRC_FILES)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/MotionShoter   SHOTER_SRC_FILES)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ImageStitcher  STITCHER_SRC_FILES)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/ImageBlender   BLENDER_SRC_FILES)
add_library(${PROJECT_NAME} SHARED
#    ${DETECTOR_SRC_FILES}
#    ${TRACKER_SRC_FILES}
    ${SHOTER_SRC_FILES}
    ${STITCHER_SRC_FILES}
    ${BLENDER_SRC_FILES}
    ${PROJECT_SOURCE_DIR}/src/utility.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${JSONCPP_LIBRARIES}
    -lssl -lcrypto -lcurl
)
set(LIB_MotionShot ${PROJECT_NAME})

add_subdirectory(Thirdparty/aip-cpp-sdk-0.8.5)
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(app)
