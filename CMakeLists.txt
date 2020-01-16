cmake_minimum_required(VERSION 2.8.3)
project(MotionShot)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)   # Release, Debug
endif()
message("===> CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# -march=native 参数不可用, 会使g2o出错.
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -ggdb") # -g
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-deprecated-declarations -Wno-unused-variable")

#set(CMAKE_AUTOMOC ON)
#set(CMAKE_AUTORCC ON)

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules/)

#IF(CMAKE_COMPILER_IS_GNUCXX)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O0 -g")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O0 -g")
#endif()

## Eigen
# find_package(Eigen REQUIRED)
# include_directories(${EIGEN_INCLUDE_DIRS})
# message("===> Eigen Include DIR: ${EIGEN_INCLUDE_DIRS}")

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
set(OpenCV_DIR /home/vance/opt/opencv-3.4.1/build)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND)
    message("===> OpenCV Information: Version: ${OpenCV_VERSION}" )
    message("===> OpenCV Information: Include Dir: ${OpenCV_INCLUDE_DIRS}" )
    message("===> OpenCV Information: Libs Dir: ${OpenCV_LIBS}" )
else()
    message("===> OpenCV Not Found!!!" )
endif()

# ## G2O
# find_package(CSparse REQUIRED)
# include_directories(${CSPARSE_INCLUDE_DIR})
# if(CSPARSE_FOUND)
#     message("===> CSparse Information: Include Dir: ${CSPARSE_INCLUDE_DIR}" )
#     message("===> CSparse Information: Libs Dir: ${CSPARSE_LIBRARY}" )
# else()
#     message("===> CSparse Not Found!!!")
# endif()
# find_package(Cholmod REQUIRED)
# include_directories(${CHOLMOD_INCLUDE_DIR})
# if(CHOLMOD_FOUND)
#     message("===> Cholmod Information: Include Dir: ${CHOLMOD_INCLUDE_DIR}" )
#     message("===> Cholmod Information: Libs Dir: ${CHOLMOD_LIBRARIES}" )
# else()
#     message("===> Cholmod Not Found!!!")
# endif()
# set(G2O_ROOT /home/vance/opt/g2o-20160424/install)
# find_package(G2O REQUIRED)
# include_directories(${G2O_INCLUDE_DIR})
# file(GLOB G2O_LIBS ${G2O_ROOT}/lib/*.so)
# # list(APPEND G2O_LIBS cxsparse cholmod)
# if (G2O_FOUND)
#     message("===> G2O Information: Root: ${G2O_ROOT}")
#     message("===> G2O Information: Include Dir: ${G2O_INCLUDE_DIR}" )
#     message("===> G2O Information: Libs Dir: ${G2O_LIBS}" )
# else()
#     message("===> G2O Not Found!!!" )
# endif()

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include/MotionShot
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/BGDifference
    ${PROJECT_SOURCE_DIR}/src/BackgroundExtractor
    ${PROJECT_SOURCE_DIR}/src/FramesDifference
    ${PROJECT_SOURCE_DIR}/src/ViBe
    ${PROJECT_SOURCE_DIR}/src/ViBe+
)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    include/MotionShot/BaseBackgroundSubtractor.h

    src/utility.cpp
    src/BaseBackgroundSubtractor.cpp
    src/MotionTracker.cpp
    src/MotionShot.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    # ${CSPARSE_LIBRARY}
    # ${CHOLMOD_LIBRARIES}
    # ${G2O_LIBS}
)
set(LIB_MotionShot ${PROJECT_NAME})

add_subdirectory(src)

