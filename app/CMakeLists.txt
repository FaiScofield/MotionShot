include_directories(
    # BAIDU AIP API
    ${PROJECT_SOURCE_DIR}/Thirdparty/
    ${PROJECT_SOURCE_DIR}/Thirdparty/aip-cpp-sdk-0.8.5
)

#find_package(PkgConfig REQUIRED)
#pkg_check_modules(JSONCPP jsoncpp)
#include_directories(${JSONCPP_INCLUDE_DIRS})
#link_libraries(${JSONCPP_LIBRARIES})

#find_package(CURL REQUIRED)
#include_directories(${CURL_INCLUDE_DIRS})
#link_libraries(${CURL_LIBRARIES})
#if (CURL_FOUND)
#    message("=====> CURL Information include: ${CURL_INCLUDE_DIRS}")
#    message("=====> CURL Information libs: ${CURL_LIBRARIES}")
#else()
#    message("=====> CURL not found!!!!!")
#endif()

#find_package(Crypt REQUIRED)
#include_directories(${CRYPTO_INCLUDE_DIRS})
#link_libraries(${CRYPTO_LIBRARIES})


# Use this to get a Foreground Mask
add_executable(grabcut grabcut.cpp)
target_link_libraries(grabcut ${OpenCV_LIBS} )

add_executable(readAndSave readAndSave.cpp)
target_link_libraries(readAndSave ${OpenCV_LIBS} ${LIB_MotionShot})

#add_executable(foregroundSeg_baidu foregroundSeg_baidu.cpp)
#target_link_libraries(foregroundSeg_baidu
#    ${OpenCV_LIBS}
#    ${LIB_MotionShot}
#    ${JSONCPP_LIBRARIES}
#    ${CURL_LIBRARIES}
##    /usr/lib/x86_64-linux-gnu/libcrypto.so.1.0.0
##    -ljsoncpp
#    )
