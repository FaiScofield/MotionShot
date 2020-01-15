

add_executable(test_FD main.cpp)
target_link_libraries(test_FD
    ${LIB_FDIFF}
    ${OpenCV_LIBS}
    ${LIB_MotionShot}
)
