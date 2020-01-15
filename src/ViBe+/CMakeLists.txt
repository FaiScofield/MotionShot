

add_executable(test_ViBe+ main.cpp)
target_link_libraries(test_ViBe+
    ${LIB_VIBEPLUS}
    ${OpenCV_LIBS}
    ${LIB_MotionShot}
)

