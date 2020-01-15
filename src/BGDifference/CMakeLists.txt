add_executable(test_BGD main.cpp)
target_link_libraries(test_BGD
    ${LIB_BGDIFF}
    ${OpenCV_LIBS}
    ${LIB_MotionShot}
)
