

add_executable(test_Vibe main.cpp)
target_link_libraries(test_Vibe
    ${LIB_VIBE}
    ${OpenCV_LIBS}
    ${LIB_MotionShot}
)
