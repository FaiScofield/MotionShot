

add_executable(test_GMM GMM.cpp)
target_link_libraries(test_GMM ${OpenCV_LIBS} ${LIB_MotionShot})
