add_executable(test_detector_bgd test_detector_bgd.cpp)
target_link_libraries(test_detector_bgd ${LIB_MotionShot})

add_executable(test_detector_fd test_detector_fd.cpp)
target_link_libraries(test_detector_fd ${LIB_MotionShot})

add_executable(test_detector_mog2 test_detector_mog2.cpp)
target_link_libraries(test_detector_mog2 ${LIB_MotionShot})

add_executable(test_detector_vibe test_detector_vibe.cpp)
target_link_libraries(test_detector_vibe ${LIB_MotionShot})

add_executable(test_detector_vibe+ test_detector_vibe+.cpp)
target_link_libraries(test_detector_vibe+ ${LIB_MotionShot})


add_executable(test_blender test_blender.cpp)
target_link_libraries(test_blender ${LIB_MotionShot})

add_executable(test_stitcher test_stitcher.cpp)
target_link_libraries(test_stitcher ${OpenCV_LIBS})



#add_executable(test_tracker test_tracker.cpp)
#target_link_libraries(test_tracker ${LIB_MotionShot})
