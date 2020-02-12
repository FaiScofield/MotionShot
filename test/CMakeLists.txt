#add_executable(test_detector_bgd test_detector_bgd.cpp)
#target_link_libraries(test_detector_bgd ${LIB_MotionShot})

add_executable(test_detector test_detector.cpp)
target_link_libraries(test_detector ${LIB_MotionShot})

add_executable(test_detector_DenseFlow_cv test_detector_DenseFlow_cv.cpp)
target_link_libraries(test_detector_DenseFlow_cv ${LIB_MotionShot})


#add_executable(test_detector_vibe test_detector_vibe.cpp)
#target_link_libraries(test_detector_vibe ${LIB_MotionShot})

#add_executable(test_detector_vibe+ test_detector_vibe+.cpp)
#target_link_libraries(test_detector_vibe+ ${LIB_MotionShot})


add_executable(test_blender test_blender.cpp)
target_link_libraries(test_blender ${LIB_MotionShot})

add_executable(test_stitcher test_stitcher.cpp)
target_link_libraries(test_stitcher ${LIB_MotionShot})

add_executable(test_shoter_simple test_shoter_simple.cpp)
target_link_libraries(test_shoter_simple ${LIB_MotionShot})

add_executable(test_shoter_full test_shoter_full.cpp)
target_link_libraries(test_shoter_full ${LIB_MotionShot})

#add_executable(test_tracker test_tracker.cpp)
#target_link_libraries(test_tracker ${LIB_MotionShot})
