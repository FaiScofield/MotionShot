#add_executable(test_detector_bgd test_detector_bgd.cpp)
#target_link_libraries(test_detector_bgd ${LIB_MotionShot})

#add_executable(test_detector test_detector.cpp)
#target_link_libraries(test_detector ${LIB_MotionShot})

#add_executable(test_detector_denseFlow test_detector_denseFlow.cpp)
#target_link_libraries(test_detector_denseFlow ${LIB_MotionShot})

#add_executable(test_detector_sparseFlow test_detector_sparseFlow.cpp)
#target_link_libraries(test_detector_sparseFlow ${LIB_MotionShot})

#add_executable(test_detector_cv test_detector_cv.cpp)
#target_link_libraries(test_detector_cv ${LIB_MotionShot})


#add_executable(test_detector_vibe test_detector_vibe.cpp)
#target_link_libraries(test_detector_vibe ${LIB_MotionShot})

#add_executable(test_detector_vibe+ test_detector_vibe+.cpp)
#target_link_libraries(test_detector_vibe+ ${LIB_MotionShot})


add_executable(test_blender test_blender.cpp)
target_link_libraries(test_blender ${LIB_MotionShot})

add_executable(test_stitcher test_stitcher.cpp)
target_link_libraries(test_stitcher ${LIB_MotionShot})

add_executable(test_stitcher_detail test_stitcher_detail.cpp)
target_link_libraries(test_stitcher_detail ${LIB_MotionShot})


#add_executable(test_tracker test_tracker.cpp)
#target_link_libraries(test_tracker ${LIB_MotionShot})


#add_executable(test_shoter_static test_shoter_static.cpp)
#target_link_libraries(test_shoter_static ${LIB_MotionShot})

#add_executable(test_shoter_full test_shoter_full.cpp)
#target_link_libraries(test_shoter_full ${LIB_MotionShot})

add_executable(test_shoter_static_lasiesta test_shoter_static_lasiesta.cpp)
target_link_libraries(test_shoter_static_lasiesta ${LIB_MotionShot})

add_executable(test_shoter_static_huawei test_shoter_static_huawei.cpp)
target_link_libraries(test_shoter_static_huawei ${LIB_MotionShot})



#add_executable(test_overlappedAreaBlending test_overlappedAreaBlending.cpp)
#target_link_libraries(test_overlappedAreaBlending ${LIB_MotionShot})


#add_executable(test_aip_bodyseg test_aip_bodyseg.cpp)
#target_link_libraries(test_aip_bodyseg ${LIB_MotionShot} ${JSONCPP_LIBRARIES} -lssl -lcrypto -lcurl)
