#aux_source_directory(MotionDetector DETECTOR_SRC_FILES)
#aux_source_directory(MotionTracker  TRACKER_SRC_FILES)
#aux_source_directory(MotionShoter   SHOTER_SRC_FILES)
#aux_source_directory(ImageStitching STITCHING_SRC_FILES)

##add_subdirectory(MotionDetector)
##add_subdirectory(MotionTracker)
##add_subdirectory(MotionShoter)
##add_subdirectory(ImageStitching)

#add_library(${PROJECT_NAME} SHARED
#    ${DETECTOR_SRC_FILES}
#    ${TRACKER_SRC_FILES}
#    ${SHOTER_SRC_FILES}
#    ${STITCHING_SRC_FILES}
#)
#target_link_libraries(${PROJECT_NAME}
#    ${OpenCV_LIBS}
#    ${Boost_LIBRARIES}
#    # ${CSPARSE_LIBRARY}
#    # ${CHOLMOD_LIBRARIES}
#    # ${G2O_LIBS}
#)
#set(LIB_MotionShot ${PROJECT_NAME})


# Use this to get a Foreground Mask
add_executable(grabcut grabcut.cpp)
target_link_libraries(grabcut ${OpenCV_LIBS} )
