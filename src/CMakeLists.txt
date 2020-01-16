
set(LIB_MOG2_SOURCE
    ./BackgroundExtractor/BS_MOG2_CV.h
    ./BackgroundExtractor/BS_MOG2_CV.cpp)
add_library(MOG2 SHARED ${LIB_MOG2_SOURCE})
target_link_libraries(MOG2 ${OpenCV_LIBS} ${LIB_MotionShot})
set(LIB_MOG2 MOG2)

# BGDifference，高斯背景差分法动态链接库生成
#SET(LIB_BGDIFF_SOURCE
#    ./BGDifference/BGDifference.h
#    ./BGDifference/BGDifference.cpp)
#ADD_LIBRARY(BGDiff SHARED ${LIB_BGDIFF_SOURCE})
#TARGET_LINK_LIBRARIES(BGDiff ${OpenCV_LIBS})
#SET(LIB_BGDIFF BGDiff)

# FramesDifference，差分法动态链接库生成
SET(LIB_FDIFF_SOURCE
    ./FramesDifference/FramesDifference.h
    ./FramesDifference/FramesDifference.cpp)
ADD_LIBRARY(FDiff SHARED ${LIB_FDIFF_SOURCE})
TARGET_LINK_LIBRARIES(FDiff ${OpenCV_LIBS} ${LIB_MotionShot})
SET(LIB_FDIFF FDiff)

## ViBe动态链接库生成
#SET(LIB_VIBE_SOURCE
#    ./ViBe/Vibe.h
#    ./ViBe/Vibe.cpp)
#ADD_LIBRARY(vibe SHARED ${LIB_VIBE_SOURCE})
#TARGET_LINK_LIBRARIES(vibe ${OpenCV_LIBS})
#SET(LIB_VIBE vibe)

## ViBe+动态链接库生成
#SET(LIB_VIBEPLUS_SOURCE
#    ./ViBe+/ViBePlus.h
#    ./ViBe+/ViBePlusMacro.h
#    ./ViBe+/ViBePlus.cpp)
#ADD_LIBRARY(vibe+ SHARED ${LIB_VIBEPLUS_SOURCE})
#TARGET_LINK_LIBRARIES(vibe+ ${OpenCV_LIBS})
#SET(LIB_VIBEPLUS vibe+)

#add_subdirectory(BGDifference)
add_subdirectory(BackgroundExtractor)
add_subdirectory(FramesDifference)
#add_subdirectory(ViBe)
#add_subdirectory(ViBe+)

add_executable(MotionTrack main.cpp)
target_link_libraries(MotionTrack
    ${LIB_MotionShot}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
