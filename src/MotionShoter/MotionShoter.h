#ifndef MOTION_SHOT_H
#define MOTION_SHOT_H

#include "precompiled.h"
#include "ImageBlender/cvBlenders.h"
#include "ImageStitcher/ImageStitcher.h"
#include "Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h"

#include <map>
#include <memory>

#include <opencv2/video/video.hpp>

namespace ms
{

class MotionShot
{
public:
    enum Status {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3,
        ERR_FORE_DETECT_FAIL = 4,
        ERR_BLEND_FAIL = 5,
        ERR_BAD_ARGUMENTS = -1
    };

    MotionShot();

    inline Mat getBaseFrame() const { return baseFrame_.clone(); }
    inline vector<Mat> getForeground() const { return foregrounds_; }


    Status setInputs(InputArrayOfArrays imgs, InputArrayOfArrays maks = cv::noArray());

    Status run();

    Status estimateTransform();

    Status warpImages();

    Status detectorForeground();

    Status compose();


    void detectorForegroundWithNN(InputArray src, InputArray mask, OutputArray dst);

private:
    void prepare(const vector<Point>& corners, const vector<Size>& sizes);
    void prepare(Rect dst_roi);
    void makeBorderForWarpedImages();

    void foreMaskFilter(InputArray src, OutputArray dst); //! TODO

    int checkOverlappedArea(const vector<Rect>&); //! TODO
    Status composeWithOverlapped();
    Status composeWithoutOverlapped();

    // baidu aip
    std::unique_ptr<aip::Bodyanalysis> bodyAnalyst_;
    std::map<string, string> bodySegOptions_;

    // custom class
    Ptr<ImageStitcher> stitcher_;
    Ptr<cvBlender> blender_;
    Ptr<cv::BackgroundSubtractorMOG2> detector_;

    // images
    vector<Mat> inputImages_, inputMasks_;
//    vector<Mat> scaledImages_, scaledMasks_;
    vector<Mat> warpedImages_, warpedMasks_;
    vector<Point> corners_/*, scaledCorners_*/;
    vector<Size> inputSizes_, warpedSize_, scaledSize_/*, scaledWarpedSize_*/;

    vector<Mat> foregrounds_, foregroundMasks_;
    vector<Mat> homograpies_;
    vector<Rect> foregroundRects_;

    Mat baseFrame_, baseFrameMask_, pano_;

    size_t numImgs_, bfi_;  // base frame index
    vector<size_t> indices_;

    Mat dstImg_, dstMask_;
    Rect dstROI_;
};

}  // namespace ms

#endif
