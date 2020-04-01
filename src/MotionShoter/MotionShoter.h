#ifndef MOTION_SHOT_H
#define MOTION_SHOT_H

#include "precompiled.h"
#include "ImageBlender/cvBlenders.h"
#include "ImageStitcher/ImageStitcher.h"
#include "Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h"

#include <map>
#include <memory>

#include <opencv2/video.hpp>
#include <opencv2/bgsegm.hpp>

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

    UMat getBaseFrame() const { return baseFrame_.clone(); }
    vector<Mat> getForeground() const { return foregrounds_; }
    void setStitcher(Ptr<ImageStitcher> stitcher) { stitcher_ = stitcher; }
    void setBlender(Ptr<cvBlender> blender) { blender_ = blender; }

    Status setInputs(InputArrayOfArrays imgs, InputArrayOfArrays masks = cv::noArray());
    Status run();
    void getResult(OutputArray pano) { pano.assign(pano_); }

    Status getWarpedImagsFromStitcher(double scale);
    Status makeBorderForWarpedImages();
    Status detectorForeground();
    Status compose();

    Status detectorForegroundWithNN(InputArray src, OutputArray dst);

    Rect getLargestInscribedRectangle(InputArray mask);

private:
    void prepare(const vector<Point>& corners, const vector<Size>& sizes);
    void prepare(Rect dst_roi);

    void foreMaskFilter(InputArray src, OutputArray dst, Rect& foreRect);

    int checkOverlappedArea(const vector<Rect>&); //! TODO
    Status composeWithOverlapped();
    Status composeWithoutOverlapped();

    // baidu aip
    std::unique_ptr<aip::Bodyanalysis> bodyAnalyst_;
    std::map<string, string> bodySegOptions_;

    // custom class
    Ptr<ImageStitcher> stitcher_;
    Ptr<cvBlender> blender_;
    Ptr<cv::BackgroundSubtractor> detector_;

    // images
    vector<UMat> inputImages_, inputMasks_;
//    vector<Mat> scaledImages_, scaledMasks_;
    vector<UMat> warpedImages_, warpedMasks_;
    vector<Point> corners_/*, scaledCorners_*/;
    vector<Size> inputSizes_, warpedSize_, scaledSize_/*, scaledWarpedSize_*/;

    vector<Mat> foregrounds_, foregroundMasks_;
    vector<Mat> homograpies_;
    vector<Rect> foregroundRectsRough_, foregroundRectsRefine_;

    UMat baseFrame_, baseFrameMask_, pano_;

    size_t numImgs_, bfi_;  // base frame index
    vector<size_t> indices_;

    UMat dstImg_, dstMask_;
    Rect dstROI_;
};

}  // namespace ms

#endif
