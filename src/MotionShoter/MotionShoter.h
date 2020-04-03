#ifndef MOTION_SHOT_H
#define MOTION_SHOT_H

#include "ImageBlender/cvBlenders.h"
#include "ImageStitcher/ImageStitcher.h"
#include "Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h"
#include "precompiled.h"

#include <map>
#include <memory>

#include <opencv2/bgsegm.hpp>
#include <opencv2/video.hpp>

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
        ERR_STITCH_FAIL = 5,
        ERR_BAD_ARGUMENTS = -1
    };

    MotionShot();
    ~MotionShot();

    void setStitcher(Ptr<ImageStitcher> stitcher) { stitcher_ = stitcher; }
    Ptr<ImageStitcher> stitcher() { return stitcher_; }
    void setForegroundDetector(Ptr<cv::BackgroundSubtractor> detector) { detector_ = detector; }
    Ptr<cv::BackgroundSubtractor> foregroundDetector() { return detector_; }

    // stitcher params
    void setFeatureType(int ft) { stitcher_->setFeatureType(ft); }
    void setBlenderType(int bt) { stitcher_->setBlenderType(bt); }
    void setRistResolutions(double r1, double r2, double r3 = 0)
    {
        stitcher_->setRistResolutions(r1, r2, r3);
    }
    void setScales(double s1 = 0.5, double s2 = 0.25, double s3 = 1.)
    {
        stitcher_->setScales(s1, s2, s3);
    }

    void setFlag_BundleAjustment(bool flag) { stitcher_->setBundleAjustment(flag); }
    void setFlag_WaveCorrection(bool flag) { stitcher_->setWaveCorrection(flag); }
    void setFlag_SeamOptimization(bool flag) { stitcher_->setSeamOptimization(flag); }
    void setFlag_ExposureCompensation(bool flag) { stitcher_->setExposureCompensation(flag); }

    void setFlag_UseBaiduAIP(bool flag) { useBaiduAIP_ = flag; }
    void setDetectScale(double scale) { detectScale_ = scale; }

    Status setInputs(InputArrayOfArrays imgs, InputArrayOfArrays masks = cv::noArray());
    Status run();
    void getResult(OutputArray pano) { pano.assign(pano_); }

    /** 特征点匹配计算图像投影变换, 得到统一坐标系后统一大小的图像 */
    Status getWarpedImagsFromStitcher(double scale);

    /**
     * @brief 检测前景, 得到前景的粗糙区域(foregroundMasksRough_)和掩模(foregroundMasksRough_).
     * 如果有用NN检测, 可以得到更精细的结果(foregroundRectsRefine_ / foregroundMasksRefine_)
     * @details 中值滤波得到无前景的背景 -> 背景消除 -> 形态学操作 -> 前景(建模)筛选(TODO) -> 得到粗糙前景
     */
    Status detectorForeground();

    /** 图像合成 */
    Status compose(OutputArray pano);

    /** 用百度AIP检测人物前景 */
    Status detectorForegroundWithNN(InputArray src, OutputArray dst);

    /**
     * @brief 获得所有掩模的共同重叠区域中的最大内接矩形
     * TODO 算法实现待改进, 当前得到的矩形严格上不是最大内接矩形
     */
    Rect getLargestInscribedRectangle(InputArray mask);

    /**
     * @brief TODO 根据前景运动的连续性确定前景的位置
     * @ref 可参考meansift算法等
     */
    Status selectFromPotentialRect(const vector<vector<Rect>>& recs) { return OK; }

private:
    /** 统一warp后的图像和掩模的大小 */
    void makeBorderForWarpedImages();

    /** 对GMM得到的前景掩模进行形态学操作, 得到粗糙的前景掩模 */
    void foreMaskFilter(InputArray src, OutputArray dst, Rect& foreRect);

    /** 检查前景的Rect之间是否有重叠 */
    int countOverlappedRects(const vector<Rect>& rects, vector<Rect>& overlapRects);
    bool isOverlapped(Rect rec1, Rect rec2, Rect& overlap);

    // baidu aip
    bool useBaiduAIP_;
    std::unique_ptr<aip::Bodyanalysis> bodyAnalyst_;
    std::map<string, string> bodySegOptions_;

    // custom class
    Ptr<ImageStitcher> stitcher_;
    Ptr<cv::BackgroundSubtractor> detector_;

    double detectScale_;
    size_t numImgs_, bfi_;  // base frame index

    // images
    vector<UMat> inputImages_, inputMasks_;
    vector<Size> inputSizes_;
    vector<size_t> indices_;

    vector<UMat> warpedImages_, warpedMasks_;  // get from stitcher
    vector<Point> corners_;                    // get from stitcher

    // fore detection
    vector<Rect> foregroundRectsRough_, foregroundRectsRefine_;
    vector<Rect> overlappedForegroundRects_;
    vector<Mat> foregroundMasksRough_, foregroundMasksRefine_;
    vector<Mat> homograpies_;

    UMat pano_;
};

}  // namespace ms

#endif
