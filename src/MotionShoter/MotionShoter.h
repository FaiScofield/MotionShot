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
    ~MotionShot();

    UMat getBaseFrame() const { return baseFrame_.clone(); }
    vector<Mat> getForeground() const { return foregrounds_; }
    void setStitcher(Ptr<ImageStitcher> stitcher) { stitcher_ = stitcher; }
    void setBlender(Ptr<cvBlender> blender) { blender_ = blender; }
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
    void prepare(const vector<Point>& corners, const vector<Size>& sizes);
    void prepare(Rect dst_roi);

    /** 对GMM得到的前景掩模进行形态学操作, 得到粗糙的前景掩模 */
    void foreMaskFilter(InputArray src, OutputArray dst, Rect& foreRect);

    /** 检查前景的Rect之间是否有重叠 */
    int countOverlappedRects(const vector<Rect>& rects, vector<Rect>& overlapRects);
    bool isOverlapped(Rect rec1, Rect rec2, Rect& overlap);

    /** 前景有/无重叠时两种合成方式 */
    Status composeWithOverlapped(OutputArray pano);
    Status composeWithoutOverlapped(OutputArray pano);

    // baidu aip
    bool useBaiduAIP_;
    std::unique_ptr<aip::Bodyanalysis> bodyAnalyst_;
    std::map<string, string> bodySegOptions_;

    // custom class
    Ptr<ImageStitcher> stitcher_;
    Ptr<cvBlender> blender_;
    Ptr<cv::BackgroundSubtractor> detector_;

    double detectScale_;

    // images
    vector<UMat> inputImages_, inputMasks_;
//    vector<Mat> scaledImages_, scaledMasks_;
    vector<UMat> warpedImages_, warpedMasks_;
    vector<Point> corners_/*, scaledCorners_*/;
    vector<Size> inputSizes_, warpedSize_, scaledSize_/*, scaledWarpedSize_*/;
    vector<vector<int>> imgBorders_;

    vector<Rect> foregroundRectsRough_, foregroundRectsRefine_;
    vector<Rect> overlappedForegroundRects_;
    vector<Mat> foregrounds_, foregroundMasksRough_, foregroundMasksRefine_;
    vector<Mat> homograpies_;


    UMat baseFrame_, baseFrameMask_, pano_;

    size_t numImgs_, bfi_;  // base frame index
    vector<size_t> indices_;

    MS_DEBUG_TO_DELETE UMat dstImg_, dstMask_;
    MS_DEBUG_TO_DELETE Rect dstROI_;   // 统一的尺寸
};

}  // namespace ms

#endif
