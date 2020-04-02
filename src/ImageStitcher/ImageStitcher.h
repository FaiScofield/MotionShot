#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include "ImageBlender/cvBlenders.h"
#include "precompiled.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/warpers.hpp>

namespace ms
{


class ImageStitcher
{
public:
    enum Status {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3,
        ERR_STITCH_FAIL = 4
    };

    enum FeatureType { ORB = 0, AKAZE = 1, SURF = 2 };

    enum FeatureMatchType { BF = 0, KNN = 1 };

    ImageStitcher(FeatureType ft = SURF, FeatureMatchType mt = KNN);
    ~ImageStitcher();

    static Ptr<ImageStitcher> create(FeatureType ft = SURF, FeatureMatchType mt = KNN);

    void setFeatureType(int ft) { featureType_ = (FeatureType)ft; }
    void setFeatureMatchMethod(int fmm) { featureMatchType_ = (FeatureMatchType)fmm; }
    void setBundleAjustment(bool flag) { doBundleAjustment_ = flag; }
    void setWaveCorrection(bool flag) { doWaveCorrection_ = flag; }
    void setSeamOptimization(bool flag) { doSeamOptimization_ = flag; }

    void setRistResolutions(double r1, double r2, double r3 = 0);
    void setScales(double s1 = 0.5, double s2 = 0.25, double s3 = 1.);

    //! 原始用法, 拼接图像(变换+合成)
    Status stitch(InputArrayOfArrays images, OutputArray pano);
    Status stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano);
    // 变换
    Status estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks = cv::noArray());
    // 合成
    Status composePanorama(OutputArray pano);

    //! 用法2, (变换+前景检测(outside)+合成), 先输出变换结果, 由外部检测前景, 再传入相关数据进行合成
    void setForegrounds(InputArrayOfArrays foreMasks, const vector<Rect>& fores);
    void setOverlappedForegroundRects(const vector<Rect>& recs) { overlappedForegroundRects_ = recs; }
    Status getWarpedImages(OutputArrayOfArrays imgs, OutputArrayOfArrays masks,
                           vector<Point>& corners, double scale = 0.5);
    Status composePanoWithoutOverlapped(OutputArray pano);
    Status composePanoWithOverlapped(OutputArray pano);  //! TODO

    //! debug functions
    MS_DEBUG_TO_DELETE Status computeHomography(InputArray img1, InputArray img2, OutputArray H12);
    MS_DEBUG_TO_DELETE bool drawSeamOnOutput_;
    MS_DEBUG_TO_DELETE vector<vector<vector<Point>>> contours_;

private:
    Status matchImages();
    Status estimateCameraParams();

    FeatureType featureType_;
    FeatureMatchType featureMatchType_;

    // custom class
    Ptr<cv::Feature2D> featureFinder_;
    // Ptr<cv::DescriptorMatcher> featureMatcher_;
    Ptr<cv::detail::FeaturesMatcher> featureMatcher_;
    Ptr<cv::detail::Estimator> motionEstimator_;
    Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster_;
    Ptr<cv::detail::ExposureCompensator> exposureCompensator_;
    Ptr<cv::detail::SeamFinder> seamFinder_;
    Ptr<cv::detail::RotationWarper> warper_;
    Ptr<ms::cvBlender> blender_;

    // flags
    bool doBundleAjustment_;
    bool doWaveCorrection_;
    bool doExposureCompensation_;
    bool doSeamOptimization_;  //! TODO 曝光补偿和缝合线优化暂时是同时进行的
    bool doSeamlessBlending_;

    // params & threshold
    bool determinScaleByResol_;
    double registResol_, seamResol_, composeResol_;  // unit [M pixel]
    double registScale_, seamScale_, composeScale_;
    double warpedImageScale_;  // = focus

    // data
    bool emptyInputMask_;
    vector<UMat> inputImages_, inputMasks_;
    vector<Size> inputImgSize_ /*, warpedImgSize_*/;

    size_t numImgs_;          // 输入图像总数
    size_t baseIndex_;        // 基准帧索引
    UMat matchingMask_;       // 与基准帧匹配的指导掩模 NxN
    vector<size_t> indices_;  // 有效图像索引

    vector<cv::detail::ImageFeatures> features_;        // 特征
    vector<cv::detail::MatchesInfo> pairwise_matches_;  // 匹配关系
    vector<cv::detail::CameraParams> cameras_;          // (在registration尺度下的)相机参数

    vector<UMat> foregroundMasks_;
    vector<Rect> foregroundRects_, overlappedForegroundRects_;
};

}  // namespace ms
#endif  // IMAGE_STITCHER_H
