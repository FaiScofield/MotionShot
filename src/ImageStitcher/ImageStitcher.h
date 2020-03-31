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

// enum ForegroundMovingDirection { LEFT, RIGHT, UNKNOWN };

// struct WarpedCorners {
//    Point2f tl;
//    Point2f tr;
//    Point2f bl;
//    Point2f br;
//    ForegroundMovingDirection direction;
//};


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

    static Ptr<ImageStitcher> create(FeatureType ft = SURF, FeatureMatchType mt = KNN);

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
    void setForegrounds(const vector<Rect>& fores) { foregroundRects_ = fores; }
    Status getWarpedImages(OutputArray imgs, OutputArray masks, vector<Point>& corners);  //! TODO
    Status composePanoramaWithForegrounds(OutputArray pano); //! TODO

    // WarpedCorners getWarpedCorners(const Mat& src, const Mat& H);

    // void alphaBlend(const Mat& img1, const Mat& img2, const WarpedCorners& corners, Mat& pano);

    //! debug functions
    Status computeHomography(InputArray img1, InputArray img2, OutputArray H12);

public:
    void getBaseFrame(){}    //! TODO
    Status matchImages();
    Status estimateCameraParams();

    // custom class
    FeatureType featureType_;
    FeatureMatchType featureMatchType_;

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
    vector<UMat> inputImages_, inputMasks_;
    vector<UMat> warpedImages_, warpedMasks_;        // 统一坐标系后的图像
    vector<Point> warpedCorners_;   // 统一坐标系后图像的左上角坐标
    vector<Size> inputImgSize_/*, warpedImgSize_*/;

    size_t numImgs_, baseIndex_;     // 基准帧索引
    UMat baseFrame_, matchingMask_;  // 基准帧, 与基准帧匹配的指导掩模 NxN
    vector<size_t> indices_;

    vector<cv::detail::ImageFeatures> features_;        // 特征
    vector<cv::detail::MatchesInfo> pairwise_matches_;  // 匹配关系
    vector<cv::detail::CameraParams> cameras_;          // 相机参数
    Mat result_mask_;

    vector<Rect> foregroundRects_;
};

}  // namespace ms
#endif  // IMAGE_STITCHER_H
