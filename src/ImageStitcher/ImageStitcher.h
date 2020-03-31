#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include "precompiled.h"
#include "ImageBlender/cvBlenders.h"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/warpers.hpp>

namespace ms
{

//enum ForegroundMovingDirection { LEFT, RIGHT, UNKNOWN };

//struct WarpedCorners {
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
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };

    enum FeatureType { ORB = 0, AKAZE = 1, SURF = 2 };

    enum FeatureMatchType { BF = 0, KNN = 1 };

    ImageStitcher(FeatureType ft = SURF, FeatureMatchType mt = KNN);

    static Ptr<ImageStitcher> create(FeatureType ft = SURF, FeatureMatchType mt = KNN);

    Status estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks = cv::noArray());
    Status composePanorama(OutputArray pano);

    Status warpImages(OutputArray imgs, OutputArray masks, vector<Point> &corners);    //! TODO

    Status stitch(InputArrayOfArrays images, OutputArray pano);
    Status stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano);

    Status computeHomography(InputArray img1, InputArray img2, OutputArray H12);

    //    WarpedCorners getWarpedCorners(const Mat& src, const Mat& H);

    //    void alphaBlend(const Mat& img1, const Mat& img2, const WarpedCorners& corners, Mat& pano);

public:
    void getBaseFrame();
    Status matchImages();
    Status estimateCameraParams();

    // custom class
    FeatureType featureType_;
    FeatureMatchType featureMatchType_;

    Ptr<cv::Feature2D> featureFinder_;
//    Ptr<cv::DescriptorMatcher> featureMatcher_;
    Ptr<cv::detail::FeaturesMatcher> featureMatcher_;
    Ptr<cv::detail::Estimator> motionEstimator_;
    Ptr<cv::detail::BundleAdjusterBase> bundleAdjuster_;
    Ptr<cv::detail::ExposureCompensator> exposureCompensator_;
    Ptr<cv::detail::RotationWarper> warper_;
    Ptr<ms::cvBlender> blender_;

    // flags
    bool doBundleAjustment_;
    bool doWaveCorrection_;
    bool doExposureCompensation_;
    bool doSeamOptimization_;
    bool doSeamlessBlending_;

    // params & threshold
    double registrResol_;  // 最大处理图像分辨率, 单位: M pixel. 超过此分辨率的图片将被缩小
    double compose_resol_;
    double scale_, invScaled_;
    double warped_image_scale_;

    // data
    vector<UMat> fullImages_, fullMasks_;
//    vector<UMat> scaledImages_, scaledMasks_;
    vector<Size> fullImgSize_, scaledImgSize_, warpedImgSize_;

    size_t numImgs_, baseIndex_;   // 基准帧索引
    UMat baseFrame_, matchingMask_;  // 基准帧, 与基准帧匹配的指导掩模 NxN
    std::vector<size_t> indices_;

    vector<Point> corners_;
    vector<UMat> warpedImages_, warpedMasks_;

    std::vector<cv::detail::ImageFeatures> features_;
    std::vector<cv::detail::MatchesInfo> pairwise_matches_;
    std::vector<cv::detail::CameraParams> cameras_;
    Mat result_mask_;
};

}  // namespace ms
#endif  // IMAGE_STITCHER_H
