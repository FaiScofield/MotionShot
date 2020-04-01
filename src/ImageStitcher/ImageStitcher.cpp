#include "ImageStitcher.h"
#include "utility.h"
//#include "ImageBlender/cvBlenders.h"
//#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

//#include <opencv2/stitching.hpp>

#define ENABLE_TIMER 1
#define ENABLE_DEBUG_STITCHER 1

namespace ms
{
using namespace std;
using namespace cv;

ImageStitcher::ImageStitcher(FeatureType ft, FeatureMatchType mt)
    : featureType_(ft), featureMatchType_(mt)
{
    switch (featureType_) {
    case ORB:
        featureFinder_ = cv::ORB::create(/*1000, 1.2, 3*/);
        break;
    case AKAZE:
        featureFinder_ = cv::AKAZE::create();
        break;
    case SURF:
    default:
        featureFinder_ = cv::xfeatures2d::SURF::create();
        break;
    }

    // featureMatcher_ = cv::BFMatcher::create();
    // featureMatcher_ = cv::FlannBasedMatcher::create();
    featureMatcher_ = makePtr<ms::detail::BestOf2NearestMatcher>(false);

    motionEstimator_ = makePtr<cv::detail::HomographyBasedEstimator>();
    // warper_ = cv::PlaneWarper::create(1.f);
    warper_ = makePtr<cv::detail::PlaneWarper>(1.f);

    // 分辨率阈值 <= 0 时则说明和输入图像一致
    registResol_ = 0;
    seamResol_ = 0;
    composeResol_ = 0;
    determinScaleByResol_ = false;  // determin resolution

    registScale_ = 0.5;            // determin scale
    seamScale_ = 0.25;
    composeScale_ = 1.;

    warpedImageScale_ = 0;

    baseIndex_ = 0;

    doBundleAjustment_ = true;
    doWaveCorrection_ = false;
    doExposureCompensation_ = true;
    doSeamOptimization_ = true;
    doSeamlessBlending_ = true;

    drawSeamOnOutput_ = false;

    if (doBundleAjustment_) {
        bundleAdjuster_ = makePtr<cv::detail::BundleAdjusterRay>();
        bundleAdjuster_->setConfThresh(1.);
    } else
        bundleAdjuster_ = makePtr<cv::detail::NoBundleAdjuster>();

    if (doExposureCompensation_ || doSeamOptimization_) {  //! TODO
        exposureCompensator_ = makePtr<cv::detail::BlocksGainCompensator>();
        seamFinder_ = makePtr<cv::detail::GraphCutSeamFinder>(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
    }

    if (doSeamlessBlending_)
        blender_ = makePtr<ms::cvFeatherBlender>();
    else
        blender_ = ms::cvBlender::createDefault(ms::cvBlender::NO, false);
}

Ptr<ImageStitcher> ImageStitcher::create(FeatureType ft, FeatureMatchType mt)
{
    return cv::makePtr<ImageStitcher>(ft, mt);
}


void ImageStitcher::setRistResolutions(double r1, double r2, double r3)
{
    registResol_ = r1;
    seamResol_ = r2;
    composeResol_ = r3;
    determinScaleByResol_ = true;

    assert(r1 > 0 && r2 > 0);
    if (r2 > r1)
        WARNING("It's better to set the seam_resolution <= registration_resolution");
}

void ImageStitcher::setScales(double s1, double s2, double s3)
{
    registScale_ = s1;
    seamScale_ = s2;
    composeScale_ = s3;
    determinScaleByResol_ = false;

    assert(s1 > 0 && s2 > 0 && s3 > 0);
}


ImageStitcher::Status ImageStitcher::stitch(InputArrayOfArrays images, OutputArray pano)
{
    return stitch(images, cv::noArray(), pano);
}

ImageStitcher::Status ImageStitcher::stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano)
{
    Status status = estimateTransform(images, masks);

    if (status != OK)
        return status;
    return composePanorama(pano);
}

ImageStitcher::Status ImageStitcher::estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks)
{
    images.getUMatVector(inputImages_);
    masks.getUMatVector(inputMasks_);
    numImgs_ = inputImages_.size();
    if (!inputMasks_.empty())
        assert(inputMasks_.size() == numImgs_);

    if (baseIndex_ == 0) {
        baseIndex_ = (numImgs_ - 1) / 2;
        matchingMask_ = UMat::zeros(numImgs_, numImgs_, CV_8UC1);
        Mat mask = matchingMask_.getMat(ACCESS_RW);
        mask.row(baseIndex_).setTo(255);
        mask.col(baseIndex_).setTo(255);
        mask.at<uchar>(baseIndex_, baseIndex_) = 0;
        INFO("base frame index is: " << baseIndex_ << ", and the matching mask is: " << endl << mask);
    }

    Status status;
    if ((status = matchImages()) != OK) {
        WARNING("Return status not OK in ImageStitcher::matchImages()");
        return status;
    }

    if ((status = estimateCameraParams()) != OK) {
        WARNING("Return status not OK in ImageStitcher::estimateCameraParams()");
        return status;
    }

    return OK;
}

ImageStitcher::Status ImageStitcher::composePanorama(OutputArray pano)
{
#define ENABLE_DEBUG_COMPOSE    1

    assert((int)numImgs_ >= 2);
    assert(indices_.size() >= 2);

    vector<UMat> warpedImages(numImgs_);
    vector<UMat> warpedMasks(numImgs_);
    vector<Point> warpedCorners(numImgs_);
    vector<Size> warpedSizes(numImgs_);

    /// 缝合线优化
    if (doExposureCompensation_ || doSeamOptimization_) {
        INFO("Warping images for seamless process (auxiliary)... ");

        const float seam_work_aspect = static_cast<float>(seamScale_ / registScale_);
        INFO("seamScale_ = " << seamScale_ << ", and seam_work_aspect = " << seam_work_aspect);

        warper_->setScale(static_cast<float>(warpedImageScale_ * seam_work_aspect));

        // 在更低的尺度上估计缝合线
        for (size_t i = 0; i < numImgs_; ++i) {
            Mat_<float> K;
            cameras_[i].K().convertTo(K, CV_32F);
            K(0, 0) *= seam_work_aspect;
            K(0, 2) *= seam_work_aspect;
            K(1, 1) *= seam_work_aspect;
            K(1, 2) *= seam_work_aspect;

            UMat seamEstImg, seamEstMask;
            resize(inputImages_[i], seamEstImg, Size(), seamScale_, seamScale_, INTER_LINEAR_EXACT);
            warpedCorners[i] = warper_->warp(seamEstImg, K, cameras_[i].R, INTER_LINEAR,
                                             BORDER_REFLECT, warpedImages[i]);
            warpedSizes[i] = warpedImages[i].size();

            //? FIXME 为何这里不需要从input mask里面warp?
            seamEstMask.create(seamEstImg.size(), CV_8UC1);
            seamEstMask.setTo(255);
            warper_->warp(seamEstMask, K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
        }

        // 先曝光补偿
        exposureCompensator_->feed(warpedCorners, warpedImages, warpedMasks);
        for (size_t i = 0; i < numImgs_; ++i)
            exposureCompensator_->apply(int(i), warpedCorners[i], warpedImages[i], warpedMasks[i]);

#if DEBUG && ENABLE_DEBUG_COMPOSE
        vector<UMat> warped_masks_tmp(numImgs_);
        for (size_t i = 0; i < numImgs_; ++i)
            warped_masks_tmp[i] = warpedMasks[i].clone();
#endif

        // 再寻找缝合线
        vector<UMat> warpedImages_f(numImgs_);
        for (size_t i = 0; i < numImgs_; ++i)
            warpedImages[i].convertTo(warpedImages_f[i], CV_32F);
        seamFinder_->find(warpedImages_f, warpedCorners, warpedMasks);

#if DEBUG && ENABLE_DEBUG_COMPOSE
        //! FIXME 0.5/0.25的scale seamFinder会使mask清空!
        for (size_t i = 0; i < numImgs_; ++i) {
            imshow("mask before seam", warped_masks_tmp[i]);
            imshow("mask after seam", warpedMasks[i]);
            waitKey(0);
        }
#endif

        if (drawSeamOnOutput_) {
            for (size_t i = 0; i < numImgs_; ++i) {
                vector<vector<Point>> contours;
                findContours(warpedMasks[i], contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
                contours_.push_back(contours);
            }
        }

    }
    destroyAllWindows();
    warpedImages.clear();

    /// 图像合成
    INFO(endl << "Compositing...");

    // 由于可能设置了合成图像的分辨率(compose_resol_), 所以要计算在合成图像尺度上的相机参数(cameras_scaled),
    // 和每帧图像变换后的尺寸(sizes_warped)和左上角坐标(corners)
    const float compose_work_aspect = static_cast<float>(composeScale_ / registScale_);
    warper_->setScale(static_cast<float>(warpedImageScale_ * compose_work_aspect));
    INFO("composeScale_ = " << composeScale_ << ", and compose_work_aspect = " << compose_work_aspect);

    // Update corners and sizes. 更新变换后的图像大小和左上角坐标, 然后初始化blender
    std::vector<ms::detail::CameraParams> cameras_ComposeScale(cameras_);
    for (size_t i = 0; i < numImgs_; ++i) {
        // Update intrinsics
        cameras_ComposeScale[i].ppx *= compose_work_aspect;
        cameras_ComposeScale[i].ppy *= compose_work_aspect;
        cameras_ComposeScale[i].focal *= compose_work_aspect;

        // Update corner and size
        Size sz = inputImgSize_[i];
        if (std::abs(composeScale_ - 1) > 1e-1) {
            sz.width = cvRound(inputImgSize_[i].width * composeScale_);
            sz.height = cvRound(inputImgSize_[i].height * composeScale_);
        }

        Mat K;
        cameras_ComposeScale[i].K().convertTo(K, CV_32F);

        // warp后的图像在pano上的roi
        const Rect roi = warper_->warpRoi(sz, K, cameras_ComposeScale[i].R);
        warpedCorners[i] = roi.tl();
        warpedSizes[i] = roi.size();
    }
    blender_->prepare(warpedCorners, warpedSizes);


    // 开始合成
    UMat img, img_warped, img_warped_s, mask, mask_warped;  // tmp var
    for (size_t j = 0, iend = indices_.size(); j < iend; ++j) {
        const size_t imgIdx = indices_[j];
        INFO("Compositing image #" << imgIdx);

        // Read image and resize it if necessary
        UMat& full_img = inputImages_[imgIdx];
        if (std::abs(composeScale_ - 1) > 1e-1)
            resize(full_img, img, Size(), composeScale_, composeScale_, INTER_LINEAR_EXACT);
        else
            img = full_img;

        if (drawSeamOnOutput_) {
            vector<vector<Point>> contours = contours_[imgIdx];
            for (size_t m = 0; m < contours.size(); ++m) {
                for (size_t n = 0; n < contours[m].size(); ++n) {
                    contours[m][n].x /= seamScale_;
                    contours[m][n].y /= seamScale_;
                }
            }
            drawContours(img, contours, -1, Scalar(0,0,255), 3);
        }

        Mat K;
        cameras_ComposeScale[imgIdx].K().convertTo(K, CV_32F);  // 合成图像尺度下的相机参数前面已经更新过了

        // Warp the current image
        //! 注意: 差值方式 #InterpolationFlags 需要和 remap() 函数的一致. INTER_LINEAR_EXACT 无法使用
        warper_->warp(img, K, cameras_[imgIdx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

#if DEBUG && ENABLE_DEBUG_COMPOSE
        NamedLargeWindow("compose scale image");
        NamedLargeWindow("compose scale warped image");
        imshow("compose scale image", img);
        imshow("compose scale warped image", img_warped);
        waitKey(0);
#endif

        // Warp the current image mask
        mask.create(img.size(), CV_8U);
        mask.setTo(Scalar::all(255));
        warper_->warp(mask, K, cameras_[imgIdx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // 如果做过缝合线优化, 则保留缝合线附近的mask(warpedMasks[imgIdx])
        if (doExposureCompensation_ || doSeamOptimization_) {
            exposureCompensator_->apply((int)imgIdx, warpedCorners[imgIdx], img_warped, mask_warped);

            // Make sure seam mask has proper size
            UMat dilated_mask, seam_mask;
            dilate(warpedMasks[imgIdx], dilated_mask, Mat());
            resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
            bitwise_and(seam_mask, mask_warped, mask_warped);  // 只留取缝合线附件的区域
        }

        // Blend the current image
        img_warped.convertTo(img_warped_s, CV_16S);
        blender_->feed(img_warped_s, mask_warped, warpedCorners[imgIdx]);
    }
    img.release();
    img_warped.release();
    img_warped_s.release();
    mask.release();
    mask_warped.release();
    destroyAllWindows();

    // 得到融合结果
    UMat result, resultMask;
    blender_->blend(result, resultMask);

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    result.convertTo(pano, CV_8U);

    if (result.empty())
        return ERR_STITCH_FAIL;
    return OK;
}

ImageStitcher::Status ImageStitcher::matchImages()
{
    if ((int)numImgs_ < 2) {
        ERROR("Need more images!");
        return ERR_NEED_MORE_IMGS;
    }

    features_.resize(numImgs_);
    inputImgSize_.resize(numImgs_);

#if ENABLE_TIMER
    INFO("Finding features...");
    int64 t = getTickCount();
#endif

    MS_DEBUG_TO_DELETE vector<Mat> Homs(numImgs_);
    vector<UMat> featureExtImgs, featureExtMasks;
    featureExtImgs.resize(numImgs_);
    featureExtMasks.resize(numImgs_);

    // 先确定一下缩放比例, 用最大图像的分辨率确定
    if (determinScaleByResol_) {
        double maxArea = 0;
        for (size_t i = 0; i < numImgs_; ++i)
            maxArea = inputImages_[i].size().area() > maxArea ? inputImages_[i].size().area() : maxArea;

        if (registResol_ > 0)
            registScale_ = std::min(1.0, std::sqrt(registResol_ * 1e6 / maxArea));
        else
            registScale_ = 1.;

        if (seamScale_ > 0)
            seamScale_ = std::min(1.0, std::sqrt(seamResol_ * 1e6 / maxArea));
        else
            seamScale_ = 1.;

        if (composeResol_ > 0)
            composeScale_ = std::min(1.0, std::sqrt(composeResol_ * 1e6 / maxArea));
        else
            composeScale_ = 1.;
    }

    // 1.缩小图像, 提取特征点
    for (size_t i = 0; i < numImgs_; ++i) {
        inputImgSize_[i] = inputImages_[i].size();

        resize(inputImages_[i], featureExtImgs[i], Size(), registScale_, registScale_, INTER_LINEAR_EXACT);
        if (!inputMasks_.empty()) {
            resize(inputMasks_[i], featureExtMasks[i], Size(), registScale_, registScale_, INTER_NEAREST);
            ms::detail::computeImageFeatures(featureFinder_, featureExtImgs[i], features_[i],
                                             featureExtMasks[i]);
        } else {
            ms::detail::computeImageFeatures(featureFinder_, featureExtImgs[i], features_[i]);
        }

        features_[i].img_idx = (int)i;
        INFO("Features in image #" << i << ": " << features_[i].keypoints.size());

//        if (i != baseIndex_)
//            computeHomography(inputImages_[baseIndex_], inputImages_[i], Homs[i]);  // Hib
    }


#if ENABLE_TIMER
    TIMER("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    t = getTickCount();
    INFO("Pairwise matching...");
#endif

    assert(!matchingMask_.empty());
    (*featureMatcher_)(features_, pairwise_matches_, matchingMask_);  // Hib
    featureMatcher_->collectGarbage();

#if ENABLE_TIMER
    TIMER("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
//    for (size_t i = 0; i < numImgs_; ++i) {
//        if (i != baseIndex_) {
//            Mat H_inv = Homs[i].inv();
//            INFO("Homographies diference: " << endl << H_inv * pairwise_matches_[i].H << endl);
//        }
//    }
#endif

    // indices_ = cv::detail::leaveBiggestComponent(features_, pairwise_matches_, 1.f);
    indices_.resize(numImgs_);  //! TODO
    for (size_t i = 0; i < numImgs_; ++i)
        indices_[i] = i;

    return OK;
}

ImageStitcher::Status ImageStitcher::estimateCameraParams()
{
    assert((int)numImgs_ >= 2);
    assert(!features_.empty() && !pairwise_matches_.empty());

    // estimate homography in global frame
    INFO("Estimating camera params...");
    if (!(*motionEstimator_)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;

    for (size_t i = 0; i < cameras_.size(); ++i) {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
#if DEBUG && ENABLE_DEBUG_STITCHER
//        INFO("Initial intrinsic parameters #" << indices_[i] << ":\n " << cameras_[i].K());
#endif
    }

    if (doBundleAjustment_) {
        bundleAdjuster_->setConfThresh(1.);
        if (!(*bundleAdjuster_)(features_, pairwise_matches_, cameras_))
            return ERR_CAMERA_PARAMS_ADJUST_FAIL;
    }

    // Find median focal length and use it as final image scale
    vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i) {
        //INFO("Camera intrinsic #" << indices_[i] << ":\n" << cameras_[i].K());
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warpedImageScale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warpedImageScale_ =
            static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    INFO("warped_image_scale_ = focal = " << warpedImageScale_);

    if (doWaveCorrection_) {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        ms::detail::waveCorrect(rmats, ms::detail::WAVE_CORRECT_HORIZ);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
    }

    return OK;
}


///////////////////////////////
/// 用法2
///////////////////////////////

ImageStitcher::Status ImageStitcher::getWarpedImages(OutputArrayOfArrays _imgs, OutputArrayOfArrays _masks,

                                                     vector<Point>& _corners, double scale)
{
    // 确保已经计算过变换关系
    if (warpedImageScale_ == 0 || cameras_.empty())
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;
    assert((int)numImgs_ >= 2);
    assert(scale > 0 && scale <= 1);

    vector<UMat> warpedImages(numImgs_);
    vector<UMat> warpedMasks(numImgs_);
    vector<Point> warpedCorners(numImgs_);

    INFO(endl << "Warping on detection scale...");

    const float detect_work_aspect = static_cast<float>(scale / registScale_);
    warper_->setScale(static_cast<float>(warpedImageScale_ * detect_work_aspect));
    INFO("input detect scale is " << scale << ", and detect_work_aspect will be " << detect_work_aspect);

    UMat img, warpedMask;
    std::vector<ms::detail::CameraParams> cameras_detectScale(cameras_);
    for (size_t i = 0; i < numImgs_; ++i) {
        cameras_detectScale[i].ppx *= detect_work_aspect;
        cameras_detectScale[i].ppy *= detect_work_aspect;
        cameras_detectScale[i].focal *= detect_work_aspect;

        UMat& full_img = inputImages_[i];
        if (std::abs(scale - 1) > 1e-2)
            resize(full_img, img, Size(), scale, scale, INTER_LINEAR_EXACT);
        else
            img = full_img;

        Mat K;
        cameras_detectScale[i].K().convertTo(K, CV_32F);

        // Warp the current image
        warpedCorners[i] = warper_->warp(img, K, cameras_[i].R, INTER_LINEAR, BORDER_REFLECT, warpedImages[i]);

        // Warp the current image mask
        UMat mask = UMat(img.size(), CV_8U);
        mask.setTo(255);
        warper_->warp(mask, K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, warpedMasks[i]);
    }
    img.release();
    warpedMask.release();

    //! NOTE assign 之前要确保 _imgs 的 size 一致
    _imgs.assign(warpedImages);
    _masks.assign(warpedMasks);
    _corners = warpedCorners;

    return OK;
}

ImageStitcher::Status ImageStitcher::computeHomography(InputArray img1, InputArray img2, OutputArray H)
{
    if (img1.empty() || img2.empty())
        return ERR_HOMOGRAPHY_EST_FAIL;

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;

    featureFinder_->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    featureFinder_->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    vector<DMatch> matches;
    Ptr<cv::DescriptorMatcher> matcher;
    switch (featureMatchType_) {
    case BF:
        matcher = cv::BFMatcher::create();
        matcher->match(descriptors1, descriptors2, matches);
        break;
    case KNN:
    default:
        matcher = cv::FlannBasedMatcher::create();
        vector<vector<DMatch>> vvMatches;
        matcher->knnMatch(descriptors1, descriptors2, vvMatches, 2);
        matches.reserve(vvMatches.size());
        for (size_t i = 0, iend = vvMatches.size(); i < iend; ++i) {
            if (vvMatches[i].size() < 2)
                continue;
            if (vvMatches[i][0].distance < 0.4 * vvMatches[i][1].distance)
                matches.push_back(vvMatches[i][0]);
        }
        break;
    }

    // 将keypoints类型转换为Point2f
    vector<Point2f> points1, points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        points1.push_back(keypoints1[it->queryIdx].pt);
        points2.push_back(keypoints2[it->trainIdx].pt);
    }
    if (points1.empty() || points2.empty()) {
        WARNING("图像差距太大! 无匹配点对!");
        return ERR_HOMOGRAPHY_EST_FAIL;
    }
    assert(points1.size() == points2.size());

    // 使用RANSAC算法估算单应矩阵
    vector<uchar> inliers;
    Mat H12 = findHomography(points1, points2, inliers, RANSAC);
    H.assign(H12);


    {
#define DEBUG_RESULT_HOMOGRAPHY 0
#if DEBUG && DEBUG_RESULT_HOMOGRAPHY
        Mat imageMatches;
        drawMatches(img1, keypoints1, img2, keypoints2, matches, imageMatches, Scalar(0, 255, 255),
                    Scalar(0, 255, 0), inliers);
        imshow("Matches inliers", imageMatches);

        // 用单应矩阵对图像进行变换
        Mat warped;
        Mat H12 = homography.inv(DECOMP_SVD);
        WarpedCorners corners = getWarpedCorners(img2, H12);
        int panoCols = max(cvCeil(max(corners.tr.x, corners.br.x)), img1.cols);
        warpPerspective(img2, warped, H12, Size(panoCols, img2.rows));
        img1.copyTo(warped.colRange(0, img1.cols));
        imshow("Pano without blending", warped);
#endif
    }

    return OK;
}

/*
WarpedCorners ImageStitcher::getWarpedCorners(const Mat& src, const Mat& H)
{
    WarpedCorners corners;

    // 左上角(0,0,1)
    double v2[] = {0, 0, 1};
    double v1[3];  // 变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2);
    Mat V1 = Mat(3, 1, CV_64FC1, v1);

    V1 = H * V2;
    corners.tl.x = v1[0] / v1[2];
    corners.tl.y = v1[1] / v1[2];

    // 左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows - 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.bl.x = v1[0] / v1[2];
    corners.bl.y = v1[1] / v1[2];

    // 右上角(src.cols,0,1)
    v2[0] = src.cols - 1;
    v2[1] = 0;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.tr.x = v1[0] / v1[2];
    corners.tr.y = v1[1] / v1[2];

    // 右下角(src.cols,src.rows,1)
    v2[0] = src.cols - 1;
    v2[1] = src.rows - 1;
    V2 = Mat(3, 1, CV_64FC1, v2);
    V1 = Mat(3, 1, CV_64FC1, v1);
    V1 = H * V2;
    corners.br.x = v1[0] / v1[2];
    corners.br.y = v1[1] / v1[2];

    if (corners.tl.x < 0 && corners.bl.x < 0)
        corners.direction = LEFT;
    else if (corners.tr.x > src.cols && corners.br.x > src.cols)
        corners.direction = RIGHT;
    else
        corners.direction = UNKNOWN;

    return corners;
}

void ImageStitcher::alphaBlend(const Mat& img1, const Mat& img2, const WarpedCorners& corners, Mat& pano)
{
#define DEBUG_RESULT_BLEND

    //! TODO 相机向左/右移动, 其重叠区域的变换?

    // 相机向右运动
    int expBorder = 0;  // TODO
    Point2i tl1_overlapped, br1_overlapped;
    tl1_overlapped.x = max(0, cvFloor(min(corners.tl.x, corners.bl.x) - expBorder));
    tl1_overlapped.y = max(0, cvFloor(min(corners.tl.y, corners.tr.y) - expBorder));
    br1_overlapped.x = min(fullImgSize_.width, cvCeil(max(corners.tr.x, corners.br.x)) + expBorder);
    br1_overlapped.y = min(fullImgSize_.height, cvCeil(max(corners.bl.y, corners.br.y)) + expBorder);
    Rect overlappedArea1(tl1_overlapped, br1_overlapped);

//    Point2i tl2_overlapped, br2_overlapped;
//    if ((corners.tl.x >= img1.cols && corners.bl.x >= img1.cols) || (corners.tr.x <= 0 && corners.br.x <= 0) ||
//        (corners.tl.y > img1.rows && corners.tr.y > img1.rows) || (corners.bl.y <= 0 && corners.br.y <= 0)) {
//        ; // TODO
//    }
//    tl2_overlapped.x =
//        min(corners.tl.x, corners.bl.x) < 0 ? 0 : cvFloor(min(corners.tl.x, corners.bl.x) - expBorder);
//    tl2_overlapped.y =
//        min(corners.tl.y, corners.tr.y) < 0 ? 0 : cvFloor(min(corners.tl.y, corners.tr.y) - expBorder);
//    br2_overlapped.x = cvCeil(tl2_overlapped.x + overlappedArea1.width);
//    br2_overlapped.y = cvCeil(tl2_overlapped.y + overlappedArea1.height);
//    Rect overlappedArea2(tl2_overlapped, br2_overlapped);
//    assert(overlappedArea1.area() == overlappedArea2.area());

    //! TODO   distanceTransform
    Mat img1Gray, img2Gray;
    Mat mask1, mask2, mask2Dist, panoMask;
    cvtColor(img1, img1Gray, COLOR_BGR2GRAY);
    cvtColor(img2, img2Gray, COLOR_BGR2GRAY);
    compare(img1Gray, 0, mask1, CMP_GT);
    compare(img2Gray, 0, mask2, CMP_GT);
    mask2.setTo(255, mask1);
    imshow("mask1", mask1);
    imshow("mask2", mask2);
    distanceTransform(mask2, mask2Dist, DIST_L1, DIST_MASK_3);
    mask2Dist.convertTo(mask2Dist, CV_8UC1);
    cout << mask2Dist.type() << endl;
    imshow("mask2Dist", mask2Dist);

    cvFeatherBlender cvFB;
    Mat img1S, img2S;
    img1.convertTo(img1S, CV_16SC3);
    img2.convertTo(img2S, CV_16SC3);
    cvFB.prepare(Rect(0,0,img1.cols, img1.rows));
    cvFB.feed(img1S, mask1, Point(0, 0));
    cvFB.feed(img2S, mask2, Point(0, 0));
    cvFB.blend(pano, panoMask);
    pano.convertTo(pano, CV_8U);
    imshow("pano", pano);


//    compare(img2, 0, mask2, CMP_EQ);
//    bitwise_and(mask1, mask2, mask);
//    Mat dw1, dw2, dw;
//    Mat ig1, ig2;
//    cvtColor(img1(overlappedArea1), ig1, COLOR_BGR2GRAY);
//    cvtColor(img2(overlappedArea1), ig2, COLOR_BGR2GRAY);
//    distanceTransform(ig1, dw1, DIST_L2, DIST_MASK_PRECISE, CV_64FC1);
//    distanceTransform(ig2, dw2, DIST_L2, DIST_MASK_PRECISE, CV_64FC1);
//    vconcat(dw1, dw2, dw);
//    imshow("distanceTransform", dw);


//    Mat weight1 = Mat(img2.size(), CV_64FC3, Scalar(1., 1., 1.));
//    Mat weight2 = Mat(img2.size(), CV_64FC3, Scalar(1., 1., 1.));
//    for (int c = tl1_overlapped.x, cend = tl1_overlapped.x + overlappedArea1.width; c < cend; ++c) {
//        const double beta = 1. * (c - tl1_overlapped.x) / overlappedArea1.width;
//        const double alpha = 1. - beta;
//        weight1.col(c).setTo(Vec3d(alpha, alpha, alpha));
//        weight2.col(c).setTo(Vec3d(beta, beta, beta));
//        // TODO 斜分割线处需要将beta设为0. distanceTransform()
//    }

#ifdef DEBUG_RESULT_BLEND
    // show
//    Mat tmp1, tmp2, tmp3;
//    tmp1 = img1.clone();
//    tmp2 = img2.clone();
//    rectangle(tmp1, overlappedArea1, Scalar(0, 0, 255));
//    rectangle(tmp2, overlappedArea1, Scalar(0, 0, 255));
//    vconcat(tmp1, tmp2, tmp3);
//    imshow("Images & Overlapped area", tmp3);

//    Mat tmp4;
//    vconcat(weight1, weight2, tmp4);
//    imshow("Overlapped area weight", tmp4);

    waitKey(0);
#endif

    // alpha blend
//    Mat w1, w2;
//    img1.convertTo(w1, CV_64FC3);
//    img2.convertTo(w2, CV_64FC3);
//    w1 = w1.mul(weight1);
//    w2 = w2.mul(weight2);
//    pano = Mat::zeros(img2.size(), CV_64FC3);
//    pano += w1 + w2;
//    pano.convertTo(pano, CV_8UC3);
}

*/

}  // namespace ms
