#include "ImageStitcher.h"
#include "utility.h"
//#include "ImageBlender/cvBlenders.h"
//#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/xfeatures2d/nonfree.hpp>

//#include <opencv2/stitching.hpp>

#define ENABLE_DEBUG_STITCHER 1
#define ENABLE_LOG 1

namespace ms
{
using namespace std;
using namespace cv;

ImageStitcher::ImageStitcher(FeatureType ft, FeatureMatchType mt)
    : featureType_(ft), featureMatchType_(mt)
{
    switch (featureType_) {
    case ORB:
        featureFinder_ = cv::ORB::create(1000, 1.2, 5);
        break;
    case AKAZE:
        featureFinder_ = cv::AKAZE::create();
        break;
    case SURF:
    default:
        featureFinder_ = cv::xfeatures2d::SURF::create();
        break;
    }

    //    featureMatcher_ = cv::BFMatcher::create();
    //    featureMatcher_ = cv::FlannBasedMatcher::create();
    featureMatcher_ = makePtr<ms::detail::BestOf2NearestMatcher>(false);

    motionEstimator_ = makePtr<cv::detail::HomographyBasedEstimator>();
//    warper_ = cv::PlaneWarper::create(1.f);
    warper_ = makePtr<cv::detail::PlaneWarper>(1.f);

    doBundleAjustment_ = false;
    doWaveCorrection_ = false;
    doExposureCompensation_ = false;
    doSeamOptimization_ = false;
    doSeamlessBlending_ = false;

    if (doBundleAjustment_)
        bundleAdjuster_ = makePtr<cv::detail::BundleAdjusterRay>();
    else
        bundleAdjuster_ = makePtr<cv::detail::NoBundleAdjuster>();

    if (doExposureCompensation_)
        exposureCompensator_ = makePtr<cv::detail::BlocksGainCompensator>();

    if (doSeamlessBlending_)
        blender_ = makePtr<ms::cvFeatherBlender>();
    else
        blender_ = ms::cvBlender::createDefault(ms::cvBlender::NO, false);
}

Ptr<ImageStitcher> ImageStitcher::create(FeatureType ft, FeatureMatchType mt)
{
    return cv::makePtr<ImageStitcher>(ft, mt);
}


ImageStitcher::Status ImageStitcher::stitch(InputArrayOfArrays images, OutputArray pano)
{
    return stitch(images, cv::noArray(), pano);
}

ImageStitcher::Status ImageStitcher::stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano)
{
    Status status = estimateTransform(images, masks);
    return status;

    if (status != OK)
        return status;
    return composePanorama(pano);
}

ImageStitcher::Status ImageStitcher::estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks)
{
    images.getUMatVector(fullImages_);
    masks.getUMatVector(fullMasks_);

    Status status;
    if ((status = matchImages()) != OK)
        return status;

    if ((status = estimateCameraParams()) != OK)
        return status;

    return OK;
}

ImageStitcher::Status ImageStitcher::composePanorama(OutputArray pano)
{
    INFO("Warping images (auxiliary)... ");

    //    vector<UMat> imgs;
    //    images.getUMatVector(imgs);
    //    if (!imgs.empty()) {
    //        CV_Assert(imgs.size() == fullImages_.size());

    //        UMat img;
    //        seam_est_fullImages_.resize(imgs.size());

    //        for (size_t i = 0; i < imgs.size(); ++i) {
    //            fullImages_[i] = imgs[i];
    //            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
    //            seam_est_fullImages_[i] = img.clone();
    //        }

    //        vector<UMat> seam_est_fullImages_subset;
    //        vector<UMat> fullImages_subset;

    //        for (size_t i = 0; i < indices_.size(); ++i) {
    //            fullImages_subset.push_back(fullImages_[indices_[i]]);
    //            seam_est_fullImages_subset.push_back(seam_est_fullImages_[indices_[i]]);
    //        }

    //        seam_est_fullImages_ = seam_est_fullImages_subset;
    //        fullImages_ = fullImages_subset;
    //    }

    UMat pano_;

#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    vector<Point> corners(fullImages_.size());
    vector<UMat> fullMasks_warped(fullImages_.size());
    vector<UMat> images_warped(fullImages_.size());
    vector<Size> sizes(fullImages_.size());
    vector<UMat> masks(fullImages_.size());

    // Prepare image masks
    for (size_t i = 0; i < fullImages_.size(); ++i) {
        masks[i].create(scaledImgSize_[i], CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    warper_->setScale(1.f);
    for (size_t i = 0; i < fullImages_.size(); ++i) {
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0, 0) *= (float)invScaled_;
        K(0, 2) *= (float)invScaled_;
        K(1, 1) *= (float)invScaled_;
        K(1, 2) *= (float)invScaled_;

        corners[i] = warper_->warp(fullImages_[i], K, cameras_[i].R, INTER_LINEAR_EXACT,
                                   BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper_->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, fullMasks_warped[i]);
    }

    INFO("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");


    images_warped.clear();
    masks.clear();

    INFO("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;

    // double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    vector<ms::detail::CameraParams> cameras_scaled(cameras_);

    UMat full_img, img;
    for (size_t img_idx = 0; img_idx < fullImages_.size(); ++img_idx) {
        INFO("Compositing image #" << indices_[img_idx] + 1);
#if ENABLE_LOG
        int64 compositing_t = getTickCount();
#endif

        // Read image and resize it if necessary
        full_img = fullImages_[img_idx];
        if (!is_compose_scale_set) {
            if (compose_resol_ > 0)
                compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            // compose_seam_aspect = compose_scale / seam_scale_;
            compose_work_aspect = compose_scale / scale_;

            // Update warped image scale
            float warp_scale = static_cast<float>(warped_image_scale_);
            warper_->setScale(warp_scale);
//            warper_ = warperCreator_->create(warp_scale);

            // Update corners and sizes
            for (size_t i = 0; i < fullImages_.size(); ++i) {
                // Update intrinsics
                cameras_scaled[i].ppx *= compose_work_aspect;
                cameras_scaled[i].ppy *= compose_work_aspect;
                cameras_scaled[i].focal *= compose_work_aspect;

                // Update corner and size
                Size sz = fullImgSize_[i];
                if (std::abs(compose_scale - 1) > 1e-1) {
                    sz.width = cvRound(fullImgSize_[i].width * compose_scale);
                    sz.height = cvRound(fullImgSize_[i].height * compose_scale);
                }

                Mat K;
                cameras_scaled[i].K().convertTo(K, CV_32F);
                Rect roi = warper_->warpRoi(sz, K, cameras_scaled[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1) {
#if ENABLE_LOG
            int64 resize_t = getTickCount();
#endif
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
            INFO("  resize time: " << ((getTickCount() - resize_t) / getTickFrequency()) << " sec");
        } else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        INFO(" after resize time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");

        Mat K;
        cameras_scaled[img_idx].K().convertTo(K, CV_32F);

#if ENABLE_LOG
        int64 pt = getTickCount();
#endif
        // Warp the current image
        warper_->warp(img, K, cameras_[img_idx].R, INTER_LINEAR_EXACT, BORDER_REFLECT, img_warped);
        INFO(" warp the current image: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper_->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        INFO(" warp the current image mask: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        // Make sure seam mask has proper size
        dilate(fullMasks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);

        bitwise_and(seam_mask, mask_warped, mask_warped);

        INFO(" other: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");
#if ENABLE_LOG
        pt = getTickCount();
#endif

        if (!is_blender_prepared) {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
        }

        INFO(" other2: " << ((getTickCount() - pt) / getTickFrequency()) << " sec");

        INFO(" feed...");
#if ENABLE_LOG
        int64 feed_t = getTickCount();
#endif
        // Blend the current image
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
        INFO(" feed time: " << ((getTickCount() - feed_t) / getTickFrequency()) << " sec");
        INFO("Compositing ## time: " << ((getTickCount() - compositing_t) / getTickFrequency()) << " sec");
    }

#if ENABLE_LOG
    int64 blend_t = getTickCount();
#endif
    UMat result;
    blender_->blend(result, result_mask_);
    INFO("blend time: " << ((getTickCount() - blend_t) / getTickFrequency()) << " sec");

    INFO("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    result.convertTo(pano, CV_8U);

    return OK;
}

ImageStitcher::Status ImageStitcher::matchImages()
{
    if ((int)fullImages_.size() < 2) {
        ERROR("Need more images!");
        return ERR_NEED_MORE_IMGS;
    }

    bool is_work_scale_set = false;
    features_.resize(fullImages_.size());
    fullImgSize_.resize(fullImages_.size());

#if ENABLE_LOG
    INFO("Finding features...");
    int64 t = getTickCount();
#endif

    vector<UMat> feature_find_imgs(fullImages_.size());
    vector<UMat> feature_find_masks(fullMasks_.size());

    vector<Mat> Homs(fullImages_.size());

    // 1.缩小图像, 提取特征点
    for (size_t i = 0; i < fullImages_.size(); ++i) {
        fullImgSize_[i] = fullImages_[i].size();

        if (registrResol_ < 0.) {
            feature_find_imgs[i] = fullImages_[i];
            scale_ = 1.;
            is_work_scale_set = true;
        } else {
            if (!is_work_scale_set) {  // 控制图片处理分辨率
                scale_ = std::min(1.0, std::sqrt(registrResol_ * 1e6 / fullImgSize_[i].area()));
                is_work_scale_set = true;
            }
            resize(fullImages_[i], feature_find_imgs[i], Size(), scale_, scale_, INTER_LINEAR_EXACT);
        }

        if (!fullMasks_.empty()) {
            resize(fullMasks_[i], feature_find_masks[i], Size(), scale_, scale_, INTER_NEAREST);
        }

        ms::detail::computeImageFeatures(featureFinder_, feature_find_imgs[i], features_[i],
                                         feature_find_masks[i]);
        features_[i].img_idx = (int)i;
        INFO("Features in image #" << i << ": " << features_[i].keypoints.size());

        if (i != baseIndex_)
            computeHomography(fullImages_[baseIndex_], fullImages_[i], Homs[i]);
    }

    // Do it to save memory
    feature_find_imgs.clear();
    feature_find_masks.clear();

#if ENABLE_LOG
    TIMER("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    t = getTickCount();
    INFO("Pairwise matching...");
#endif

    (*featureMatcher_)(features_, pairwise_matches_, matchingMask_);
    //    featureMatcher_->collectGarbage();

#if ENABLE_LOG
    TIMER("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    for (size_t i = 0; i < fullImages_.size(); ++i) {
        if (i != baseIndex_)
            INFO("Homographies diference: (1) " << endl
                                                << pairwise_matches_[i].H << endl
                                                << "    (2) " << endl
                                                << Homs[i]);
    }
#endif

    return OK;
}

ImageStitcher::Status ImageStitcher::estimateCameraParams()
{
    // estimate homography in global frame
    if (!(*motionEstimator_)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;

    for (size_t i = 0; i < cameras_.size(); ++i) {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
#if DEBUG && ENABLE_DEBUG_STITCHER
        INFO("Initial intrinsic parameters #" << indices_[i] << ":\n " << cameras_[i].K());
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
        // INFO("Camera #" << indices_[i] + 1 << ":\n" << cameras_[i].K());
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ =
            static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

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


ImageStitcher::Status ImageStitcher::warpImages(OutputArray _imgs, OutputArray _masks, vector<Point>& _corners)
{
    numImgs_ = fullImages_.size();

    vector<Point> corners(numImgs_);
    vector<UMat> images_warped(numImgs_), masks_warped(numImgs_);
    vector<Size> sizes_warped(numImgs_);

    // Prepare images masks
    vector<UMat> masks(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        masks[i].create(fullImages_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    const double compose_scale = 1.;
    const double compose_work_aspect = compose_scale / scale_;
    INFO("compose_scale: " << compose_scale);
    INFO("compose_work_aspect: " << compose_work_aspect);

    warper_->setScale(compose_work_aspect);

    // Update corners and sizes
    for (size_t i = 0; i < numImgs_; ++i) {
        // Update intrinsics
        cameras_[i].focal *= compose_work_aspect;
        cameras_[i].ppx *= compose_work_aspect;
        cameras_[i].ppy *= compose_work_aspect;

        // Update corner and size
        Size sz = fullImgSize_[i];
        if (std::abs(compose_scale - 1) > 1e-1) {
            sz.width = cvRound(fullImgSize_[i].width * compose_scale);
            sz.height = cvRound(fullImgSize_[i].height * compose_scale);
        }

        Mat K;
        cameras_[i].K().convertTo(K, CV_32F);
        Rect roi = warper_->warpRoi(sz, K, cameras_[i].R);
        INFO("scaled corner to composited corner: " << corners[i] << " ===> " << roi.tl());
        INFO("scaled size to composited size: " << sz << " ===> " << roi.size());
        corners_[i] = roi.tl();
        warpedImgSize_[i] = roi.size();
    }

    _imgs.assign(images_warped);
    _masks.assign(masks_warped);
    _corners = corners;

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
        // float x = keypoints1[it->queryIdx].pt.x;
        // float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(keypoints1[it->queryIdx].pt);

        // x = keypoints2[it->trainIdx].pt.x;
        // y = keypoints2[it->trainIdx].pt.y;
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
