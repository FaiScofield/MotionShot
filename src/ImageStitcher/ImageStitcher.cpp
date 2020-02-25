#include "ImageStitcher.h"
#include "MotionShoter/utility.h"
#include "ImageBlender/cvBlenders.h"
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/stitching.hpp>

namespace ms
{

using namespace std;
using namespace cv;

// ImageStitcher::ImageStitcher() : _featureType(surf)
//{
//    _featureExtractor = cv::xfeatures2d::SURF::create();
//    _matcher = DescriptorMatcher::create("FlannBased");
//}

ImageStitcher::ImageStitcher(FeatureType ft, MatchType mt) : _featureType(ft), _matchType(mt)
{
    switch (_featureType) {
    case sift:
        _featureExtractor = cv::xfeatures2d::SIFT::create(400);
        break;
    case orb:
        _featureExtractor = cv::ORB::create();
        break;
    case akeke:
        _featureExtractor = cv::AKAZE::create();
        break;
    case surf:
    default:
        _featureExtractor = cv::xfeatures2d::SURF::create();
        break;
    }

    switch (_matchType) {
    case bf:
        _matcher = cv::BFMatcher::create();
        break;
    case flann:
    default:
        _matcher = cv::FlannBasedMatcher::create();
        break;
    }
}

// bool ImageStitcher::setMatcher(const string &descriptorMatcherType){
//    _matcher = DescriptorMatcher::create(descriptorMatcherType);
//    return !_matcher.empty();
//}
bool ImageStitcher::stitch(const std::vector<Mat>& images, Mat& pano)
{
    if (images.size() < 2)
        return false;

    _ImgSizeInput = images[0].size();
    Mat warpedMask1;
    return stitch(images.front(), images.back(), pano, warpedMask1);
}

//! TODO 要以尾帧做基准
bool ImageStitcher::stitch(const Mat& img1, const Mat& img2, Mat& pano, Mat& warpedMask2)
{
//#define DEBUG_RESULT_STITCH

    Mat H21 = computeHomography(img1, img2);
    if (H21.empty())
        return false;

    Mat H12 = H21.inv(DECOMP_SVD);

    Mat warpedImg2;
    WarpedCorners corners = getWarpedCorners(img2, H12);
//    cout << "Direction  = " << corners.direction << endl;
    assert(corners.direction = RIGHT);  //! TODO


/*
    int panoCols;
    switch (corners.direction) {
    case LEFT:  //! TODO 显示不全问题!
        panoCols = img2.cols - min(cvRound(min(corners.tl.x, corners.bl.x)), 0);
        cout << "min x  = " << min(corners.tl.x, corners.bl.x) << endl;
        break;
    case RIGHT: //! 目前只能用前景从左到右的情况! 且以首帧为基准.
        panoCols = max(cvRound(max(corners.tr.x, corners.br.x)), img2.cols);
        cout << "max x  = " << max(corners.tr.x, corners.br.x) << endl;
        break;
    case UNKNOWN:
        WARNING("无法确定前景移动方向!");
        panoCols = max(cvRound(max(corners.tr.x, corners.br.x)), img2.cols) -
                    min(cvRound(min(corners.tl.x, corners.bl.x)), 0);
        cout << "min x  = " << min(corners.tl.x, corners.bl.x) << endl;
        cout << "max x  = " << max(corners.tr.x, corners.br.x) << endl;
    default:
        break;
    }
    cout << "pano_cols = " << panoCols << endl;
    warpPerspective(img2, warpedImg2, H12, Size(panoCols, img1.rows));
*/

    // get warped image and mask
    int panoCols = max(cvCeil(max(corners.tr.x, corners.br.x)), img1.cols);
    warpPerspective(img2, warpedImg2, H12, Size(panoCols, img2.rows));

    Mat mask2(img2.size(), CV_8UC1);
    mask2.setTo(255);
    warpPerspective(mask2, warpedMask2, H12, Size(panoCols, img1.rows));

#ifdef DEBUG_RESULT_STITCH
    Mat tmp1, tmp2;
    cvtColor(warpedMask2, tmp1, COLOR_GRAY2BGR);
    vconcat(warpedImg2, tmp1, tmp2);
    imshow("img2 & mask warped", tmp2);
#endif

//    pano = warpedImg2.clone();
//    switch (corners.direction) {
//    case LEFT:
//        img2.copyTo(pano.colRange(pano.cols - img2.cols, pano.cols));
//        break;
//    case RIGHT:
//        img2.copyTo(pano.colRange(0, img2.cols));
//        break;
//    case UNKNOWN:
//    default:
//        int left = -min(cvRound(min(corners.tl.x, corners.bl.x)), 0);
//        img2.copyTo(pano.colRange(left, left + img2.cols));
//        break;
//    }

    Mat warpedImg1 = Mat::zeros(warpedImg2.size(), CV_8UC3);
    img1.copyTo(warpedImg1.colRange(0, img1.cols));
    alphaBlend(warpedImg1, warpedImg2, corners, pano);

    imshow("Pano with blending", pano);
    waitKey(0);
//    return (!pano.empty());
}

Mat ImageStitcher::computeHomography(const Mat& img1, const Mat& img2)
{
//#define DEBUG_RESULT_HOMOGRAPHY

    if (img1.empty() || img2.empty())
        return Mat();

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;

    _featureExtractor->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    _featureExtractor->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    vector<DMatch> matches;
    switch (_matchType) {
    case bf:
        _matcher->match(descriptors1, descriptors2, matches);
        break;
    case flann:
    default:
        vector<vector<DMatch>> vvMatches;
        _matcher->knnMatch(descriptors1, descriptors2, vvMatches, 2);
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
    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it) {
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(Point2f(x, y));

        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        points2.push_back(Point2f(x, y));
    }
    if (points1.empty() || points2.empty()) {
        cerr << "图像差距太大!" << endl;
        return Mat();
    }

    // 使用RANSAC算法估算单应矩阵
    vector<char> inliers;
    Mat homography = findHomography(points1, points2, inliers, RANSAC, 3.);
    if (homography.empty()) {
        return Mat();
    }

#ifdef DEBUG_RESULT_HOMOGRAPHY
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

    return homography;
}

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
    br1_overlapped.x = min(_ImgSizeInput.width, cvCeil(max(corners.tr.x, corners.br.x)) + expBorder);
    br1_overlapped.y = min(_ImgSizeInput.height, cvCeil(max(corners.bl.y, corners.br.y)) + expBorder);
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


}  // namespace ms
