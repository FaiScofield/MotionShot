#include "ImageStitcher.h"

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define SHOW_RESULTS 0

namespace ms
{

using namespace std;
using namespace cv;

ImageStitcher::ImageStitcher() : _featureType(SIFT)
{
    _featureExtractor = cv::xfeatures2d::SIFT::create(200);
}

ImageStitcher::ImageStitcher(FeatureType ft) : _featureType(ft)
{
    switch (_featureType) {
    case SURF:
        _featureExtractor = cv::xfeatures2d::SURF::create();
        break;
    case ORB:
        _featureExtractor = cv::ORB::create();
        break;
    default:
        _featureExtractor = cv::xfeatures2d::SIFT::create(200);
        break;
    }
}

Mat ImageStitcher::computeHomography(const Mat& img1, const Mat& img2)
{
    if (img1.empty() || img2.empty())
        return Mat();

    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1, descriptors2;

    // 检测SIFT特征并生成描述子
    _featureExtractor->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    _featureExtractor->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // 使用欧氏距离和交叉匹配策略进行图像匹配
    BFMatcher matcher(NORM_L2, true);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

#if SHOW_RESULTS
    Mat imageMatches;
    drawMatches(img1, keypoints1,     // 1st image and its keypoints
                img2, keypoints2,     // 2nd image and its keypoints
                matches,                // the matches
                imageMatches,           // the image produced
                Scalar(255, 255, 255),  // color of the lines
                Scalar(255, 255, 255),  // color of the keypoints
                vector<char>(), 2);
    imshow("Matches (pure rotation case)", imageMatches);
#endif

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
    cout << "number of matched points: " << points1.size() << " & " << points2.size() << endl;

    // 使用RANSAC算法估算单应矩阵
    vector<char> inliers;
    Mat homography = findHomography(points1, points2,  // corresponding points
                                    inliers,           // outputed inliers matches
                                    RANSAC,            // RANSAC method
                                    1.);               // max distance to reprojection point

    // 用单应矩阵对图像进行变换
    Mat warped;
    warpPerspective(img2,                               // input image
                    warped,                               // output image
                    homography.inv(),                           // homography
                    Size(1.2*img1.cols, img1.rows));  // size of output image

#if SHOW_RESULTS
    //画出局内匹配项
    drawMatches(img1, keypoints1,     // 1st image and its keypoints
                img2, keypoints2,     // 2nd image and its keypoints
                matches,                // the matches
                imageMatches,           // the image produced
                Scalar(255, 255, 255),  // color of the lines
                Scalar(255, 255, 255),  // color of the keypoints
                inliers, 2);

    imshow("Homography inlier points", imageMatches);

    img1.copyTo(warped.colRange(0, img1.cols));
    imshow("Image mosaic", warped);
#endif


    return homography;
}


}  // namespace ms
