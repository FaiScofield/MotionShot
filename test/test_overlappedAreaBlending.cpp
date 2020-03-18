#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/photo.hpp>
#include <iostream>
#include "MotionShoter/utility.h"

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    const Mat img1 = imread("/home/vance/dataset/rk/Phone/withGT3/demo4/image_full/7.jpg");
    const Mat img2 = imread("/home/vance/dataset/rk/Phone/withGT3/demo4/image_full/8.jpg");
    const Mat mask1 = imread("/home/vance/dataset/rk/Phone/withGT3/demo4/gt_rect_baidu/7.png", 0);
    const Mat mask2 = imread("/home/vance/dataset/rk/Phone/withGT3/demo4/gt_rect_baidu/8.png", 0);
    const Rect rect1(2659, 1208, 3143 - 2659, 2394 - 1208);
    const Rect rect2(2644, 1001, 3471 - 2644, 2542 - 1001);
    Mat foreMask1 = Mat::zeros(img1.size(), CV_8UC1), foreMask2 = Mat::zeros(img2.size(), CV_8UC1);
    mask1.copyTo(foreMask1(rect1));
    mask2.copyTo(foreMask2(rect2));

    assert(!img1.empty() && !img2.empty() && !foreMask1.empty() && !foreMask2.empty());
    assert(mask1.size() == rect1.size());
    assert(mask2.size() == rect2.size());


    // 计算重叠区域overlappedArea, 重叠区域边缘edge, 白边区域gapMask
    Mat overlappedArea, edge, ignoreArea;
    bitwise_and(foreMask1, 255, overlappedArea, foreMask2);     // 得到重叠区域
    morphologyEx(overlappedArea, edge, MORPH_GRADIENT, Mat());  // 得到重叠区域边缘
    erode(foreMask2, ignoreArea, Mat());    // 考虑覆盖
    bitwise_and(edge, 0, edge, ignoreArea);

    const int s = 2 * 7 + 1;
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(s, s));

    Mat gapMask, overlappedEroded, maksEroded;
    erode(overlappedArea, overlappedEroded, kernel);
    erode(foreMask2, maksEroded, kernel);
    gapMask = overlappedArea - overlappedEroded;
    bitwise_and(gapMask, 0, gapMask, maksEroded);

    Mat overlappedAreaEdge = img2.clone();
    overlappedAreaEdge.setTo(Scalar(255,0,0), edge);
    vector<vector<Point>> contours;
    findContours(gapMask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    drawContours(overlappedAreaEdge, contours, -1, Scalar(0,255,0));

    ms::namedLargeWindow("重叠区域(边缘)");
    imshow("重叠区域(边缘)", overlappedAreaEdge(rect2));

    // 将白边区域进行泊松融合
    Mat dst = Mat::zeros(img1.size(), CV_8UC3);
    img1.copyTo(dst, foreMask1);
    Mat pano = dst.clone();
    Mat foreMaskNew2 = foreMask2 - gapMask;
    img2.copyTo(dst, foreMaskNew2);
    img2.copyTo(pano, foreMask2);

    pano.setTo(Scalar(255,0,0), edge);
    drawContours(pano, contours, -1, Scalar(0,255,0));
    imwrite("/home/vance/output/ms/pano.jpg", pano);

//    ms::namedLargeWindow("gapMask");
//    imshow("gapMask", gapMask);
//    ms::namedLargeWindow("foreMaskNew2");
//    imshow("foreMaskNew2", foreMaskNew2);

    Mat result;
    Rect rec = boundingRect(gapMask);
    Point center = (rec.tl() + rec.br()) * 0.5;
    seamlessClone(img2, dst, gapMask, center, result, NORMAL_CLONE);
    ms::namedLargeWindow("result");
    imshow("result", result);
    imwrite("/home/vance/output/ms/result.jpg", result);

    waitKey(0);

    return 0;
}
