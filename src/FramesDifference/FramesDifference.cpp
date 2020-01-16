#include "FramesDifference.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

namespace ms {

using namespace cv;
using namespace std;

void FramesDifference::apply(const cv::Mat& input, cv::Mat& mask)
{
    assert(!input.empty());

    Mat img;
    if (input.type() == CV_8UC3)
        cvtColor(input, img, COLOR_BGR2GRAY);
    else
        img = input;

    if (_delta == 3) {
        img.copyTo(_image3);
        mask = getMotionMask3();
    } else {
        img.copyTo(_image2);
        mask = getMotionMask2();
    }
}

Mat FramesDifference::getMotionMask2()
{
    assert(!_image2.empty());

    Mat diff;
    Mat img1, img2;

    if (_image1.empty()) {
        _image1 = _image2.clone();
        return Mat(_image2.size(), CV_8UC1, Scalar(0));
    } else {
        _image1.convertTo(img1, CV_32FC1);
        _image2.convertTo(img2, CV_32FC1);
        absdiff(img1, img2, diff);

        diff.convertTo(diff, CV_8UC1);

        threshold(diff, diff, 25, 255, CV_THRESH_BINARY);

        filterMask(diff, _structureSize);

        _diff1 = diff.clone();
        _image1 = _image2.clone();
        return diff;
    }
}

Mat FramesDifference::getMotionMask3()
{
    assert(!_image3.empty());

    Mat diff;
    Mat img1, img2, img3;

    if (_image1.empty()) {
        _image1 = _image3.clone();
        return Mat(_image3.size(), CV_8UC1, Scalar(0));
    } else if (_image2.empty()) {
        assert(!_image1.empty());
        _image2 = _image3.clone();
        _image1.convertTo(img1, CV_32FC1);
        _image2.convertTo(img2, CV_32FC1);
        absdiff(img1, img2, _diff1);
        return Mat(_image3.size(), CV_8UC1, Scalar(0));
    } else {
        assert(!_image1.empty());
        assert(!_image2.empty());
        assert(!_diff1.empty());
        assert(_diff1.type() == CV_32FC1);

        _image1.convertTo(img1, CV_32FC1);
        _image2.convertTo(img2, CV_32FC1);
        _image3.convertTo(img3, CV_32FC1);
        absdiff(img2, img3, _diff2);

        bitwise_and(_diff1, _diff2, diff);
        _diff1 = _diff2.clone();

        diff.convertTo(diff, CV_8UC1);
        threshold(diff, diff, 25, 255, CV_THRESH_BINARY);

        filterMask(diff, _structureSize);

        _image1 = _image2.clone();
        _image2 = _image3.clone();
        return diff;
    }
}

//void FramesDifference::filterMask(Mat &mask, int size)
//{
//    Mat element = getStructuringElement(MORPH_RECT, Size(size, size));
//    erode(mask, mask, Mat());   // 腐蚀
//    dilate(mask, mask, Mat());  // 膨胀
//    dilate(mask, mask, element);
//    erode(mask, mask, element);
//    erode(mask, mask, element);
//    dilate(mask, mask, element);
//}

}
