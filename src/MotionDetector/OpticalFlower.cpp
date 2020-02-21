#include "OpticalFlower.h"
#include "utility.h"

#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>

namespace ms
{

using namespace std;
using namespace cv;

OpticalFlower::OpticalFlower()
{
#ifdef USE_OPENCV4
    _denseFlow = dynamic_cast<DenseOpticalFlow*>(
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM).get());
#else
    //    _denseFlow = FarnebackOpticalFlow::create();
    _denseFlow = DualTVL1OpticalFlow::create();

//    _denseFlow = dynamic_cast<DenseOpticalFlow*>(FarnebackOpticalFlow::create().get());
//    _denseFlow = dynamic_cast<DenseOpticalFlow*>(DualTVL1OpticalFlow::create().get());
#endif
}

void OpticalFlower::setCompareToPano(const Mat& pano)
{
    pano.copyTo(_pano);
    cvtColor(pano, _panoGray, COLOR_BGR2GRAY);
    _compareToPano = true;
}

void OpticalFlower::apply(const Mat& img, Mat& mask)
{
    if (img.channels() == 3)
        cvtColor(img, _imgCurr, COLOR_BGR2GRAY);
    else if (img.channels() == 1)
        _imgCurr = img.clone();

    if (_compareToPano) {
        _denseFlow->calc(_panoGray, _imgCurr, _flow);
    } else {
        if (isFirstFrame()) {
            _imgLast = _imgCurr.clone();
            setFirstFrameFlag(false);
            mask.release();
            return;
        }

        _denseFlow->calc(_imgLast, _imgCurr, _flow);
    }

    Mat flowColor;
    showFlow(_flow, flowColor);
    //    imshow("flowColor", flowColor);
    //    waitKey(30);

    flowToWeightMask(_flow, _weightMask);
    mask = _weightMask.clone();
    //    flowToMask(flowColor, mask);

    // TODO
    if (!_compareToPano)
        _imgLast = _imgCurr.clone();
}

void OpticalFlower::apply(const std::vector<cv::Mat>& vImgs, std::vector<cv::Mat>& vMasks)
{
    assert(!vImgs.empty());

    const size_t N = vImgs.size();

    vector<Mat> imgs;
    imgs.reserve(N);
    if (vImgs[0].channels() == 3)
        for_each(vImgs.begin(), vImgs.end(), [&](const Mat& img) {
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            imgs.push_back(gray);
        });
    else if (vImgs[0].channels() == 1)
        imgs = vImgs;

    vMasks.clear();
    vMasks.resize(N);
    Mat lastFrame = imgs[0], currFrame;
    for (size_t i = 1; i < N; ++i) {
        Mat flow, weight;
        currFrame = imgs[i];
        _denseFlow->calc(lastFrame/*imgs[0]*/, currFrame, flow);
        flowToWeightMask(flow, weight);
        vMasks[i] = weight.clone();

//        Mat flowColor;
//        showFlow(flow, flowColor);
//        imshow("flowColor", flowColor);
//        imshow("flow weight", weight);
//        waitKey(30);
    }

    // 反向计算首帧的光流
    Mat firstFlow, firstWeight;
    _denseFlow->calc(imgs[1], imgs[0], firstFlow);
    flowToWeightMask(firstFlow, firstWeight);
    vMasks[0] = firstWeight.clone();


}

// void OpticalFlower::flowToMask(const Mat &flow, Mat &mask)
//{
//    assert(flow.channels() == 2);

//    Mat flow_uv[2], mag, ang, hsv, hsv_split[3], bgr, color;
//    split(flow, flow_uv);
//    multiply(flow_uv[1], -1, flow_uv[1]);
//    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true); // 笛卡尔转极坐标系
//    normalize(mag, mag, 0, 1, NORM_MINMAX);
//    hsv_split[0] = ang;
//    hsv_split[1] = mag;
//    hsv_split[2] = Mat::ones(ang.size(), ang.type());
//    merge(hsv_split, 3, hsv);
//    cvtColor(hsv, bgr, COLOR_HSV2BGR);    // bgr1 type = CV_32FC3
//    bgr.convertTo(color, CV_8UC3, 255, 0);

//    const Mat kernel = getStructuringElement(MORPH_RECT, Size(structureSize(), structureSize()));

//    Mat flowU;
//    cvtColor(color, flowU, COLOR_BGR2GRAY);
//    bitwise_not(flowU, flowU);
//    threshold(flowU, flowU, 30, 255, THRESH_BINARY); // THRESH_OTSU, THRESH_BINARY
//    erode(flowU, flowU, kernel);
//    dilate(flowU, flowU, kernel);
//    dilate(flowU, flowU, kernel);

//    mask = flowU.clone();
//}

void OpticalFlower::flowToWeightMask(const Mat& flow, Mat& weight)
{
    assert(flow.channels() == 2);

    Mat flow_uv[2], mag, ang, hsv, hsv_split[3], bgr, color;
    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);  // 笛卡尔转极坐标系
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, bgr, COLOR_HSV2BGR);  // bgr1 type = CV_32FC3
    bgr.convertTo(color, CV_8UC3, 255, 0);

    cvtColor(color, weight, COLOR_BGR2GRAY);
    bitwise_not(weight, weight);
}

}  // namespace ms
