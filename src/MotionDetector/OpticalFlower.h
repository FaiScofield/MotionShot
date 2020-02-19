#ifndef OPTICALFLOWER_H
#define OPTICALFLOWER_H

#include "BaseMotionDetector.h"
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>

namespace ms
{

class OpticalFlower : public BaseMotionDetector
{
public:
    OpticalFlower();
    OpticalFlower(cv::DenseOpticalFlow* dof): _denseFlow(dof) {}

    void setCompareToPano(const cv::Mat& pano);

    void apply(const cv::Mat& img, cv::Mat& mask);

private:
    cv::DenseOpticalFlow* _denseFlow;
    cv::Mat _imgLast, _imgCurr, _flow;

    bool _compareToPano = false;
    cv::Mat _pano, _panoGray;
};

}  // namespace ms
#endif  // OPTICALFLOWER_H
