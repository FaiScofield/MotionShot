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
    OpticalFlower(cv::DenseOpticalFlow* dof): denseFlow_(dof) {}

    void apply(const cv::Mat& img, cv::Mat& mask);

private:
    cv::DenseOpticalFlow* denseFlow_;
    cv::Mat imgLast_, imgCurr;

};

}  // namespace ms
#endif  // OPTICALFLOWER_H
