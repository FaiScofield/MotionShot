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
//    OpticalFlower(cv::DenseOpticalFlow* dof): _denseFlow(dof) {}

    cv::Mat getFlow() const { return _flow; }
    cv::Mat getWeightMask() const { return _weightMask; }

    void setCompareToPano(const cv::Mat& pano);
//    void flowToMask(const cv::Mat& flow, cv::Mat& mask);
    void flowToWeightMask(const cv::Mat& flow, cv::Mat& mask);

    void apply(const cv::Mat& img, cv::Mat& mask);

    void apply(const std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& masks);

private:
//    cv::DenseOpticalFlow* _denseFlow;
    cv::Ptr<cv::DenseOpticalFlow> _denseFlow;
    cv::Mat _imgLast, _imgCurr, _flow, _weightMask;

    bool _compareToPano = false;
    cv::Mat _pano, _panoGray;
};

}  // namespace ms
#endif  // OPTICALFLOWER_H
