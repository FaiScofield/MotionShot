#ifndef BS_MOG2_CV_H
#define BS_MOG2_CV_H

#include "BaseMotionDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

class BS_MOG2_CV : public BaseMotionDetector
{
public:
    BS_MOG2_CV(cv::BackgroundSubtractorMOG2* bs) : BaseMotionDetector(), _extractor(bs) {}
    ~BS_MOG2_CV() { _extractor = nullptr; }

    inline void applyMask(const cv::Mat& mask);

    void apply(const cv::Mat& input, cv::Mat& mask) override;

private:
    cv::BackgroundSubtractorMOG2* _extractor;
    cv::Mat_<uint8_t> _foreground;
};

}  // namespace ms
#endif  // BS_MOG2_CV_H
