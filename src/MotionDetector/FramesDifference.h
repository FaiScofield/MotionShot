#ifndef FRAMES_DIFFERENCE_H
#define FRAMES_DIFFERENCE_H

#include "BaseMotionDetector.h"
#include <opencv2/core/core.hpp>

namespace ms
{

class FramesDifference : public BaseMotionDetector
{
public:
    FramesDifference() : BaseMotionDetector(), _delta(2), _structureSize(5) {}
    FramesDifference(int d, int s) : BaseMotionDetector(), _delta(d), _structureSize(s) {}
    ~FramesDifference() {}

    inline void setDelta(int delta) { _delta = delta; }
    inline int  getDelta() const { return _delta; }
    inline void setStructureSize(int size) { _structureSize = size; }

    void apply(const cv::Mat& img, cv::Mat& mask);

private:
    // void filterMask(cv::Mat& mask);
    cv::Mat getMotionMask2();
    cv::Mat getMotionMask3();

    int _delta;          // 帧差间隔
    int _structureSize;  // 结构元尺寸
    int _iteration;      // 形态学运算次数

    cv::Mat _image1, _image2, _image3;
    cv::Mat _diff1, _diff2;
};

}  // namespace ms
#endif  // FRAMES_DIFFERENCE_H
