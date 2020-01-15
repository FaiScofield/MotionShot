#ifndef FRAMESDIFFERENCE_H
#define FRAMESDIFFERENCE_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ms
{

class FramesDifference
{
public:
    FramesDifference() : _delta(2), _structureSize(5) {}
    FramesDifference(int d, int s) : _delta(d), _structureSize(s) {}

    inline void setDelta(int delta) { _delta = delta; }
    inline int  getDelta() const { return _delta; }
    inline void setStructureSize(int size) { _structureSize = size; }

    void apply(const cv::Mat& img, cv::Mat& mask);

private:
    cv::Mat getMotionMask2();
    cv::Mat getMotionMask3();

    int _delta;          // 帧差间隔
    int _structureSize;  // 结构元尺寸
    int _iteration;      // 形态学运算次数

    cv::Mat _image1, _image2, _image3;
    cv::Mat _diff1, _diff2;
};

}  // namespace ms
#endif  // FRAMESDIFFERENCE_H
