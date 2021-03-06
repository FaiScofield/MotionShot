#ifndef BASE_MOTION_DETECTOR_H
#define BASE_MOTION_DETECTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

class BaseMotionDetector
{
public:
    BaseMotionDetector() : _isFirstFrame(true), _isFixedBackground(false), _structureSize(5) {}
    virtual ~BaseMotionDetector() {}

    // virtual void init();

    inline void setStructureSize(int size) { _structureSize = size; }
    inline int structureSize() const { return _structureSize; }
    inline void setFirstFrameFlag(bool flag) { _isFirstFrame = flag; }
    inline bool isFirstFrame() const { return _isFirstFrame; }
    inline void setFixedBackground(bool flag) { _isFixedBackground = flag; }
    inline bool isFixedBackground() const { return _isFixedBackground; }

    void filterMask(cv::Mat& mask, int size = 0);

    virtual void apply(const cv::Mat& input, cv::Mat& mask){}

private:
    bool _isFirstFrame;
    bool _isFixedBackground;
    int  _structureSize;

    cv::Mat _background;
};

}  // namespace ms

#endif  // BASE_MOTION_DETECTOR_H
