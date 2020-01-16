#ifndef BASE_BACKGROUND_SUBTRACTOR_H
#define BASE_BACKGROUND_SUBTRACTOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

class BaseBackgroundSubtractor
{
public:
    BaseBackgroundSubtractor() : _isFirstFrame(true), _structureSize(5) {}
    virtual ~BaseBackgroundSubtractor() {}

    // virtual void init();

    inline void setStructureSize(int size) { _structureSize = size; }

    void filterMask(cv::Mat& mask, int size = 0);

    /*virtual*/ void apply(const cv::Mat& input, cv::Mat& mask) {}

private:
    bool _isFirstFrame;
    int  _structureSize;
};

}  // namespace ms

#endif  // BASE_BACKGROUND_SUBTRACTOR_H
