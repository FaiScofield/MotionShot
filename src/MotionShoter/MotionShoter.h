#ifndef MOTION_SHOT_H
#define MOTION_SHOT_H

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace ms
{

//class BackgroundExtractor;

class MotionShot
{
public:
    MotionShot() : _N(0), _delta(3) {}
//    MotionShot(BackgroundExtractor* fe) : _pForegroundExtractor(fe), _N(0), _delta(3) {}
    ~MotionShot() {}

    inline void SetDelta(int d) { _delta = d; }
    inline int GetDelta() const { return _delta; }
    inline cv::Mat GetResult() { return _result; }
    inline std::vector<cv::Mat> GetForeground() const { return _vForegrounds; }

    void Apply(int d = 2);

private:
//    BackgroundExtractor* _pForegroundExtractor;

    std::vector<cv::Mat> _vFrames;
    std::vector<cv::Mat> _vForegrounds;
    std::vector<cv::Mat> _vForegroundGTs;

    size_t _N;
    int _delta;
    cv::Mat _result;
};

}  // namespace ms

#endif
