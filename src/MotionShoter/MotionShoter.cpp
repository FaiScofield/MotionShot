#include "MotionShoter.h"
#include "utility.h"

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

using namespace std;
using namespace cv;

void MotionShot::Apply(int d)
{
    assert(d > 1);

    SetDelta(d);

    _vForegrounds.resize(_N);
    for (size_t i = 0; i < _N; ++i) {
        const Mat& frame = _vFrames[i];
//        _pForegroundExtractor->ExtractForeground(frame);
//        _vForegrounds[i] = _pForegroundExtractor->GetForeground();
    }

    _result = _vFrames[0].clone();
    for (size_t i = 0; i < _N; ++i) {
        if (i % _delta == 0) {
            Mat front;
            cv::bitwise_and(_vFrames[i], _vForegrounds[i], front);
            _result += front;
        }
    }
}


}  // namespace ms
