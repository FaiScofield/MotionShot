#include "OpticalFlower.h"
#include "utility.h"

#include <opencv2/highgui.hpp>

namespace ms
{

using namespace cv;

OpticalFlower::OpticalFlower()
{
#ifdef USE_OPENCV4
    denseFlow_ = dynamic_cast<DenseOpticalFlow*>(
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM).get());
#else
//    denseFlow_ = dynamic_cast<DenseOpticalFlow*>(FarnebackOpticalFlow::create().get());
    _denseFlow = dynamic_cast<DualTVL1OpticalFlow*>(DualTVL1OpticalFlow::create().get());
#endif
}

void OpticalFlower::setCompareToPano(const Mat& pano)
{
    pano.copyTo(_pano);
    cvtColor(pano, _panoGray, COLOR_BGR2GRAY);
    _compareToPano = true;
}

void OpticalFlower::apply(const Mat& img, Mat& mask)
{
    cvtColor(img, _imgCurr, COLOR_BGR2GRAY);
//    _imgCurr = img.clone();

    if (_compareToPano) {
        _denseFlow->calc(_panoGray, _imgCurr, _flow);
    } else {
        if (isFirstFrame()) {
            _imgLast = _imgCurr.clone();
            setFirstFrameFlag(false);
            mask.release();
            return;
        }

        _denseFlow->calc(_imgLast, _imgCurr, _flow);
    }

    Mat flowColor;
    showFlow(_flow, flowColor);
    imshow("flowColor", flowColor);
    waitKey(0);

    // TODO
    if (!_compareToPano)
        _imgLast = _imgCurr.clone();
}

}  // namespace ms
