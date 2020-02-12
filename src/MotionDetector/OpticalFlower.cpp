#include "OpticalFlower.h"

namespace ms
{

using namespace cv;

OpticalFlower::OpticalFlower()
{
#ifdef USE_OPENCV4
    denseFlow_ = dynamic_cast<DenseOpticalFlow*>(
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM).get());
#else
    denseFlow_ = dynamic_cast<DenseOpticalFlow*>(FarnebackOpticalFlow::create().get());
//    denseFlow_ = dynamic_cast<DualTVL1OpticalFlow*>(DualTVL1OpticalFlow::create().get());
#endif
}

void OpticalFlower::apply(const Mat& img, Mat& mask)
{
    Mat imgGray;
    if (img.channels() == 3)
        cvtColor(img, imgGray, COLOR_BGR2GRAY);

    if (isFirstFrame()) {
        imgLast_ = imgGray.clone();
        setFirstFrameFlag(false);
        mask.release();
        return;
    }

    imgCurr = imgGray;
    denseFlow_->calc(imgLast_, imgCurr, mask);  // get mask type CV_32FC2
    // TODO
    imgGray.copyTo(imgLast_);
}

}  // namespace ms
