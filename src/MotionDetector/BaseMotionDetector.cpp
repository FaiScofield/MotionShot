#include "BaseMotionDetector.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace ms {

void BaseMotionDetector::filterMask(cv::Mat &mask, int size)
{
    size = size < 2 ? _structureSize : size;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
    cv::erode(mask, mask, cv::Mat());   // 腐蚀
    cv::dilate(mask, mask, cv::Mat());  // 膨胀
    cv::dilate(mask, mask, element);
    cv::erode(mask, mask, element);
//    cv::erode(mask, mask, element);
//    cv::dilate(mask, mask, element);
}

}
