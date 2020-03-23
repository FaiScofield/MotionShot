#include "BaseMotionDetector.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace ms {

void BaseMotionDetector::filterMask(cv::Mat &mask, int size)
{
    size = size < 2 ? _structureSize : size;

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
}

}
