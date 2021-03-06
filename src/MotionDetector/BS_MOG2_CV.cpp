#include "BS_MOG2_CV.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace ms
{

using namespace cv;


void BS_MOG2_CV::applyMask(const cv::Mat& mask)
{
    bitwise_and(_foreground, mask, _foreground);
}

void BS_MOG2_CV::apply(const Mat& input, Mat& mask)
{
    if (isFixedBackground())
        _extractor->apply(input, mask, 0);
    else
        _extractor->apply(input, mask);
    if (mask.empty())
        return;

    filterMask(mask);
    _foreground = mask.clone();
}


}  // namespace ms
