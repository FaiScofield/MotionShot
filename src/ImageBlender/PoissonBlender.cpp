#include "PoissonBlender.h"
#include <opencv2/photo.hpp>

namespace ms
{

using namespace std;
using namespace cv;

// PoissionBlender::PoissionBlender()
//{
//    a_ = 0.f;
//}

//void PoissionBlender::prepare(const vector<Point>& corners, const vector<Size>& sizes) {}

//void PoissionBlender::prepare(Rect dst_roi) {}

void PoissonBlender::feed(cv::InputArray src, cv::InputArray mask, cv::Point center)
{
    src_ = src.getMat();
    src_mask_ = mask.getMat();
    roi_center_ = center;
}

void PoissonBlender::blend(cv::InputOutputArray dst, cv::InputOutputArray dst_mask)
{
    cv::seamlessClone(src_, dst, dst_mask, roi_center_, dst, cv::NORMAL_CLONE);
}

}  // namespace ms
