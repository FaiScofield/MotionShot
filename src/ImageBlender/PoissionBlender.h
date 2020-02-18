#ifndef POISSIONBLENDER_H
#define POISSIONBLENDER_H

#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/photo.hpp>

namespace ms
{


class PoissionBlender : public cv::detail::Blender
{
public:
    PoissionBlender() /*: cv::detail::Blender() */ {}
    ~PoissionBlender() {}

//    void prepare(const std::vector<cv::Point>& corners, const std::vector<cv::Size>& sizes) {}

//    void prepare(cv::Rect dst_roi) {}

    void feed(cv::InputArray img, cv::InputArray mask, cv::Point tl)
    {
        src_ = img.getMat();
        src_mask_ = mask.getMat();
        to_place_ = tl;
    }

    void blend(cv::InputOutputArray dst, cv::InputOutputArray dst_mask)
    {

        cv::seamlessClone(src_, dst, dst_mask, to_place_, result_, cv::NORMAL_CLONE);

//        cv::detail::Blender::blend(dst, dst_mask);
    }

    cv::Mat getResult() const { return result_; }

private:
    cv::Mat src_;
    cv::Mat src_mask_;
    cv::Point to_place_;

    cv::Mat result_;
};

}  // namespace ms

#endif  // POISSIONBLENDER_H
