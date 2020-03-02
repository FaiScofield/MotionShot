#ifndef POISSIONBLENDER_H
#define POISSIONBLENDER_H

#include <opencv2/photo.hpp>
#include "cvBlenders.h"

namespace ms
{


class PoissonBlender : public cvBlender
{
public:
    PoissonBlender() {}
    ~PoissonBlender() {}

    // void prepare(const std::vector<cv::Point>& corners, const std::vector<cv::Size>& sizes) {}

    // void prepare(cv::Rect dst_roi) {}

    void feed(cv::InputArray src, cv::InputArray mask, cv::Point center) override;

    void blend(cv::InputOutputArray dst, cv::InputOutputArray dst_mask) override;

    cv::Mat getResult() const { return result_; }

    // 求解梯度
    void computeGradientX(const cv::Mat& img, cv::Mat& gx);
    void computeGradientY(const cv::Mat& img, cv::Mat& gy);
    // 梯度求偏导, 求解散度
    void computeLaplacianX(const cv::Mat& img, cv::Mat& laplacianX);
    void computeLaplacianY(const cv::Mat& img, cv::Mat& laplacianY);

private:
    cv::Mat src_;
    cv::Mat src_mask_;
    cv::Point roi_center_;

    cv::Mat result_;
};

}  // namespace ms

#endif  // POISSIONBLENDER_H


