#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>

namespace ms
{

struct WarpedCorners {
    cv::Point2f tl;
    cv::Point2f tr;
    cv::Point2f bl;
    cv::Point2f br;
};


class ImageStitcher
{
public:
    enum FeatureType { sift, surf, orb, akeke };

    enum MatchType { bf, flann };

    ImageStitcher(FeatureType ft = surf, MatchType mt = flann);

    //    bool setMatcher(const std::string& descriptorMatcherType);

    bool stitch(const std::vector<cv::Mat>& images, cv::Mat& pano);
    bool stitch(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& pano);

    cv::Mat computeHomography(const cv::Mat& img1, const cv::Mat& img2);

    WarpedCorners getWarpedCorners(const cv::Mat& src, const cv::Mat& H);

    void alphaBlend(const cv::Mat& img1, const cv::Mat& img2, const WarpedCorners& corners, cv::Mat& pano);

private:
    cv::Ptr<cv::Feature2D> _featureExtractor;
    cv::Ptr<cv::DescriptorMatcher> _matcher;
    FeatureType _featureType;
    MatchType _matchType;
    cv::Size _ImgSizeInput;
};

}  // namespace ms
#endif  // IMAGE_STITCHER_H
