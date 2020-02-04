#ifndef IMAGE_STITCHER_H
#define IMAGE_STITCHER_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>

namespace ms
{

class ImageStitcher
{
public:
    enum FeatureType {
        SIFT,
        SURF,
        ORB
    };


    ImageStitcher();
    ImageStitcher(FeatureType ft);



    cv::Mat computeHomography(const cv::Mat& img1, const cv::Mat& img2);
    bool stitch(const std::vector<cv::Mat>& images, cv::Mat& pano);

private:
    void extractFeatures();

    cv::Ptr<cv::Feature2D> _featureExtractor;
    FeatureType _featureType;
};

}  // namespace ms
#endif  // IMAGE_STITCHER_H
