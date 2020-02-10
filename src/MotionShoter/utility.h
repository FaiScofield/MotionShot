#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace ms
{

enum InputType {
    VIDEO,
    DATASET,
    SEQUENCE,
    TWO_IMAGES
};

void ReadImagesFromFolder_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs);
void ReadGroundtruthFromFolder_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs);

void ReadImageSequence_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs, std::vector<cv::Mat>& gts, int startIndex = 0, int num = 0);


void ReadImageSequence(const std::string& prefix, const std::string& suffix, std::vector<cv::Mat>& imgs,
                       int startIndex, int num);

void ReadImagesFromVideo(const std::string& video, std::vector<cv::Mat>& imgs);
void ReadImagesFromVideo(const std::string& video, std::vector<cv::Mat>& imgs, int startIndex, int num);

}  // namespace ms

#endif  // UTILITY_HPP
