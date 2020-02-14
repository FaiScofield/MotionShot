#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace ms
{

enum InputType {
    VIDEO,      // 视频序列
    LASISESTA,  // LASISESTA数据集
    HUAWEI,     // 华为P30拍摄的数据集
    SEQUENCE,   // 图像序列(以数字序号命名)
    TWO_IMAGES  // 两张图片
};

void ReadImagesFromFolder_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs);
void ReadGroundtruthFromFolder_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs);

void ReadImagesFromVideo(const std::string& video, std::vector<cv::Mat>& imgs);

void ReadImageSequence(const std::string& prefix, const std::string& suffix,
                       std::vector<cv::Mat>& imgs, int startIndex, int num);

void ReadImageSequence_lasisesta(const std::string& folder, std::vector<cv::Mat>& imgs,
                                 std::vector<cv::Mat>& gts, int startIndex = 0, int num = -1);

void ReadImageSequence_huawei(const std::string& folder, std::vector<cv::Mat>& imgs, int startIndex = 0, int num = -1);

void ReadImageSequence_video(const std::string& video, std::vector<cv::Mat>& imgs, int startIndex = 0, int num = -1);

void flowToColor(const cv::Mat& flow, cv::Mat& color);

void drawhistogram(const cv::Mat& src, cv::Mat& hist);

}  // namespace ms

#endif  // UTILITY_HPP
