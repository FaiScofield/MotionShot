#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
//#include <map>

namespace ms
{

#define INFO(msg) (std::cout << "[INFO ] " << msg << std::endl)
#define TIMER(msg) (std::cout << "[TIMER] " << msg << std::endl)
#define WARNING(msg) (std::cerr << "[WARNI] " << msg << std::endl)
#define ERROR(msg) (std::cerr << "[ERROR] " << msg << std::endl)

enum InputType {
    VIDEO       = 0,  // 视频序列
    LASIESTA    = 1,  // LASIESTA数据集
    HUAWEI      = 2,  // 华为P30拍摄的数据集
    SEQUENCE    = 3,  // 图像序列(以数字序号命名)
    TWO_IMAGES  = 4   // 两张图片
};

//enum SmoothMaskType {
//    ERODE   = 0,    // 往里创建过渡区域
//    DILATE  = 1,    // 往外创建过渡区域
//    BOTH    = 2     // 往两个方向都创建过渡区域
//};

void ReadImageNamesFromFolder(const std::string& folder, std::vector<std::string>& names);

void ReadImagesFromFolder_lasiesta(const std::string& folder, std::vector<cv::Mat>& imgs);
void ReadGroundtruthFromFolder_lasiesta(const std::string& folder, std::vector<cv::Mat>& imgs);

void ReadImagesFromVideo(const std::string& video, std::vector<cv::Mat>& imgs);

void ReadImageSequence(const std::string& prefix, const std::string& suffix,
                       std::vector<cv::Mat>& imgs, int startIndex, int num);

void ReadImageSequence_lasiesta(const std::string& folder, std::vector<cv::Mat>& imgs,
                                std::vector<cv::Mat>& gts, int startIndex = 0, int num = -1);

void ReadImageSequence_huawei(const std::string& folder, std::vector<cv::Mat>& imgs,
                              int startIndex = 0, int num = -1);

void ReadImageSequence_video(const std::string& video, std::vector<cv::Mat>& imgs,
                             int startIndex = 0, int num = -1);

// 输入图像的缩放,翻转和旋转. 主要是输入视频可能不正.
void resizeFlipRotateImages(std::vector<cv::Mat>& imgs, double scale, int flip = 0, int rotate = -1);

// 根据前景需要的数量提取出必要的帧进行处理. TODO 需要对输入进行筛选, 去掉模糊/不合适的图像

void extractImagesToStitch(const std::vector<cv::Mat>& vImages, std::vector<cv::Mat>& vImagesToProcess,
                           std::vector<int>& vIdxToProcess, std::vector<std::vector<int>>& vvIdxPerIter,
                           int minFores = 3, int maxFores = 8);

void flowToColor(const cv::Mat& flow, cv::Mat& color);

void showFlow(const cv::Mat& flow, cv::Mat& color);

void drawhistogram(const cv::Mat& src, cv::Mat& hist, const cv::Mat& mask = cv::Mat(), int binSize = 1);

void drawFlowAndHist(const cv::Mat& flow, cv::Mat& flowGray, cv::Mat& hist, cv::Mat& histGraph,
                     int chanel = 1, int binSize = 1);

cv::Rect resultRoi(const std::vector<cv::Point>& corners, const std::vector<cv::Size>& sizes);

cv::Rect resultRoi(const std::vector<cv::Point>& corners, const std::vector<cv::UMat>& images);

void shrinkRoi(const cv::Mat& src, cv::Mat& dst, int size);

void smoothMaskWeightEdge(const cv::Mat& src, cv::Mat& dst, int b1, int b2 = 0);

// 双线性插值
float getPixelValue(const cv::Mat& img, float x, float y);

//导向滤波器
cv::Mat guidedFilter(const cv::Mat& src, int radius, double eps);

}  // namespace ms

#endif  // UTILITY_HPP
