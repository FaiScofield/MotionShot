#ifndef UTILITY_HPP
#define UTILITY_HPP

#include "precompiled.h"
//#include <opencv2/core/core.hpp>
//#include <string>
//#include <vector>
//#include <map>

namespace ms
{

enum InputType {
    VIDEO = 0,      // 视频序列
    LASIESTA = 1,   // LASIESTA数据集
    HUAWEI = 2,     // 华为P30拍摄的数据集
    SEQUENCE = 3,   // 图像序列(以数字序号命名)
    TWO_IMAGES = 4  // 两张图片
};


void ReadImageNamesFromFolder(const string& folder, vector<string>& names);
void ReadImagesFromFolder_lasiesta(const string& folder, vector<Mat>& imgs);
void ReadGroundtruthFromFolder_lasiesta(const string& folder, vector<Mat>& imgs);
void ReadImagesFromVideo(const string& video, vector<Mat>& imgs);
void ReadImageSequence(const string& prefix, const string& suffix, vector<Mat>& imgs, int beginIdx, int num);
void ReadImageSequence_lasiesta(const string& folder, vector<Mat>& imgs, vector<Mat>& gts,
                                int beginIdx = 0, int num = -1);
void ReadImageSequence_huawei(const string& folder, vector<Mat>& imgs, int beginIdx = 0, int num = -1);
void ReadImageSequence_video(const string& video, vector<Mat>& imgs, int beginIdx = 0, int num = -1);
void ReadGroundtruthRectFromFolder(const string& folder, const string& suffix, vector<Mat>& masks,
                                   vector<Rect>& rects, int beginIdx, int num);

void ColorMask2Gray(const vector<Mat>& colors, vector<Mat>& grays);

// 输入图像的缩放,翻转和旋转. 主要是输入视频可能不正.
void ResizeFlipRotateImages(vector<Mat>& imgs, double scale, int flip = 0, int rotate = -1);

// 根据前景需要的数量提取出必要的帧进行处理. TODO 需要对输入进行筛选, 去掉模糊/不合适的图像
void ExtractImagesToStitch(const vector<Mat>& vImages, vector<Mat>& vImagesToProcess, vector<int>& vIdxToProcess,
                           vector<vector<int>>& vvIdxPerIter, int minFores = 3, int maxFores = 8);


Rect ResultRoi(const vector<Point>& corners, const vector<Size>& sizes);
Rect ResultRoi(const vector<Point>& corners, const vector<UMat>& images);

void SmoothMaskWeightEdge(const Mat& src, Mat& dst, int b1, int b2 = 0);

// 双线性插值
float GetPixelValue(const Mat& img, float x, float y);

// 导向滤波器
Mat GuidedFilter(const Mat& src, int radius, double eps);

// 边缘滤波
// void applyEdgeFilter(Mat& img, int x, int y, int dir);
void OverlappedEdgesSmoothing(const Mat& src, const Mat& mask, Mat& dst, double scale = 0.5);

Rect ResizeRectangle(const Rect& rec, const Size& size, int a, int b);

#ifdef DEBUG
void NamedLargeWindow(const string& title, bool flag = true);

void FlowToColor(const Mat& flow, Mat& color);
void ShowFlow(const Mat& flow, Mat& color);

void Drawhistogram(const Mat& src, Mat& hist, const Mat& mask = Mat(), int binSize = 1);
void DrawFlowAndHist(const Mat& flow, Mat& flowGray, Mat& hist, Mat& histGraph, int chanel = 1, int binSize = 1);
#endif

}  // namespace ms

#endif  // UTILITY_HPP
