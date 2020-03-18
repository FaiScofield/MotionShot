/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

//#include "precomp.hpp"
//#include "opencl_kernels_stitching.hpp"

#include "cvBlenders.h"
#include "utility.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

#define ENABLE_DEBUG_RESULT 1
#if ENABLE_DEBUG_RESULT
#include <opencv2/highgui.hpp>
#endif

namespace ms
{

using namespace std;
using namespace cv;

static const float WEIGHT_EPS = 1e-5f;  // 加上微小量防止分母为0


Ptr<cvBlender> cvBlender::createDefault(int type, bool try_gpu)
{
    if (type == FEATHER)
        return makePtr<cvFeatherBlender>(try_gpu);
    if (type == MULTI_BAND)
        return makePtr<cvMultiBandBlender>(try_gpu);
    return makePtr<cvBlender>();
}


void cvBlender::prepare(const std::vector<Point>& corners, const std::vector<Size>& sizes)
{
    prepare(resultRoi(corners, sizes));
}


void cvBlender::prepare(Rect dst_roi)
{
    dst_.create(dst_roi.size(), CV_16SC3);
    dst_.setTo(Scalar::all(0));
    dst_mask_.create(dst_roi.size(), CV_8U);
    dst_mask_.setTo(Scalar::all(0));
    dst_roi_ = dst_roi;

    overlapped_edges_mask_width_ = 5;
    overlapped_edges_.create(dst_roi.size(), CV_8U);
    overlapped_edges_.setTo(Scalar::all(0));
    overlapped_edges_mask_.create(dst_roi.size(), CV_8U);
    overlapped_edges_mask_.setTo(Scalar::all(0));
}


void cvBlender::feed(InputArray _img, InputArray _mask, Point tl)
{
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);
    Mat dst_mask = dst_mask_.getMat(ACCESS_RW);

    assert(img.type() == CV_16SC3);
    assert(mask.type() == CV_8U);
    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

    for (int y = 0; y < img.rows; ++y) {
        const Point3_<short>* src_row = img.ptr<Point3_<short>>(y);
        Point3_<short>* dst_row = dst.ptr<Point3_<short>>(dy + y);
        const uchar* mask_row = mask.ptr<uchar>(y);
        uchar* dst_mask_row = dst_mask.ptr<uchar>(dy + y);

        for (int x = 0; x < img.cols; ++x) {
            if (mask_row[x])
                dst_row[dx + x] = src_row[x];
            dst_mask_row[dx + x] |= mask_row[x];
        }
    }
}


void cvBlender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    UMat mask;
    compare(dst_mask_, 0, mask, CMP_EQ);
    dst_.setTo(Scalar::all(0), mask);
    dst.assign(dst_);
    dst_mask.assign(dst_mask_);
    dst_.release();
    dst_mask_.release();
}

Mat cvBlender::getOverlappedEdgesMask(int size) const
{
    Mat res;

    if (size > 1) {
        const Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
        dilate(overlapped_edges_mask_, res, kernel);  // 膨胀
    } else {
        res = overlapped_edges_mask_.clone();
    }
    return res;
}


cvFeatherBlender::cvFeatherBlender(float sharpness, bool cover) : enable_cover_(cover)
{
    if (cover)
        sharpness_ = 1.0;
    else
        sharpness_ = sharpness;
}

void cvFeatherBlender::prepare(Rect dst_roi)
{
    cvBlender::prepare(dst_roi);
    dst_weight_map_.create(dst_roi.size(), CV_32F);
    dst_weight_map_.setTo(0);
}


void cvFeatherBlender::feed(InputArray _img, InputArray mask, Point tl)
{
#define ENABLE_DEBUG_RESULT_FEATHER_FEED1 0


    Mat img = _img.getMat();
    Mat dst = dst_.getMat(ACCESS_RW);
    const bool largeWin = (img.cols > 1600 || img.rows > 1200);

    assert(img.type() == CV_16SC3);
    assert(mask.type() == CV_8UC1);

    // 1.为输入的掩模分析权重
    if (enable_cover_) {  // 覆盖模式下输入的掩模已经自带权重
        mask.getUMat().convertTo(weight_map_, CV_32F, 1. / 255.);
    } else {  // 普通情况下权重从中心区域到边缘递减(@ref distanceTransform())
        createWeightMap(mask, sharpness_, weight_map_);
    }

    Mat weight_map = weight_map_.getMat(ACCESS_READ);        // 32F
    Mat dst_weight_map = dst_weight_map_.getMat(ACCESS_RW);  // 32F

    // 2.计算重叠区域的边缘掩模
    if (enable_cover_) {
        Mat srcMask, dstMask;
        srcMask = mask.getMat();
        dst_weight_map.convertTo(dstMask, CV_8UC1, 255);
        threshold(srcMask, srcMask, 254, 255, THRESH_BINARY); // 获取100%是前景的区域

        Mat overlappedArea, ignoreArea, edge;
        bitwise_and(dstMask, 255, overlappedArea, srcMask);         // 得到重叠区域
        morphologyEx(overlappedArea, edge, MORPH_GRADIENT, Mat());  // 得到重叠区域边缘

        //! 被覆盖的边缘需要消除
        morphologyEx(srcMask, ignoreArea, MORPH_ERODE, Mat());
        bitwise_and(edge, 0, edge, ignoreArea);
        bitwise_and(overlapped_edges_, 0, overlapped_edges_, srcMask);
        overlapped_edges_ += edge;

        //! 获得前景边缘往内的一部分掩模, 这里往往会有一些白边
        const int s = 2 * overlapped_edges_mask_width_ + 1;
        const Mat kernel = getStructuringElement(MORPH_RECT, Size(s, s));
        Mat gapMask, overEroded, maksEroded;
        erode(overlappedArea, overEroded, kernel);
        erode(srcMask, maksEroded, kernel);
        gapMask = overlappedArea - overEroded;
        bitwise_and(gapMask, 0, gapMask, maksEroded);
        overlapped_edges_mask_ += gapMask;

        Mat imgU, gap, weightMap_U;
        img.convertTo(imgU, CV_8UC3);
        weight_map.convertTo(weightMap_U, CV_8UC1);
        imwrite("/home/vance/output/ms/去除gap前.png", weightMap_U);
        smoothEdgesOnOverlappedArea(imgU, gapMask, gap, 0.5); //! TODO
        bitwise_and(weightMap_U, 0, weightMap_U, gap);
        imwrite("/home/vance/output/ms/去除gap后.png", weightMap_U);

//        namedLargeWindow("重叠区域边缘(考虑覆盖)", largeWin);
//        imshow("重叠区域边缘(考虑覆盖)", overlapped_edges_);
//        namedLargeWindow("重叠区域边缘掩模(考虑覆盖)", largeWin);
//        imshow("重叠区域边缘掩模(考虑覆盖)", overlapped_edges_mask_);
//        waitKey(0);
    }

    int dx = tl.x - dst_roi_.x;
    int dy = tl.y - dst_roi_.y;

#if ENABLE_DEBUG_RESULT_FEATHER_FEED1 && ENABLE_DEBUG_RESULT
    Mat disU;
    dst.convertTo(disU, CV_8U);
//    imshow("dst before feed", disU);
//    imshow("dst_weight_map_ before feed", dst_weight_map_);
//    Mat foreInput, imgU, maskU;
//    img.convertTo(imgU, CV_8UC3);
//    maskU = mask.getMat();
//    bitwise_and(imgU, 255, foreInput, maskU);
//    imshow("mask input", maskU);
//    imshow("foreground input", foreInput);
#endif

    // 3.当前图像的像素乘上对应权重加到输出图像上, 并记录每个pixel对应的总权重和
    for (int y = 0; y < img.rows; ++y) {
        const Point3_<short>* src_row = img.ptr<Point3_<short>>(y);
        Point3_<short>* dst_row = dst.ptr<Point3_<short>>(dy + y);
        const float* weight_row = weight_map.ptr<float>(y);
        float* dst_weight_row = dst_weight_map.ptr<float>(dy + y);
        assert(src_row != nullptr && dst_row != nullptr && weight_row != nullptr && dst_weight_row != nullptr);

        for (int x = 0; x < img.cols; ++x) {
            if (enable_cover_) {
                if (weight_row[x] > 0.9999f) {  // 权值为1, 直接覆盖
                    dst_row[dx + x].x = static_cast<short>(src_row[x].x * weight_row[x]);
                    dst_row[dx + x].y = static_cast<short>(src_row[x].y * weight_row[x]);
                    dst_row[dx + x].z = static_cast<short>(src_row[x].z * weight_row[x]);
                    dst_weight_row[dx + x] = weight_row[x];
                } else if (dst_weight_row[dx + x] == 1.f) {
                    // 待添加区域已经有精确前景则不添加
                } else {
                    dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                    dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                    dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                    dst_weight_row[dx + x] += weight_row[x];
                }
            } else {
                dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                dst_weight_row[dx + x] += weight_row[x];
            }
        }
    }


#if ENABLE_DEBUG_RESULT_FEATHER_FEED1 && ENABLE_DEBUG_RESULT
//    imshow("tatal edge", overlapped_edges_mask_);

//    vector<vector<Point>> contours;
//    findContours(overlapped, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    dst.convertTo(disU, CV_8U);
//    drawContours(disU, contours, -1, Scalar(0, 255, 0), 2);
    namedLargeWindow("dst after feed & overlapped area(green)", largeWin);
    namedLargeWindow("dst_weight_map_ after feed", largeWin);
    imshow("dst after feed & overlapped area(green)", disU);
    imshow("dst_weight_map_ after feed", dst_weight_map_);
    waitKey(0);
#endif
}

void cvFeatherBlender::feed(const vector<Mat>& vImgs, const vector<Mat>& vMasks, const vector<Point>& vTopleftCorners)
{
#define ENABLE_DEBUG_RESULT_FEATHER_FEED2 0

    //! TODO . 开启强覆盖功能.
    //! 在重叠区域设置调整强覆盖区域的权重, 使在时序上后面的高权前景部分覆盖前面的前景.

    assert(vImgs.size() == vMasks.size());

    Mat dst = dst_.getMat(ACCESS_RW);
    Mat dst_weight_map = dst_weight_map_.getMat(ACCESS_RW);

    const size_t N = vImgs.size();
    for (size_t i = 0; i < N; ++i) {
        assert(vImgs[i].type() == CV_16SC3);
        assert(vMasks[i].type() == CV_8U);

        const Mat& image = vImgs[i];
        const Mat& mask = vMasks[i];
        Mat weightMap;
        if (enable_cover_)
            mask.convertTo(weightMap, CV_8UC1);
        else
            createWeightMap(mask, sharpness_, weightMap);

        const int dx = vTopleftCorners[i].x - dst_roi_.x;
        const int dy = vTopleftCorners[i].y - dst_roi_.y;

#if ENABLE_DEBUG_RESULT_FEATHER_FEED2 && ENABLE_DEBUG_RESULT
        Mat disU;
        dst.convertTo(disU, CV_8U);
        imshow("dst before feed", disU);
        imshow("dst_weight_map_ before feed", dst_weight_map_);
#endif

        // 当前图像的像素乘上对应权重加到输出图像上, 并记录每个pixel对应的总权重和
        for (int y = 0; y < image.rows; ++y) {
            const Point3_<short>* src_row = image.ptr<Point3_<short>>(y);
            const float* weight_row = weightMap.ptr<float>(y);
            Point3_<short>* dst_row = dst.ptr<Point3_<short>>(dy + y);
            float* dst_weight_row = dst_weight_map.ptr<float>(dy + y);
            assert(src_row != nullptr && dst_row != nullptr && weight_row != nullptr && dst_weight_row != nullptr);

            for (int x = 0; x < image.cols; ++x) {
                if (enable_cover_) {                         // 时序后覆盖前
                    if (abs(1.f - weight_row[x]) < 1e-6f) {  // 权值为1, 直接覆盖
                        dst_row[dx + x].x = static_cast<short>(src_row[x].x * weight_row[x]);
                        dst_row[dx + x].y = static_cast<short>(src_row[x].y * weight_row[x]);
                        dst_row[dx + x].z = static_cast<short>(src_row[x].z * weight_row[x]);
                        dst_weight_row[dx + x] = weight_row[x];
                    } else if (abs(1.f - dst_weight_row[dx + x]) < 1e-6f) {  // 待添加区域已经有前景则将降低此区域权值强行设为0.1
                        dst_row[dx + x].x += static_cast<short>(src_row[x].x * 0.01);
                        dst_row[dx + x].y += static_cast<short>(src_row[x].y * 0.01);
                        dst_row[dx + x].z += static_cast<short>(src_row[x].z * 0.01);
                        dst_weight_row[dx + x] += 0.01;
                    }
                } else {
                    dst_row[dx + x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                    dst_row[dx + x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                    dst_row[dx + x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                    dst_weight_row[dx + x] += weight_row[x];
                }
            }
        }

#if ENABLE_DEBUG_RESULT_FEATHER_FEED2 && ENABLE_DEBUG_RESULT
        dst.convertTo(disU, CV_8U);
        imshow("dst after feed", disU);
        imshow("dst_weight_map_ after feed", dst_weight_map_);
        waitKey(0);
#endif
    }
}

void cvFeatherBlender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
#define ENABLE_DEBUG_RESULT_FEATHER_BLEND 0

#if ENABLE_DEBUG_RESULT_FEATHER_BLEND && ENABLE_DEBUG_RESULT
    Mat tmp;
    dst_.convertTo(tmp, CV_8U, 255);
    imshow("before blending dst_weight_map_", dst_weight_map_);
    imshow("before blending dst_", tmp);
#endif

    normalizeUsingWeightMap(dst_weight_map_, dst_);

#if ENABLE_DEBUG_RESULT_FEATHER_BLEND && ENABLE_DEBUG_RESULT
    dst_.convertTo(tmp, CV_8U, 255);
    imshow("after blending dst_weight_map_", dst_weight_map_);
    imshow("after blending dst_", tmp);
    waitKey(0);
#endif

    if (enable_cover_)
        dst_weight_map_.convertTo(dst_mask_, CV_8U, 255);
    else
        compare(dst_weight_map_, WEIGHT_EPS, dst_mask_, CMP_GT);
    cvBlender::blend(dst, dst_mask);
}

/**
 * @brief overlappedEdgesSmoothing  重叠区域的边缘平滑, 主要是为了消除前景重叠区域边缘处的锯齿
 * @param src   拼接后的前景图, 16SC3
 * @param mask  前景图中重叠区域的边缘掩模(降低计算量, 限制处理范围), 8UC1
 * @param dst   结果图
 * @param scale 下采样的尺度比例
 */
void cvFeatherBlender::smoothEdgesOnOverlappedArea(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, float scale)
{
#define ENABLE_DEBUG_RESULT_FEATHER_SMOOTH 1

    assert(scale > 0 && scale <= 1.);
    const double invScale = 1. / scale;

    // 初始化梯度计算的核和梯度方向颜色
    static vector<Mat> vKernels;
    if (vKernels.empty()) {
        vKernels.resize(4);
        vKernels[0] = (Mat_<char>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
        vKernels[1] = vKernels[0].t();
        vKernels[2] = (Mat_<char>(3, 3) << -10, -3, 0, -3, 0, 3, 0, 3, 10);
        vKernels[3] = (Mat_<char>(3, 3) << 0, 3, 10, -3, 0, 3, -10, -3, 0);
    }
    static vector<Point3_<uchar>> vLableColor;
    if (vLableColor.empty()) {
        vLableColor.resize(5);
        vLableColor[0] = Point3_<uchar>(0, 0, 0);
        vLableColor[1] = Point3_<uchar>(0, 0, 255);
        vLableColor[2] = Point3_<uchar>(0, 255, 0);
        vLableColor[3] = Point3_<uchar>(255, 0, 0);
        vLableColor[4] = Point3_<uchar>(255, 255, 255);
    }

    // 1.下采样并在Y域上求解4个方向的梯度
    Mat srcGray, srcScaled, srcGrayScaled, maskScaled;
    resize(src, srcScaled, Size(), scale, scale);
    resize(mask, maskScaled, Size(), scale, scale);
    cvtColor(src, srcGray, COLOR_BGR2GRAY);
    resize(srcGray, srcGrayScaled, Size(), scale, scale);
    vector<Mat> vEdges_F(4);
    for (int i = 0; i < 4; ++i)
        filter2D(srcGrayScaled, vEdges_F[i], CV_32FC1, vKernels[i], Point(-1, -1), 0, BORDER_REPLICATE);

//#if ENABLE_DEBUG_RESULT && ENABLE_DEBUG_RESULT_FEATHER_SMOOTH
//    Mat g12 = (cv::abs(vEdges_F[0]) + cv::abs(vEdges_F[1])) * 0.5;
//    Mat g34 = (cv::abs(vEdges_F[2]) + cv::abs(vEdges_F[3])) * 0.5;
//    normalize(g12, g12, 0.f, 1.f, NORM_MINMAX);
//    g12.convertTo(g12, CV_8UC1, 255);
//    normalize(g34, g34, 0.f, 1.f, NORM_MINMAX);
//    g34.convertTo(g34, CV_8UC1, 255);
//    bitwise_or(0, g12, g12, maskScaled);
//    bitwise_or(0, g34, g34, maskScaled);
//    imwrite("/home/vance/output/ms/初始梯度方向1+2.png", g12);
//    imwrite("/home/vance/output/ms/初始梯度方向3+4.png", g34);
//#endif

    // 2.对掩模对应的区域的梯度, 选出4个方向中的最大者并标记最大梯度方向
    Mat maxEdge_Full = Mat::zeros(srcGrayScaled.size(), CV_32FC1);
    Mat maxEdge_F = Mat::zeros(srcGrayScaled.size(), CV_32FC1);
    Mat dstLabel = Mat::zeros(srcGrayScaled.size(), CV_8UC1);   // 掩模区内边缘的方向
    Mat distLabelColor = Mat::zeros(srcGrayScaled.size(), CV_8UC3);
    for (int y = 0; y < srcGrayScaled.rows; ++y) {
        const char* mask_row = maskScaled.ptr<char>(y);
        float* dst_row = maxEdge_F.ptr<float>(y);
        char* label_row = dstLabel.ptr<char>(y);
        Point3_<uchar>* distLabelColor_row = distLabelColor.ptr<Point3_<uchar>>(y);

        float* tmp_row = maxEdge_Full.ptr<float>(y);

        for (int x = 0; x < srcGrayScaled.cols; ++x) {
//            if (mask_row[x] != 0) {
                int label = 0;  // 默认无梯度则黑色
                float maxGradient = 0.f;
                for (int i = 0; i < 4; ++i) {
                    const float g = abs(vEdges_F[i].at<float>(y, x)); // 绝对值符号使正负方向为同一方向
                    if (g > maxGradient) {
                        maxGradient = g;
                        label = i + 1;
                    }
                }
                tmp_row[x] = maxGradient;
                if (mask_row[x] != 0) {
                dst_row[x] = maxGradient;
                label_row[x] = label;
                distLabelColor_row[x] = vLableColor[label];
            }
        }
    }
    Mat finalEdge, gap;
    maxEdge_F.convertTo(finalEdge, CV_8UC1);
    normalize(finalEdge, finalEdge, 0, 255, NORM_MINMAX);  // 归一化
    threshold(finalEdge, finalEdge, 10, 0, THRESH_TOZERO);  // 太小的梯度归0
    threshold(finalEdge, gap, 150, 255, THRESH_OTSU);   // 白边区域

#if ENABLE_DEBUG_RESULT && ENABLE_DEBUG_RESULT_FEATHER_SMOOTH
    imwrite("/home/vance/output/ms/重叠区域最大梯度otsu.png", gap);

    Mat tmp1;
    imwrite("/home/vance/output/ms/重叠区域最大梯度强度.png", finalEdge);
    imwrite("/home/vance/output/ms/重叠区域最大梯度方向.png", distLabelColor);

    normalize(maxEdge_Full, maxEdge_Full, 0.f, 1.f, NORM_MINMAX);
    maxEdge_Full.convertTo(tmp1, CV_8UC1, 255);
    imwrite("/home/vance/output/ms/初始梯度最大值.png", tmp1);
    Mat toShow = src.clone();
#endif

    //! 方法1, 直接去掉gap
    dst = gap;
/*
    // 3.应用边缘滤波
    vector<Mat> srcChannels(3), dstChanels(3);
    split(src, srcChannels);
    dstChanels = srcChannels;

    vector<Point> vPointsToSmooth;
    vector<int> vDirsToSmooth;
    vPointsToSmooth.reserve(10000);
    vDirsToSmooth.reserve(10000);


    // 先在低尺度上滤波一次, 然后上采样回原尺度加权融合
    vector<Mat> vDstScaledFilterd(3), vDstScaled(3);
    resize(dstChanels[0], vDstScaled[0], Size(), scale, scale);
    resize(dstChanels[1], vDstScaled[1], Size(), scale, scale);
    resize(dstChanels[2], vDstScaled[2], Size(), scale, scale);
    for (int i = 0; i < 3; ++i)
        applyEdgeFilter(vDstScaled[i], finalEdge, dstLabel, vDstScaledFilterd[i]);

    Mat dstScaledFilterd, dstFilterdOnce;
    merge(vDstScaledFilterd, dstScaledFilterd);
    resize(dstScaledFilterd, dstFilterdOnce, src.size());

    imwrite("/home/vance/output/ms/前景(scaled).png", srcScaled);
    imwrite("/home/vance/output/ms/前景(scaled)低尺度滤波.png", dstScaledFilterd);
    imwrite("/home/vance/output/ms/前景低尺度滤波.png", dstFilterdOnce);

    dst = src.clone();
    dst.setTo(0, mask);

    bitwise_or(dst, dstFilterdOnce, dst, mask);
    imwrite("/home/vance/output/ms/前景低尺度滤波应用到原尺度.png", dst);
    split(dst, dstChanels);

    // 再到原尺度上滤波
    for (int y = 0; y < src.rows; ++y) {
        const int y_scaled = cvRound(y * scale);
        if (y_scaled >= dstLabel.rows)
            continue;

        const uchar* edge_row = finalEdge.ptr<uchar>(y_scaled);
        const uchar* label_row = dstLabel.ptr<uchar>(y_scaled);
        for (int x = 0; x < src.cols; ++x) {
            const int x_scaled = cvRound(x * scale);
            if (x_scaled >= dstLabel.cols)
                continue;

            if (label_row[x_scaled] > 0 && edge_row[x_scaled] > 10) {
                applyEdgeFilter(dstChanels[0], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[1], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[2], x, y, label_row[x_scaled]);
//                vPointsToSmooth.push_back(Point(x, y));
//                vDirsToSmooth.push_back(label_row[x_scaled]);
                toShow.at<Vec3b>(y, x) = Vec3b(0,255,0);
            }
        }
    }

//    applyEdgeFilter(srcChannels[0], dstChanels[0], vPointsToSmooth, vDirsToSmooth);
//    applyEdgeFilter(srcChannels[1], dstChanels[1], vPointsToSmooth, vDirsToSmooth);
//    applyEdgeFilter(srcChannels[2], dstChanels[2], vPointsToSmooth, vDirsToSmooth);
    merge(dstChanels, dst);

#if ENABLE_DEBUG_RESULT && ENABLE_DEBUG_RESULT_FEATHER_SMOOTH
    Mat input = src.clone(), output = dst.clone();
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    drawContours(input, contours, -1, Scalar(0, 255, 0), 1);
    drawContours(output, contours, -1, Scalar(0, 255, 0), 1);
    //    namedLargeWindow("points to smooth", 1);
    namedLargeWindow("src", 1);
    namedLargeWindow("dst", 1);
    //    imshow("src", input);
    //    imshow("dst", dst);
    imwrite("/home/vance/output/ms/输入前景和边缘掩模.png", input);
    imwrite("/home/vance/output/ms/输出图像和边缘掩模.png", output);
    imwrite("/home/vance/output/ms/被处理的区域.png", toShow);
    //    imshow("points to smooth", toShow);
    waitKey(10);
    destroyAllWindows();
#endif
*/
}


Rect cvFeatherBlender::createWeightMaps(const std::vector<UMat>& masks,
                                        const std::vector<Point>& corners, std::vector<UMat>& weight_maps)
{
    weight_maps.resize(masks.size());
    for (size_t i = 0; i < masks.size(); ++i)
        createWeightMap(masks[i], sharpness_, weight_maps[i]);

    Rect dst_roi = resultRoi(corners, masks);
    Mat weights_sum(dst_roi.size(), CV_32F);
    weights_sum.setTo(0);

    for (size_t i = 0; i < weight_maps.size(); ++i) {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y, weight_maps[i].cols,
                 weight_maps[i].rows);
        add(weights_sum(roi), weight_maps[i], weights_sum(roi));
    }

    for (size_t i = 0; i < weight_maps.size(); ++i) {
        Rect roi(corners[i].x - dst_roi.x, corners[i].y - dst_roi.y, weight_maps[i].cols,
                 weight_maps[i].rows);
        Mat tmp = weights_sum(roi);
        tmp.setTo(1, tmp < std::numeric_limits<float>::epsilon());
        divide(weight_maps[i], tmp, weight_maps[i]);
    }

    return dst_roi;
}


cvMultiBandBlender::cvMultiBandBlender(int try_gpu, int num_bands, int weight_type)
{
    num_bands_ = 0;
    setNumBands(num_bands);

    CV_UNUSED(try_gpu);
    can_use_gpu_ = false;

    assert(weight_type == CV_32F || weight_type == CV_16S);
    weight_type_ = weight_type;
}

// 初始化金字塔
void cvMultiBandBlender::prepare(Rect dst_roi)
{
    dst_roi_final_ = dst_roi;

    // Crop unnecessary bands
    double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
    num_bands_ = std::min(actual_num_bands_, static_cast<int>(ceil(std::log(max_len) / std::log(2.0))));

    // Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
    // 尺寸凑到层数的偶数倍, 以便金字塔缩放. 但最终的输出图像尺寸是由初始输入参数决定的.
    dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
    dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

    cvBlender::prepare(dst_roi);
    {
        dst_pyr_laplace_.resize(num_bands_ + 1);
        dst_pyr_laplace_[0] = dst_;

        dst_band_weights_.resize(num_bands_ + 1);
        dst_band_weights_[0].create(dst_roi.size(), weight_type_);
        dst_band_weights_[0].setTo(0);

        for (int i = 1; i <= num_bands_; ++i) {
            dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
                                       (dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_16SC3);
            dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
                                        (dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
            dst_pyr_laplace_[i].setTo(Scalar::all(0));
            dst_band_weights_[i].setTo(0);
        }
    }
}


void cvMultiBandBlender::feed(InputArray _img, InputArray mask, Point tl)
{
    UMat img = _img.getUMat();
    assert(img.type() == CV_16SC3 || img.type() == CV_8UC3);
    assert(mask.type() == CV_8U);

    // Keep source image in memory with small border
    // 给输入图像添加一个gap大小的边界, 但不超过dis_roi的范围.
    int gap = 3 * (1 << num_bands_);
    Point tl_new(std::max(dst_roi_.x, tl.x - gap), std::max(dst_roi_.y, tl.y - gap));
    Point br_new(std::min(dst_roi_.br().x, tl.x + img.cols + gap),
                 std::min(dst_roi_.br().y, tl.y + img.rows + gap));

    // Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
    // After that scale between layers is exactly 2.
    //
    // We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
    // image is bordered to have size equal to the final image size, but this is too memory hungry approach.
    //
    // 根据添加gap后的大小,微调尺寸以便被(1 << num_bands_)整除,这样金字塔之间的比例才能刚好为2.
    tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
    tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
    int width = br_new.x - tl_new.x;
    int height = br_new.y - tl_new.y;
    width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
    height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
    br_new.x = tl_new.x + width;
    br_new.y = tl_new.y + height;
    int dy = std::max(br_new.y - dst_roi_.br().y, 0);
    int dx = std::max(br_new.x - dst_roi_.br().x, 0);
    tl_new.x -= dx;
    br_new.x -= dx;
    tl_new.y -= dy;
    br_new.y -= dy;

    int top = tl.y - tl_new.y;
    int left = tl.x - tl_new.x;
    int bottom = br_new.y - tl.y - img.rows;
    int right = br_new.x - tl.x - img.cols;

    // Create the source image Laplacian pyramid
    UMat img_with_border;
    copyMakeBorder(_img, img_with_border, top, bottom, left, right, BORDER_REFLECT);  // 生成添加了边界的图像

    std::vector<UMat> src_pyr_laplace;
    createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);

    // Create the weight map Gaussian pyramid. 掩模也要添加相同的边界,并生成金字塔掩模
    UMat weight_map;
    std::vector<UMat> weight_pyr_gauss(num_bands_ + 1);

    if (weight_type_ == CV_32F) {
        mask.getUMat().convertTo(weight_map, CV_32F, 1. / 255.);
    } else {  // weight_type_ == CV_16S
        mask.getUMat().convertTo(weight_map, CV_16S);
        UMat add_mask;
        compare(mask, 0, add_mask, CMP_NE);
        add(weight_map, Scalar::all(1), weight_map, add_mask);
    }

    copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, BORDER_CONSTANT);

    for (int i = 0; i < num_bands_; ++i)
        pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

    // 扩边后的img在最终图像上对应的区域
    int y_tl = tl_new.y - dst_roi_.y;
    int y_br = br_new.y - dst_roi_.y;
    int x_tl = tl_new.x - dst_roi_.x;
    int x_br = br_new.x - dst_roi_.x;

    // Add weighted layer of the source image to the final Laplacian pyramid layer
    for (int i = 0; i <= num_bands_; ++i) {
        Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);

        Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(ACCESS_READ);
        Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc).getMat(ACCESS_RW);
        Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(ACCESS_READ);
        Mat _dst_band_weights = dst_band_weights_[i](rc).getMat(ACCESS_RW);
        if (weight_type_ == CV_32F) {
            for (int y = 0; y < rc.height; ++y) {
                const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short>>(y);
                Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short>>(y);
                const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
                float* dst_weight_row = _dst_band_weights.ptr<float>(y);

                for (int x = 0; x < rc.width; ++x) {
                    dst_row[x].x += static_cast<short>(src_row[x].x * weight_row[x]);
                    dst_row[x].y += static_cast<short>(src_row[x].y * weight_row[x]);
                    dst_row[x].z += static_cast<short>(src_row[x].z * weight_row[x]);
                    dst_weight_row[x] += weight_row[x];
                }
            }
        } else {  // weight_type_ == CV_16S
            for (int y = 0; y < y_br - y_tl; ++y) {
                const Point3_<short>* src_row = _src_pyr_laplace.ptr<Point3_<short>>(y);
                Point3_<short>* dst_row = _dst_pyr_laplace.ptr<Point3_<short>>(y);
                const short* weight_row = _weight_pyr_gauss.ptr<short>(y);
                short* dst_weight_row = _dst_band_weights.ptr<short>(y);

                for (int x = 0; x < x_br - x_tl; ++x) {
                    dst_row[x].x += short((src_row[x].x * weight_row[x]) >> 8);
                    dst_row[x].y += short((src_row[x].y * weight_row[x]) >> 8);
                    dst_row[x].z += short((src_row[x].z * weight_row[x]) >> 8);
                    dst_weight_row[x] += weight_row[x];
                }
            }
        }

        // 更新一下到下一层的对应区域
        x_tl /= 2;
        y_tl /= 2;
        x_br /= 2;
        y_br /= 2;
    }
}


void cvMultiBandBlender::blend(InputOutputArray dst, InputOutputArray dst_mask)
{
    Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height);
    {
        cv::UMat dst_band_weights_0;

        for (int i = 0; i <= num_bands_; ++i)
            normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);

        restoreImageFromLaplacePyr(dst_pyr_laplace_);

        dst_ = dst_pyr_laplace_[0](dst_rc);
        dst_band_weights_0 = dst_band_weights_[0];

        dst_pyr_laplace_.clear();
        dst_band_weights_.clear();

        compare(dst_band_weights_0(dst_rc), WEIGHT_EPS, dst_mask_, CMP_GT);

        cvBlender::blend(dst, dst_mask);
    }
}


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

/**
 * @brief normalizeUsingWeightMap   输出图像归一化, 除去总权重
 * @param _weight   输出图像mask中每个像素的总权值, 即 dst_weight_map_
 * @param _src      输出图像, src += weight * input
 */
void normalizeUsingWeightMap(InputArray _weight, InputOutputArray _src)
{
    Mat src = _src.getMat();
    Mat weight = _weight.getMat();
    assert(src.type() == CV_16SC3);

    if (weight.type() == CV_32FC1) {
        for (int y = 0; y < src.rows; ++y) {
            Point3_<short>* row = src.ptr<Point3_<short>>(y);
            const float* weight_row = weight.ptr<float>(y);

            for (int x = 0; x < src.cols; ++x) {
                row[x].x = static_cast<short>(row[x].x / (weight_row[x] + WEIGHT_EPS));
                row[x].y = static_cast<short>(row[x].y / (weight_row[x] + WEIGHT_EPS));
                row[x].z = static_cast<short>(row[x].z / (weight_row[x] + WEIGHT_EPS));
            }
        }
    } else {
        assert(weight.type() == CV_16SC1);

        for (int y = 0; y < src.rows; ++y) {
            const short* weight_row = weight.ptr<short>(y);
            Point3_<short>* row = src.ptr<Point3_<short>>(y);

            for (int x = 0; x < src.cols; ++x) {
                int w = weight_row[x] + 1;
                row[x].x = static_cast<short>((row[x].x << 8) / w);
                row[x].y = static_cast<short>((row[x].y << 8) / w);
                row[x].z = static_cast<short>((row[x].z << 8) / w);
            }
        }
    }
}


void createWeightMap(InputArray mask, float sharpness, InputOutputArray weight)
{
    assert(mask.type() == CV_8U);

    distanceTransform(mask, weight, DIST_L1, 3);  // CV_32F

    //#if ENABLE_DEBUG_RESULT
    //    Mat w, wu;

    //    //cout << "weight type:" << weight.getMat().type() << endl;
    //    weight.getMat().convertTo(wu, CV_8UC1, 255);
    //    hconcat(mask, wu, w);
    //    imshow("mask & weight", w);
    //#endif

    UMat tmp;
    multiply(weight, sharpness, tmp);
    threshold(tmp, weight, 1.f, 1.f, THRESH_TRUNC);

    //#if ENABLE_DEBUG_RESULT
    //    //cout << "tmp type:" << tmp.type() << endl;
    //    //cout << "weight type:" << weight.type() << endl;

    //    Mat tmp1, tmp2;

    //    normalize(weight, tmp1, 0, 255, NORM_MINMAX);
    //    Mat w2;
    //    hconcat(tmp, tmp1, w2);
    //    imshow("weight * sharpness & threshold weight", w2) ;
    //    waitKey(0);
    //#endif
}


void createLaplacePyr(InputArray img, int num_levels, std::vector<UMat>& pyr)
{
    pyr.resize(num_levels + 1);

    if (img.depth() == CV_8U) {
        if (num_levels == 0) {
            img.getUMat().convertTo(pyr[0], CV_16S);
            return;
        }

        UMat downNext;
        UMat current = img.getUMat();
        pyrDown(img, downNext);

        for (int i = 1; i < num_levels; ++i) {
            UMat lvl_up;
            UMat lvl_down;

            pyrDown(downNext, lvl_down);
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[i - 1], noArray(), CV_16S);

            current = downNext;
            downNext = lvl_down;
        }

        {
            UMat lvl_up;
            pyrUp(downNext, lvl_up, current.size());
            subtract(current, lvl_up, pyr[num_levels - 1], noArray(), CV_16S);

            downNext.convertTo(pyr[num_levels], CV_16S);
        }
    } else {
        pyr[0] = img.getUMat();
        for (int i = 0; i < num_levels; ++i)
            pyrDown(pyr[i], pyr[i + 1]);
        UMat tmp;
        for (int i = 0; i < num_levels; ++i) {
            pyrUp(pyr[i + 1], tmp, pyr[i].size());
            subtract(pyr[i], tmp, pyr[i]);  // 即 pyr[i] -= tmp
        }
    }
}


void createLaplacePyrGpu(InputArray img, int num_levels, std::vector<UMat>& pyr)
{
    CV_UNUSED(img);
    CV_UNUSED(num_levels);
    CV_UNUSED(pyr);
    CV_Error(Error::StsNotImplemented, "CUDA optimization is unavailable");
}


void restoreImageFromLaplacePyr(std::vector<UMat>& pyr)
{
    if (pyr.empty())
        return;
    UMat tmp;
    for (size_t i = pyr.size() - 1; i > 0; --i) {
        pyrUp(pyr[i], tmp, pyr[i - 1].size());
        add(tmp, pyr[i - 1], pyr[i - 1]);
    }
}


void restoreImageFromLaplacePyrGpu(std::vector<UMat>& pyr)
{
    CV_UNUSED(pyr);
    CV_Error(Error::StsNotImplemented, "CUDA optimization is unavailable");
}

}  // namespace ms
