/*=================================================================
 * Extract Background & Foreground Model by ViBe Algorithm using OpenCV Library.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/

#include "BaseMotionDetector.h"
#include "opencv2/opencv.hpp"
#include <cstdio>
#include <iostream>

namespace ms
{

class ViBe : public BaseMotionDetector
{
public:
    ViBe(int num_sam = 20,    // Number of pixel's samples
         int min_match = 2,   // Match Number of make pixel as Background
         int r = 20,          // Radius of pixel value
         int rand_sam = 16);  // the probability of random sample
    ~ViBe();

    // 背景模型初始化
    // Init Background Model.
    void init(const cv::Mat& img);

    // 处理第一帧图像
    // Process First Frame of Video Query
    void ProcessFirstFrame(const cv::Mat& img);

    // 运行 ViBe 算法，提取前景区域并更新背景模型样本库
    // Run the ViBe Algorithm: Extract Foreground Areas & Update Background Model Sample Library.
    void Run(const cv::Mat& img);

    // 获取前景模型二值图像
    // get Foreground Model Binary Image.
    cv::Mat getFGModel();

    // 删除样本库
    // Delete Sample Library.
    void deleteSamples();

    // x的邻居点
    // x's neighborhood points
    int c_xoff[9];

    // y的邻居点
    // y's neighborhood points
    int c_yoff[9];

private:
    // 样本库
    // Sample Library, size = img.rows * img.cols *  DEFAULT_NUM_SAMPLES
    unsigned char*** samples;

    // 前景模型二值图像
    // Foreground Model Binary Image
    cv::Mat FGModel;

    // 每个像素点的样本个数
    // Number of pixel's samples
    int num_samples;

    // #min指数
    // Match Number of make pixel as Background
    int num_min_matches;

    // Sqthere半径
    // Radius of pixel value
    int radius;

    // 子采样概率
    // the probability of random sample
    int random_sample;
};


#define NUM_SAMPLES 20       //每个像素点的样本个数
#define MIN_MATCHES 2        //#min指数
#define RADIUS 20            // Sqthere半径
#define SUBSAMPLE_FACTOR 16  //子采样概率


class ViBe_BGS
{
public:
    ViBe_BGS(void);
    ~ViBe_BGS(void);

    void init(const cv::Mat _image);  //初始化
    void processFirstFrame(const cv::Mat _image);
    void testAndUpdate(const cv::Mat _image);  //更新
    cv::Mat getMask(void) { return m_mask; }

private:
    cv::Mat m_samples[NUM_SAMPLES];
    cv::Mat m_foregroundMatchCount;
    cv::Mat m_mask;
};

}  // namespace ms
