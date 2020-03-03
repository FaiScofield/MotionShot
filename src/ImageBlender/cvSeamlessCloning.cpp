/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "cvSeamlessCloning.h"

#define ENABLE_DEBUG    0

#if ENABLE_DEBUG
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

namespace ms
{

using namespace cv;
using namespace std;


void cvSeamlessCloning::computeGradientX(const Mat& img, Mat& gx)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0, 2) = 1;
    kernel.at<char>(0, 1) = -1;

    if (img.channels() == 3) {
        filter2D(img, gx, CV_32F, kernel);
    } else if (img.channels() == 1) {
        Mat tmp[3];
        for (int chan = 0; chan < 3; ++chan) {
            filter2D(img, tmp[chan], CV_32F, kernel);
        }
        merge(tmp, 3, gx);
    }
}

void cvSeamlessCloning::computeGradientY(const Mat& img, Mat& gy)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(2, 0) = 1;
    kernel.at<char>(1, 0) = -1;

    if (img.channels() == 3) {
        filter2D(img, gy, CV_32F, kernel);
    } else if (img.channels() == 1) {
        Mat tmp[3];
        for (int chan = 0; chan < 3; ++chan) {
            filter2D(img, tmp[chan], CV_32F, kernel);
        }
        merge(tmp, 3, gy);
    }
}

void cvSeamlessCloning::computeLaplacianX(const Mat& img, Mat& laplacianX)
{
    Mat kernel = Mat::zeros(1, 3, CV_8S);
    kernel.at<char>(0, 0) = -1;
    kernel.at<char>(0, 1) = 1;
    filter2D(img, laplacianX, CV_32F, kernel);
}

void cvSeamlessCloning::computeLaplacianY(const Mat& img, Mat& laplacianY)
{
    Mat kernel = Mat::zeros(3, 1, CV_8S);
    kernel.at<char>(0, 0) = -1;
    kernel.at<char>(1, 0) = 1;
    filter2D(img, laplacianY, CV_32F, kernel);
}

void cvSeamlessCloning::dst(const Mat& src, Mat& dest, bool invert)
{
    Mat temp = Mat::zeros(src.rows, 2 * src.cols + 2, CV_32F);

    int flag = invert ? DFT_ROWS + DFT_SCALE + DFT_INVERSE : DFT_ROWS;

    src.copyTo(temp(Rect(1, 0, src.cols, src.rows)));

    for (int j = 0; j < src.rows; ++j) {
        float* tempLinePtr = temp.ptr<float>(j);
        const float* srcLinePtr = src.ptr<float>(j);
        for (int i = 0; i < src.cols; ++i) {
            tempLinePtr[src.cols + 2 + i] = -srcLinePtr[src.cols - 1 - i];
        }
    }

    Mat planes[] = {temp, Mat::zeros(temp.size(), CV_32F)};
    Mat complex;

    merge(planes, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes);
    temp = Mat::zeros(src.cols, 2 * src.rows + 2, CV_32F);

    for (int j = 0; j < src.cols; ++j) {
        float* tempLinePtr = temp.ptr<float>(j);
        for (int i = 0; i < src.rows; ++i) {
            float val = planes[1].ptr<float>(i)[j + 1];
            tempLinePtr[i + 1] = val;
            tempLinePtr[temp.cols - 1 - i] = -val;
        }
    }

    Mat planes2[] = {temp, Mat::zeros(temp.size(), CV_32F)};

    merge(planes2, 2, complex);
    dft(complex, complex, flag);
    split(complex, planes2);

    temp = planes2[1].t();
    temp(Rect(0, 1, src.cols, src.rows)).copyTo(dest);
}

/**
 * @brief cvSeamlessCloning::solve  解泊松方程
 * 这个函数其实功能是快速求解泊松方程的一种方法，就是针对AX=B，由于泊松方程系数矩阵的特殊性
 * @param img       背景图
 * @param mod_diff  散度, 大小为(w-2)*(h-2), 最外围值散度为0
 * @param result    结果
 */
void cvSeamlessCloning::solve(const Mat& img, Mat& mod_diff, Mat& result)
{
    const int w = img.cols;
    const int h = img.rows;

    Mat res;
    dst(mod_diff, res);

    for (int j = 0; j < h - 2; j++) {
        float* resLinePtr = res.ptr<float>(j);
        for (int i = 0; i < w - 2; i++) {
            resLinePtr[i] /= (filter_X[i] + filter_Y[j] - 4);
        }
    }

    dst(res, mod_diff, true);

    unsigned char* resLinePtr = result.ptr<unsigned char>(0);
    const unsigned char* imgLinePtr = img.ptr<unsigned char>(0);
    const float* interpLinePtr = NULL;

    // first col
    for (int i = 0; i < w; ++i)
        result.ptr<unsigned char>(0)[i] = img.ptr<unsigned char>(0)[i];

    for (int j = 1; j < h - 1; ++j) {
        resLinePtr = result.ptr<unsigned char>(j);
        imgLinePtr = img.ptr<unsigned char>(j);
        interpLinePtr = mod_diff.ptr<float>(j - 1);

        // first row
        resLinePtr[0] = imgLinePtr[0];

        for (int i = 1; i < w - 1; ++i) {
            // saturate cast is not used here, because it behaves differently from the previous implementation
            // most notable, saturate_cast rounds before truncating, here it's the opposite.
            float value = interpLinePtr[i - 1];
            if (value < 0.)
                resLinePtr[i] = 0;
            else if (value > 255.0)
                resLinePtr[i] = 255;
            else
                resLinePtr[i] = static_cast<unsigned char>(value);
        }

        // last row
        resLinePtr[w - 1] = imgLinePtr[w - 1];
    }

    // last col
    resLinePtr = result.ptr<unsigned char>(h - 1);
    imgLinePtr = img.ptr<unsigned char>(h - 1);
    for (int i = 0; i < w; ++i)
        resLinePtr[i] = imgLinePtr[i];
}

void cvSeamlessCloning::poissonSolver(const Mat& img, Mat& laplacianX, Mat& laplacianY, Mat& result)
{
    const int w = img.cols;
    const int h = img.rows;

    Mat lap = laplacianX + laplacianY;  // 散度

    Mat bound = img.clone();

    // 边界修正，opencv为了方便，直接把图片最外围的像素点排除在外，不参与泊松重建
    rectangle(bound, Point(1, 1), Point(img.cols - 2, img.rows - 2), Scalar::all(0), -1);
    Mat boundary_points;
    Laplacian(bound, boundary_points, CV_32F);

    boundary_points = lap - boundary_points;

    Mat mod_diff = boundary_points(Rect(1, 1, w - 2, h - 2));

    solve(img, mod_diff, result);
}

void cvSeamlessCloning::initVariables(const Mat& destination, const Mat& binaryMask)
{
    destinationGradientX = Mat(destination.size(), CV_32FC3);
    destinationGradientY = Mat(destination.size(), CV_32FC3);
    patchGradientX = Mat(destination.size(), CV_32FC3);
    patchGradientY = Mat(destination.size(), CV_32FC3);

    binaryMaskFloat = Mat(binaryMask.size(), CV_32FC1);
    binaryMaskFloatInverted = Mat(binaryMask.size(), CV_32FC1);

    // init of the filters used in the dst
    const int w = destination.cols;
    filter_X.resize(w - 2);
    double scale = CV_PI / (w - 1);
    for (int i = 0; i < w - 2; ++i)
        filter_X[i] = 2.0f * (float)std::cos(scale * (i + 1));

    const int h = destination.rows;
    filter_Y.resize(h - 2);
    scale = CV_PI / (h - 1);
    for (int j = 0; j < h - 2; ++j)
        filter_Y[j] = 2.0f * (float)std::cos(scale * (j + 1));
}

void cvSeamlessCloning::computeDerivatives(const Mat& destination, const Mat& patch, const Mat& binaryMask)
{
    initVariables(destination, binaryMask);

    // 计算前背景的梯度
    computeGradientX(destination, destinationGradientX);
    computeGradientY(destination, destinationGradientY);

    computeGradientX(patch, patchGradientX);
    computeGradientY(patch, patchGradientY);

#if ENABLE_DEBUG
    imshow("input binaryMask", binaryMask);
    Mat tmp = binaryMask.clone();
#endif

    Mat Kernel(Size(3, 3), CV_8UC1);
    Kernel.setTo(Scalar(1));
    erode(binaryMask, binaryMask, Kernel, Point(-1, -1), 3);  //? 腐蚀3个像素 ? 这不是const 吗?

#if ENABLE_DEBUG
    Mat diff;
    absdiff(tmp, binaryMask, diff);
    imshow("diff after eroding", diff);
    waitKey(0);
    destroyAllWindows();
#endif

    binaryMask.convertTo(binaryMaskFloat, CV_32FC1, 1.0 / 255.0);
}

void cvSeamlessCloning::scalarProduct(Mat mat, float r, float g, float b)
{
    vector<Mat> channels;
    split(mat, channels);
    multiply(channels[2], r, channels[2]);
    multiply(channels[1], g, channels[1]);
    multiply(channels[0], b, channels[0]);
    merge(channels, mat);
}

void cvSeamlessCloning::arrayProduct(const cv::Mat& lhs, const cv::Mat& rhs, cv::Mat& result) const
{
    vector<Mat> lhs_channels;
    vector<Mat> result_channels;

    split(lhs, lhs_channels);
    split(result, result_channels);

    for (int chan = 0; chan < 3; ++chan)
        multiply(lhs_channels[chan], rhs, result_channels[chan]);

    merge(result_channels, result);
}

// 泊松重建
void cvSeamlessCloning::poisson(const Mat& destination)
{
    Mat laplacianX = destinationGradientX + patchGradientX;
    Mat laplacianY = destinationGradientY + patchGradientY;

    computeLaplacianX(laplacianX, laplacianX);
    computeLaplacianY(laplacianY, laplacianY);

    split(laplacianX, rgbx_channel);
    split(laplacianY, rgby_channel);

    split(destination, output);

    for (int chan = 0; chan < 3; ++chan) {
        poissonSolver(output[chan], rgbx_channel[chan], rgby_channel[chan], output[chan]);
    }
}

/**
 * @brief cvSeamlessCloning::evaluate
 * @param I         destination
 * @param wmask     带权的浮点型mask(经过3像素的腐蚀), 由binaryMask传入
 * @param cloned    cloned
 */
void cvSeamlessCloning::evaluate(const Mat& I, const Mat& wmask, const Mat& cloned)
{
    bitwise_not(wmask, wmask);

    wmask.convertTo(binaryMaskFloatInverted, CV_32FC1, 1.0 / 255.0);

    arrayProduct(destinationGradientX, binaryMaskFloatInverted, destinationGradientX);
    arrayProduct(destinationGradientY, binaryMaskFloatInverted, destinationGradientY);

    poisson(I);

    merge(output, cloned);
}

/**
 * @brief cvSeamlessCloning::normalClone 外部接口函数
 * @param destination   背景图
 * @param patch         前景图
 * @param binaryMask    前景掩模
 * @param cloned        目标图像对应区域
 * @param flag          融合方式, 有 NORMAL_CLONE, MIXED_CLONE, MONOCHROME_TRANSFER(细节风格转换,见论文图5) 3种
 */
void cvSeamlessCloning::normalClone(const Mat& destination, const Mat& patch, const Mat& binaryMask, Mat& cloned, int flag)
{
    const int w = destination.cols;
    const int h = destination.rows;
    const int channel = destination.channels();
    const int n_elem_in_line = w * channel;

    computeDerivatives(destination, patch, binaryMask); // 计算梯度, 腐蚀前景掩模三个像素(去除毛刺)

    switch (flag) {
    case NORMAL_CLONE:
        arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
        arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
        break;

    case MIXED_CLONE: {
        AutoBuffer<int> maskIndices(n_elem_in_line);
        for (int i = 0; i < n_elem_in_line; ++i)
            maskIndices[i] = i / channel;

        for (int i = 0; i < h; i++) {
            float* patchXLinePtr = patchGradientX.ptr<float>(i);
            float* patchYLinePtr = patchGradientY.ptr<float>(i);
            const float* destinationXLinePtr = destinationGradientX.ptr<float>(i);
            const float* destinationYLinePtr = destinationGradientY.ptr<float>(i);
            const float* binaryMaskLinePtr = binaryMaskFloat.ptr<float>(i);

            for (int j = 0; j < n_elem_in_line; j++) {
                int maskIndex = maskIndices[j];

                if (abs(patchXLinePtr[j] - patchYLinePtr[j]) >
                    abs(destinationXLinePtr[j] - destinationYLinePtr[j])) {
                    patchXLinePtr[j] *= binaryMaskLinePtr[maskIndex];
                    patchYLinePtr[j] *= binaryMaskLinePtr[maskIndex];
                } else {
                    patchXLinePtr[j] = destinationXLinePtr[j] * binaryMaskLinePtr[maskIndex];
                    patchYLinePtr[j] = destinationYLinePtr[j] * binaryMaskLinePtr[maskIndex];
                }
            }
        }
    } break;

    case MONOCHROME_TRANSFER:
        Mat gray;
        cvtColor(patch, gray, COLOR_BGR2GRAY);

        computeGradientX(gray, patchGradientX);
        computeGradientY(gray, patchGradientY);

        arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
        arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
        break;
    }

    evaluate(destination, binaryMask, cloned);
}

void cvSeamlessCloning::localColorChange(Mat& I, Mat& mask, Mat& wmask, Mat& cloned, float red_mul = 1.0,
                               float green_mul = 1.0, float blue_mul = 1.0)
{
    computeDerivatives(I, mask, wmask);

    arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);
    scalarProduct(patchGradientX, red_mul, green_mul, blue_mul);
    scalarProduct(patchGradientY, red_mul, green_mul, blue_mul);

    evaluate(I, wmask, cloned);
}

void cvSeamlessCloning::illuminationChange(Mat& I, Mat& mask, Mat& wmask, Mat& cloned, float alpha, float beta)
{
    //CV_INSTRUMENT_REGION();

    computeDerivatives(I, mask, wmask);

    arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);

    Mat mag;
    magnitude(patchGradientX, patchGradientY, mag);

    Mat multX, multY, multx_temp, multy_temp;

    multiply(patchGradientX, pow(alpha, beta), multX);
    pow(mag, -1 * beta, multx_temp);
    multiply(multX, multx_temp, patchGradientX);
    patchNaNs(patchGradientX);

    multiply(patchGradientY, pow(alpha, beta), multY);
    pow(mag, -1 * beta, multy_temp);
    multiply(multY, multy_temp, patchGradientY);
    patchNaNs(patchGradientY);

    Mat zeroMask = (patchGradientX != 0);

    patchGradientX.copyTo(patchGradientX, zeroMask);
    patchGradientY.copyTo(patchGradientY, zeroMask);

    evaluate(I, wmask, cloned);
}

void cvSeamlessCloning::textureFlatten(Mat& I, Mat& mask, Mat& wmask, float low_threshold,
                             float high_threshold, int kernel_size, Mat& cloned)
{
    computeDerivatives(I, mask, wmask);

    Mat out;
    Canny(mask, out, low_threshold, high_threshold, kernel_size);

    Mat zeros = Mat::zeros(patchGradientX.size(), CV_32FC3);
    Mat zerosMask = (out != 255);
    zeros.copyTo(patchGradientX, zerosMask);
    zeros.copyTo(patchGradientY, zerosMask);

    arrayProduct(patchGradientX, binaryMaskFloat, patchGradientX);
    arrayProduct(patchGradientY, binaryMaskFloat, patchGradientY);

    evaluate(I, wmask, cloned);
}

}  // namespace ms
