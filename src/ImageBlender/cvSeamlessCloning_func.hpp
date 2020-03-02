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
#include <opencv2/photo.hpp>

#define ENABLE_DEBUG    1

#if ENABLE_DEBUG
#include <iostream>
#include <opencv2/highgui.hpp>
#endif

namespace ms
{


using namespace std;
using namespace cv;

static Mat cvCheckMask(InputArray _mask, Size size)
{
    Mat mask = _mask.getMat();
    Mat gray;
    if (mask.channels() == 3)
        cvtColor(mask, gray, COLOR_BGR2GRAY);
    else {
        if (mask.empty())
            gray = Mat(size.height, size.width, CV_8UC1, Scalar(255));
        else
            mask.copyTo(gray);
    }

    return gray;
}

/**
 * @brief cvSeamlessClone   OpenCV的无缝融合函数接口, <泊松编辑>论文中两种应用
 * 这个函数就是把ROI对应的区域裁剪出来, 减少计算区域, 为调用类的API传入参数
 * @param _src      前景图像
 * @param _dst      背景图像
 * @param _mask     前景图像的掩模
 * @param p         背景中待融合区域的中心点,
 * @param _blend    融合结果
 * @param flags     融合方式, 就 NORMAL_CLONE 和 case MIXED_CLONE 两种
 * @cite    03'<Poisson image editing>
 */
void cvSeamlessClone(InputArray _src, InputArray _dst, InputArray _mask, Point p, OutputArray _blend, int flags)
{
    const Mat src = _src.getMat();
    const Mat dest = _dst.getMat();
    Mat mask = cvCheckMask(_mask, src.size());
    dest.copyTo(_blend);
    Mat blend = _blend.getMat();

    // 将前景掩模上下左右各一行/列的值置0, 防止掩模提取到边界上的像素
    Mat mask_inner = mask(Rect(1, 1, mask.cols - 2, mask.rows - 2));
    copyMakeBorder(mask_inner, mask, 1, 1, 1, 1, BORDER_ISOLATED | BORDER_CONSTANT, Scalar(0));

    Rect roi_s = boundingRect(mask); // 前景的ROI(不包括边界)
    if (roi_s.empty())
        return;
    Rect roi_d(p.x - roi_s.width / 2, p.y - roi_s.height / 2, roi_s.width, roi_s.height); // 背景的ROI

    Mat destinationROI = dest(roi_d).clone();
    Mat sourceROI = Mat::zeros(roi_s.height, roi_s.width, src.type());  // 应用了掩模后的前景
    src(roi_s).copyTo(sourceROI, mask(roi_s));

    Mat maskROI = mask(roi_s);  // 浅拷贝
    Mat recoveredROI = blend(roi_d);  // 浅拷贝

#if ENABLE_DEBUG
    imshow("destinationROI", destinationROI);
    imshow("sourceROI", sourceROI);
    imshow("maskROI", maskROI);
    waitKey(0);
    destroyAllWindows();
#endif

    cvSeamlessCloning obj;
    obj.normalClone(destinationROI, sourceROI, maskROI, recoveredROI, flags);
}

void cvColorChange(InputArray _src, InputArray _mask, OutputArray _dst, float red, float green, float blue)
{
    Mat src = _src.getMat();
    Mat mask = cvCheckMask(_mask, src.size());
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();

    Mat cs_mask = Mat::zeros(src.size(), src.type());
    src.copyTo(cs_mask, mask);

    cvSeamlessCloning obj;
    obj.localColorChange(src, cs_mask, mask, blend, red, green, blue);
}

void cvIlluminationChange(InputArray _src, InputArray _mask, OutputArray _dst, float alpha, float beta)
{
    Mat src = _src.getMat();
    Mat mask = cvCheckMask(_mask, src.size());
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();

    Mat cs_mask = Mat::zeros(src.size(), src.type());
    src.copyTo(cs_mask, mask);

    cvSeamlessCloning obj;
    obj.illuminationChange(src, cs_mask, mask, blend, alpha, beta);
}

void cvTextureFlattening(InputArray _src, InputArray _mask, OutputArray _dst, float low_threshold,
                           float high_threshold, int kernel_size)
{
    Mat src = _src.getMat();
    Mat mask = cvCheckMask(_mask, src.size());
    _dst.create(src.size(), src.type());
    Mat blend = _dst.getMat();

    Mat cs_mask = Mat::zeros(src.size(), src.type());
    src.copyTo(cs_mask, mask);

    cvSeamlessCloning obj;
    obj.textureFlatten(src, cs_mask, mask, low_threshold, high_threshold, kernel_size, blend);
}

}  // namespace ms
