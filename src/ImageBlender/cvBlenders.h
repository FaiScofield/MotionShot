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

//! 注意这里宏定义要做修改, 否则OpenCV中的原头文件将无法被include
#ifndef OPENCV_STITCHING_BLENDERS_CUSTUM_H
#define OPENCV_STITCHING_BLENDERS_CUSTUM_H

#include "precompiled.h"
#include <opencv2/core.hpp>

namespace ms
{

/**
 * @brief 融合基类
 * dst_(输出全图) 和 dst_mask_(输出掩模) 是关键量.
 * feed() 是关键函数, 在于 dst_ 和 dst_mask_ 的更新
 */
class cvBlender
{
public:
    virtual ~cvBlender() {}

    enum BlenderType { NO, FEATHER, MULTI_BAND };

    static Ptr<cvBlender> createDefault(int type, bool try_gpu = false);

    /**
     * @brief Prepares the blender for blending.
     * @param corners Source images top-left corners
     * @param sizes Source image sizes
     */
    virtual void prepare(const std::vector<Point>& corners, const std::vector<Size>& sizes);
    /** @overload */
    virtual void prepare(Rect dst_roi);

    /**
     * @brief Processes the image. 关键函数, 本质是更新 dst_ 和 dst_mask_
     * @param img Source image
     * @param mask Source image mask
     * @param tl Source image top-left corners
     */
    virtual void feed(InputArray img, InputArray mask, Point tl);

    /**
     * @brief Blends and returns the final pano.
     * @param dst Final pano
     * @param dst_mask Final pano mask
     */
    virtual void blend(InputOutputArray dst, InputOutputArray dst_mask);

    /**
     * @brief 获得中间结果.
     * 实际上在feed()以后融合已经做完了, 该函数可以不用调用blend()获取feed()后的结果
     */
    virtual void getMiddleResult(InputOutputArray dst, InputOutputArray dst_mask);

    Rect getDstROI() const { return dst_roi_; }

    Mat getOverlappedEdges() const { return overlapped_edges_; }
    Mat getOverlappedEdgesMask(int size) const;

    void setOverlappedEdgesMaskWidth(int w)
    {
        assert(w > 1);
        overlapped_edges_mask_width_ = w;
    }

protected:
    UMat dst_, dst_mask_;   // 输出全图 和 输出掩模
    Rect dst_roi_;          // 根据输入图像的左上角坐标和尺寸形成的最小外包围矩形

    int overlapped_edges_mask_width_;
    Mat overlapped_edges_;
    Mat overlapped_edges_mask_;
};

/**
 * @brief Simple blender which mixes images at its borders. 羽化融合
 * 最简单的加权平均, 权值由distanceTransform()决定, 图片添加顺序不影响最终效果!
 */
class cvFeatherBlender : public cvBlender
{
public:
    cvFeatherBlender(float sharpness = 0.02f, bool cover = false);

    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }
    bool enableCover() const { return enable_cover_; }
    void setEnableCover(bool flag) { enable_cover_ = flag; }

    void prepare(Rect dst_roi) override;
    void feed(InputArray img, InputArray mask, Point tl) override;
    void blend(InputOutputArray dst, InputOutputArray dst_mask) override;

    //! TODO
//    void feed(const std::vector<Mat>& vImgs, const std::vector<Mat>& vMasks,
//              const std::vector<Point>& vTopleftCorners);

    void getMiddleResult(InputOutputArray dst, InputOutputArray dst_mask) override;

    //! Creates weight maps for fixed set of source images by their masks and top-left corners.
    //! Final image can be obtained by simple weighting of the source images.
    Rect createWeightMaps(const std::vector<UMat>& masks, const std::vector<Point>& corners,
                          std::vector<UMat>& weight_maps);

    //! TODO
    void smoothEdgesOnOverlappedArea(const Mat& src, const Mat& mask, Mat& dst, float scale);

private:
    float sharpness_;
    UMat weight_map_;
    UMat dst_weight_map_;

    bool enable_cover_; // 是否考虑添加顺序的覆盖
};


/**
 * @brief   多频段融合
 * 利用拉普拉斯金字塔进行多频段融合. 拉普拉斯金字塔是通过源图像减去先缩小后再放大的图像(即图像缩小后丢失的那一部分信息)
 * $$ L_{i}=G_{i}-\operatorname{PyrUp}\left(G_{i+1}\right) $$
 *
 * @note    mask的元素值的大小也是权重的大小, 可以不用简单的0/255区分.
 * @see @cite OpenCV @ref BA83
 */
class cvMultiBandBlender : public cvBlender
{
public:
    cvMultiBandBlender(int try_gpu = false, int num_bands = 5, int weight_type = CV_32F);

    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi) override;
    void feed(InputArray img, InputArray mask, Point tl) override;
    void blend(InputOutputArray dst, InputOutputArray dst_mask) override;

    void getMiddleResult(InputOutputArray dst, InputOutputArray dst_mask) override;

private:
    int actual_num_bands_, num_bands_;    // 实际金字塔层数
    std::vector<UMat> dst_pyr_laplace_;   // 拉普拉斯金字塔
    std::vector<UMat> dst_band_weights_;  // 每层金字塔的权重
    Rect dst_roi_final_;                  // 最终mask
    bool can_use_gpu_;
    int weight_type_;  // CV_32F or CV_16S
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void normalizeUsingWeightMap(InputArray weight, InputOutputArray src);
void createWeightMap(InputArray mask, float sharpness, InputOutputArray weight);

void createLaplacePyr(InputArray img, int num_levels, std::vector<UMat>& pyr);

// Restores source image
void restoreImageFromLaplacePyr(std::vector<UMat>& pyr);


}  // namespace ms

#endif  // OPENCV_STITCHING_BLENDERS_CUSTUM_H
