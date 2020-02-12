/*=================================================================
 * Calculate Background Model of a list of Frames(Normally a video stream) in the
 * method of Background Difference Method & OTSU Algorithm By OpenCV.
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

#ifndef BGDIFFERENCE_H
#define BGDIFFERENCE_H

#include "BaseMotionDetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ms {

class BGDiff : public BaseMotionDetector
{
public:
    void apply(const cv::Mat& input, cv::Mat& mask) override;

    // 背景差分算法 Background Difference Algorithm
    void BackgroundDiff(cv::Mat src, cv::Mat& imgForeground, cv::Mat& imgBackground, int nFrmNum,
                        int threshold_method = cv::THRESH_OTSU, double updateSpeed = 0.03);

    // 大津法 OTSU Algorithm
    void Otsu(cv::Mat src, int& thresholdValue, bool ToShowValue = false);
};

}
#endif  // BGDIFFERENCE_H
