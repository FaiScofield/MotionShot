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

#include "BGDifference.h"
#include "utility.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace ms;

InputType g_type;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Arguments: <video_file> (or <data_path>) [delta]" << endl;
        exit(-1);
    }

    string dataPath(argv[1]); // "不要以/结尾"
    boost::filesystem::path path(dataPath);
    if (boost::filesystem::is_directory(path))
        g_type = DATASET;
    else
        g_type = VIDEO;

    VideoCapture vc;
    vector<string> vImags, vMaskGTs;
    if (g_type == VIDEO) {
        vc.open(dataPath);
        if (!vc.isOpened()) {
            cerr << "Unable to open video file: " << dataPath << endl;
            exit(-1);
        }
    } else {
        assert(g_type == DATASET);

        string gtPath = dataPath + "-GT/";
        ReadImageFiles(dataPath, vImags);
        ReadImageGTFiles(gtPath, vMaskGTs);
    }

    // 原图像
    Mat pFrame;
    // 原始OTSU算法输出图像
    Mat pFroundImg;
    // 背景图像
    Mat pBackgroundImg;
    // 改进的OTSU算法输出图像
    Mat pFroundImg_c;
    Mat pBackgroundImg_c;

    //视频控制全局变量,
    // 's' 画面stop
    // 'q' 退出播放
    // 'p' 打印OTSU算法中找到的阈值
    char ctrl = NULL;

    BGDiff BGDif;

    int nFrmNum = 0;

    // 逐帧读取视频
    vc >> pFrame;
    while (!pFrame.empty()) {
        vc >> pFrame;
        nFrmNum++;

        // 视频控制
        if ((ctrl = cvWaitKey(1000 / 25)) == 's')
            waitKey(0);
        else if (ctrl == 'p')
            cout << "Current Frame = " << nFrmNum << endl;
        else if (ctrl == 'q')
            break;

        // OpenCV自带OTSU
        BGDif.BackgroundDiff(pFrame, pFroundImg, pBackgroundImg, nFrmNum, CV_THRESH_OTSU);
        // 阈值筛选后的OTSU
        BGDif.BackgroundDiff(pFrame, pFroundImg_c, pBackgroundImg_c, nFrmNum, CV_THRESH_BINARY);

        // 显示图像
        imshow("Source Video", pFrame);
        imshow("Background", pBackgroundImg);
        imshow("OTSU ForeGround", pFroundImg);
        imshow("Advanced OTSU ForeGround", pFroundImg_c);
    }
    return 0;
}
