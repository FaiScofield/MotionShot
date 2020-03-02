#include "utility.h"
#include <iostream>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>


#define USE_OPENCV_BLENDER  0   //! NOTE OpenCV融合没有考虑时序
#define BLEND_FOREGROUND    1

#if USE_OPENCV_BLENDER
#include <opencv2/stitching.hpp>
#else
#include "ImageBlender/cvBlenders.h"
#include "ImageBlender/PoissonBlender.h"
#include "ImageBlender/cvSeamlessCloning.h"
#include "ImageBlender/cvSeamlessCloning_func.hpp"
#endif

using namespace std;
using namespace cv;
using namespace ms;

enum BlenderType { NO, FEATHER, MULTI_BAND, POISSON_BLAND };

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder     f| |LASIESTA dataset folder}"
               "{blender    b|multiband|valid blend type: \"feather\", \"multiband\", \"poission\"}"
               "{delta      d|10|interval from frame to frame for foreground. If delta = 0, output 3~10 images}"
               "{start      s|0|start index for image sequence}"
               "{end        e|-1|end index for image sequence}"
               "{help       h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    TickMeter timer;
    timer.start();

    String str_blender = parser.get<String>("blender");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    cout << " - folder = " << str_folder << endl;
    cout << " - blender = " << str_blender << endl;

    BlenderType blenderType = NO;
#if USE_OPENCV_BLENDER
    detail::Blender* blender = nullptr;
    if (str_blender == "feather") {
        blender = dynamic_cast<detail::Blender*>(new detail::FeatherBlender());
        blenderType = FEATHER;
    } else if (str_blender == "multiband") {
        blender = dynamic_cast<detail::Blender*>(new detail::MultiBandBlender(false, 5, CV_32F));
        blenderType = MULTI_BAND;
    }
#else
    ms::cvBlender* blender = nullptr;
    if (str_blender == "feather") {
        blender = new ms::cvFeatherBlender(1.0, true);
        blenderType = FEATHER;
    } else if (str_blender == "multiband") {
        blender = new ms::cvMultiBandBlender(false, 5, CV_32F);
        blenderType = MULTI_BAND;
    } /*else if (str_blender == "poission") {
        blender = new ms::PoissionBlender(); // TODO
    }*/
#endif
    else {
        cerr << "[Error] Unknown blender type for " << str_blender << endl;
        exit(-1);
    }

    string strMode1;
    if (blenderType == FEATHER)
        strMode1 = "羽化";
    else if (blenderType == MULTI_BAND)
        strMode1 = "多频带";

    int start = parser.get<int>("start");
    int end = parser.get<int>("end");
    int delta = parser.get<int>("delta");
    cout << " - start index = " << start << endl;
    cout << " - end index = " << end << endl;
    cout << " - delta = " << delta << endl;

    vector<Mat> vImages, vGTsColor, vForegroundMasks;
    ReadImageSequence_lasiesta(str_folder, vImages, vGTsColor, start, end - start);
    if (vImages.empty() || vGTsColor.empty())
        exit(-1);
    int N = vImages.size();
    assert(N > 3);
    assert((delta == 0) || (delta > 1 && delta < N));
    assert(vImages.size() == vGTsColor.size());

    vForegroundMasks.reserve(N);
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m) {
        Mat mask, mask2, maskBin;
        cvtColor(m, mask, COLOR_BGR2GRAY);  // 有颜色, 转灰度后不一定是255
//        erode(mask, mask2, getStructuringElement(MORPH_RECT, Size(3, 3))); // 边缘1个Pixel保持不变, 其他设为255
        compare(mask, 0, maskBin, CMP_GT);
        mask.setTo(255, maskBin);
        vForegroundMasks.push_back(mask);
    });

    /// calc the result in different gaps
    int maxFores = 10, minFores = 3; // 前景最多存在8个, 最少3个
    if (delta != 0)
        maxFores = minFores = N / delta;
    vector<Mat> vImgsToProcess;
    vector<int> vIdxToProcess;
    vector<vector<int>> vvIdxPerIter;
    extractImagesToStitch(vImages, vImgsToProcess, vIdxToProcess, vvIdxPerIter, minFores, maxFores);

    timer.stop();
    TIMER("系统初始化耗时(s): " << timer.getTimeSec());

    /// main loop
    for (int k = minFores; k <= maxFores; ++k) { // k为前景数量
        timer.start();

        // 1.前景拼接融合
        const vector<int>& vIdxThisIter = vvIdxPerIter[k - minFores];
        const Mat pano = vImgsToProcess.front().clone();
        const Rect disRoi(0, 0, pano.cols, pano.rows);

#if BLEND_FOREGROUND
        blender->prepare(disRoi);

        const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        for (size_t j = 0, jend = vIdxThisIter.size(); j < jend; ++j) {
            const int& imgIdx = vIdxThisIter[j];
            const Mat& frame = vImages[imgIdx];
            const Mat& foreMask = vForegroundMasks[imgIdx];

            // blender
            Mat frame_S, maskFiltered;
            frame.convertTo(frame_S, CV_16SC3);
//            smoothMaskWeightEdge(foreMask, maskFiltered, 5); // 过渡边缘
            dilate(foreMask, maskFiltered, kernel);   // 膨胀
            erode(maskFiltered, maskFiltered, kernel);   // 腐蚀
            blender->feed(frame_S, maskFiltered, Point(0, 0));
        }

        Mat allForeground, allForeground_S, allForegroundMask;
        blender->blend(allForeground_S, allForegroundMask);
        allForeground_S.convertTo(allForeground, CV_8UC3);
#else
        Mat allForeground, allForeground_S, allForegroundMask;
        allForeground = Mat::zeros(pano.size(), CV_8UC3);
        allForegroundMask = Mat::zeros(pano.size(), CV_8UC1);
        for (size_t j = 0, jend = vIdxThisIter.size(); j < jend; ++j) {
            const int& imgIdx = vIdxThisIter[j];
            const Mat& frame = vImages[imgIdx];
            const Mat& foreMask = vForegroundMasks[imgIdx];

            frame.copyTo(allForeground, foreMask);
            add(allForegroundMask, foreMask, allForegroundMask);
        }
        allForeground.convertTo(allForeground_S, CV_16SC3);
#endif
//        Mat tmp1, tmp2;
//        cvtColor(allForegroundMask, tmp1, COLOR_GRAY2BGR);
//        hconcat(allForeground, tmp1, tmp2);
//        imshow("allForeground and mask", tmp2);
//        string txt1 = "/home/vance/output/" + to_string(k) + "个前景拼接融合结果-" + strMode1 + ".jpg";
//        imwrite(txt1, tmp2);

        // 2.前背景融合. 把pano和前景对应的mask区域的稍微缩收/扩张, 设置平滑的权重过渡, 然后再融合.
        //! (不能用羽化, 羽化不能自定义权重) 目前看多频段的效果还不如羽化!
#if USE_OPENCV_BLENDER
        detail::FeatherBlender* blender2 = new detail::FeatherBlender();
#else
//        ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender(1.0, false);
//        const string strMode2 = "羽化";
        ms::PoissonBlender* blender2 = new ms::PoissonBlender();
        const string strMode2 = "泊松";
        BlenderType blenderType2 = POISSON_BLAND;
#endif
        blender2->prepare(disRoi);

        Mat foregroundMaskFinal, backgroundMaskFinal, maskFinal;
        smoothMaskWeightEdge(allForegroundMask, foregroundMaskFinal, 5); // 过渡边缘
        bitwise_not(foregroundMaskFinal, backgroundMaskFinal);
        hconcat(foregroundMaskFinal, backgroundMaskFinal, maskFinal);
//        imshow("foreground & background maskFinal", maskFinal);

        Mat fmf, fmfOut;
        cvtColor(foregroundMaskFinal, fmf, COLOR_GRAY2BGR);
        hconcat(allForeground, fmf, fmfOut);
//        string txt0 = "/home/vance/output/前景与掩模-" + strMode1 + "-" + to_string(k) + ".jpg";
//        imwrite(txt0, fmfOut);0

        Mat pano_S, result, resultMask;
        pano.convertTo(pano_S, CV_16SC3);
        if (blenderType2 == POISSON_BLAND) {
            // foregroundMaskFinal to renctangle to center point
//            Mat dilatedMask;
//            dilate(allForegroundMask, dilatedMask, Mat());
            Rect2i foregroundRect = boundingRect(/*dilatedMask*/allForegroundMask);
            Point center = (foregroundRect.tl() + foregroundRect.br()) / 2;
//            Mat foregroundRectShow = pano.clone();
//            rectangle(foregroundRectShow, foregroundRect, Scalar(0,0,255));
//            circle(foregroundRectShow, center, 2, Scalar(0,0,255));
//            imshow("foregroundRectShow", foregroundRectShow);
//            waitKey(0);
//            blender2->feed(allForeground_S, foregroundMaskFinal, center);
            Mat blended;
            ms::cvSeamlessClone(allForeground, pano, allForegroundMask, center, blended, cv::NORMAL_CLONE);
            imshow("SeamlessClone blended", blended);
            waitKey(0);
        } else {
            blender2->feed(pano_S, backgroundMaskFinal, Point(0, 0));
            blender2->feed(allForeground_S, foregroundMaskFinal, Point(0, 0));
        }

//        blender2->blend(result, resultMask);
//        result.convertTo(result, CV_8U);

//        hconcat(fmfOut, result, result);
//        imshow("blend result", result);

//        string txt2 = "/home/vance/output/" + to_string(k) + "个前景最终结果-" + strMode1 + "+" + strMode2 + ".jpg";
//        imwrite(txt2, result);

        waitKey(100);

        timer.stop();
        TIMER("间隔" << N / k << "帧, 算法整体耗时(s): " << timer.getTimeSec());
    }

    waitKey(0);
    return 0;
}
