#include "utility.h"
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <set>


#define USE_OPENCV_BLENDER 0  //! NOTE OpenCV融合没有考虑时序
#define BLEND_FOREGROUND 1

#if USE_OPENCV_BLENDER
#include <opencv2/stitching.hpp>
#else
#include "ImageBlender/cvBlenders.h"
#endif

using namespace std;
using namespace cv;
using namespace ms;

enum BlenderType { NO, FEATHER, MULTI_BAND };

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder     f| |huawei SEQUENCE dataset folder}"
               "{blender    b|multiband|valid blend type: \"feather\", \"multiband\", \"poission\"}"
               "{scale      c|0.5|scale of inpute image}"
               "{begin      b|1|start index for image sequence}"
               "{end        e|10|end index for image sequence}"
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

    double scale = parser.get<double>("scale");
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
    cout << " - begin index = " << beginIdx << endl;
    cout << " - end index = " << endIdx << endl;
    cout << " - scale = " << scale << endl;

    vector<Mat> vImages, vGTsColor, vForegroundMasks;
    ReadImageSequence(str_folder, "jpg", vImages, beginIdx, endIdx - beginIdx);  // 1 ~ 12
    resizeFlipRotateImages(vImages, 0.5);
    resizeFlipRotateImages(vImages, scale);

    // 掩膜要羽化一下
    ReadImageSequence(str_folder + "-gt", "jpg", vGTsColor, beginIdx, endIdx - beginIdx);
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m) {
        erode(m, m, kernel);
    });
    resizeFlipRotateImages(vGTsColor, scale);

    if (vImages.empty() || vGTsColor.empty())
        exit(-1);
    int N = vImages.size();
    assert(N > 3);
    assert(vImages.size() == vGTsColor.size());

    vForegroundMasks.reserve(N);
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m) {
        Mat mask, mask2, maskBin;
        cvtColor(m, mask, COLOR_BGR2GRAY);  // 有颜色, 转灰度后不一定是255
        // erode(mask, mask2, getStructuringElement(MORPH_RECT, Size(3, 3))); // 边缘1个Pixel保持不变, 其他设为255
        compare(mask, 0, maskBin, CMP_GT);
        mask.setTo(255, maskBin);
        vForegroundMasks.push_back(mask);
    });

    /// calc the result in different gaps
    int maxFores = 9, minFores = 3;  // 前景最多存在9个, 最少3个
    vector<Mat> vImgsToProcess = vImages;
    vector<int> vIdxToProcess{0, 1, 2, 3, 4, 5, 6, 7, 8};
    vector<vector<int>> vvIdxPerIter;
    vvIdxPerIter.emplace_back(vector<int>{0, 3, 6});           // 3
    vvIdxPerIter.emplace_back(vector<int>{0, 3, 6, 8});
    vvIdxPerIter.emplace_back(vector<int>{0, 2, 4, 6, 8});
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 3, 4, 6, 7});  // 6
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 3, 4, 6, 7, 8});
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8});   // 9

    timer.stop();
    TIMER("系统初始化耗时(s): " << timer.getTimeSec());

    /// main loop
    for (int k = minFores; k <= maxFores; ++k) {  // k为前景数量
        timer.start();

        // 1.前景拼接融合
        const vector<int>& vIdxThisIter = vvIdxPerIter[k - minFores];
        const Mat pano = vImgsToProcess.front().clone();
        const Rect disRoi(0, 0, pano.cols, pano.rows);

#if BLEND_FOREGROUND
        blender->prepare(disRoi);

        for (size_t j = 0, jend = vIdxThisIter.size(); j < jend; ++j) {
            const int& imgIdx = vIdxThisIter[j];
            const Mat& frame = vImages[imgIdx];
            const Mat& foreMask = vForegroundMasks[imgIdx];

            // blender
            Mat frame_S;
            frame.convertTo(frame_S, CV_16SC3);
            blender->feed(frame_S, foreMask, Point(0, 0));
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
        //        string txt1 = "/home/vance/output/" + to_string(k) + "个前景拼接融合结果-" +
        //        strMode1 + ".jpg"; imwrite(txt1, tmp2);

        // 2.前背景融合. 把pano和前景对应的mask区域的稍微缩收/扩张, 设置平滑的权重过渡, 然后再融合.
        //! (不能用羽化, 羽化不能自定义权重) 目前看多频段的效果还不如羽化!
#if USE_OPENCV_BLENDER
        detail::FeatherBlender* blender2 = new detail::FeatherBlender();
#else
        ms::cvMultiBandBlender* blender2 = new ms::cvMultiBandBlender(false, 2);
//        ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender(1.0, false);
        const string strMode2 = "羽化";
#endif

        blender2->prepare(disRoi);

        Mat foregroundMaskFinal, backgroundMaskFinal, maskFinal;
        smoothMaskWeightEdge(allForegroundMask, foregroundMaskFinal, 0);  // 过渡边缘
        bitwise_not(foregroundMaskFinal, backgroundMaskFinal);
        hconcat(foregroundMaskFinal, backgroundMaskFinal, maskFinal);
//        imshow("foreground & background maskFinal", maskFinal);

        Mat fmf, fmfOut;
        cvtColor(foregroundMaskFinal, fmf, COLOR_GRAY2BGR);
        hconcat(allForeground, fmf, fmfOut);
        //        string txt0 = "/home/vance/output/前景与掩模-" + strMode1 + "-" + to_string(k) +
        //        ".jpg"; imwrite(txt0, fmfOut);

        Mat pano_S, result, resultMask;
        pano.convertTo(pano_S, CV_16SC3);
        blender2->feed(pano_S, backgroundMaskFinal, Point(0, 0));
        blender2->feed(allForeground_S, foregroundMaskFinal, Point(0, 0));
        blender2->blend(result, resultMask);
        result.convertTo(result, CV_8U);

//        namedWindow("blend result", WINDOW_GUI_EXPANDED);
        hconcat(fmfOut, result, result);
        imshow("blend result", result);

        string txt2 = "/home/vance/output/" + to_string(k) + "个前景最终结果-" + strMode1 + "+" + strMode2 + ".jpg";
        imwrite(txt2, result);

        waitKey(100);

        timer.stop();
        TIMER(k << "个前景, 算法整体耗时(s): " << timer.getTimeSec());
    }

    waitKey(0);
    return 0;
}
