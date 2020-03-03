#include "utility.h"
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <set>


#define USE_OPENCV_BLENDER 0  //! NOTE OpenCV融合没有考虑时序
#define DEBUG_BLEND_SINGLE_FOREGROUND   0


#if USE_OPENCV_BLENDER
#include <opencv2/stitching.hpp>
#else
#include "ImageBlender/cvBlenders.h"
#include "ImageBlender/cvSeamlessCloning_func.hpp"
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
               "{blender    b|feather|valid blend type: \"feather\", \"multiband\", \"poission\"}"
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
        blender = new ms::cvFeatherBlender(0.1, true);
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

    ReadImageSequence(str_folder + "-gt", "jpg", vGTsColor, beginIdx, endIdx - beginIdx);
    resizeFlipRotateImages(vGTsColor, scale);

    if (vImages.empty() || vGTsColor.empty())
        exit(-1);
    int N = vImages.size();
    assert(N > 3);
    assert(vImages.size() == vGTsColor.size());

    // 掩膜边缘平滑
    const Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_CROSS, Size(7, 7));
    vForegroundMasks.reserve(N);
    int idx = 1;
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m) {
        Mat mask, maskBin, tmpMask1, tmpMask2, tmpMask3;
        cvtColor(m, mask, COLOR_BGR2GRAY);  // 有颜色, 转灰度后不一定是255
        compare(mask, 0, maskBin, CMP_GT);
        mask.setTo(255, maskBin);

        tmpMask1 = mask.clone();   // 形态学操作前
        morphologyEx(mask, mask, MORPH_OPEN, kernel1);// 平滑边界, 去噪点
        morphologyEx(mask, mask, MORPH_CLOSE, kernel2);// 去孔洞
        vForegroundMasks.push_back(mask);

//        resize(tmpMask1, tmpMask1, Size(), 0.5, 0.5);
//        resize(mask, tmpMask2, Size(), 0.5, 0.5);   // 形态学操作后
//        hconcat(tmpMask1, tmpMask2, tmpMask3);
//        imshow("形态学操作前后的MASK", tmpMask3);
//        string txt = "/home/vance/output/ms/" + to_string(idx++) + "前景掩模平滑前后.jpg";
//        imwrite(txt, tmpMask3);

//        waitKey(0);
    });
    destroyAllWindows();

#if DEBUG_BLEND_SINGLE_FOREGROUND
    ms::cvMultiBandBlender* blender1 = new ms::cvMultiBandBlender();
    ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender(0.1, false);
    ms::cvSeamlessCloning* blender3 = new ms::cvSeamlessCloning();

    Mat pano = vImages.front().clone(), panoS;
    pano.convertTo(panoS, CV_16SC3);
    const Rect dstROI(0, 0, pano.cols, pano.rows);

    for (int i = 1; i < N; ++i) {
        Mat foreground, backgroundMask;
        vImages[i].copyTo(foreground, vForegroundMasks[i]);
        Rect foreRectPosition = boundingRect(vForegroundMasks[i]);
        bitwise_not(vForegroundMasks[i], backgroundMask);

        Mat frameS, blendResult1, blendResult2, blendResult3, blendMask1, blendMask2;
        vImages[i].convertTo(frameS, CV_16SC3);

//        Mat panoWithForeInner, foreMaskEdge, foreMaskInner, backMaskInner;
//        erode(vForegroundMasks[i], foreMaskInner, kernel2);
//        foreMaskEdge = vForegroundMasks[i] - foreMaskInner;
//        bitwise_not(foreMaskEdge, backMaskInner);
//        panoWithForeInner = pano.clone();
//        vImages[i].copyTo(panoWithForeInner, foreMaskInner);
//        panoWithForeInner.convertTo(panoWithForeInner, CV_16SC3);
//        imshow("panoWithForeInner", panoWithForeInner);

        // test blending
        blender1->prepare(dstROI);
        blender1->feed(frameS(foreRectPosition), vForegroundMasks[i](foreRectPosition), foreRectPosition.tl());
        blender1->feed(panoS, backgroundMask, Point(0, 0));
//        blender1->feed(frameS(foreRectPosition), foreMaskEdge, foreRectPosition.tl());
//        blender1->feed(panoWithForeInner, backMaskInner, Point(0, 0));
        blender1->blend(blendResult1, blendMask1);
        blendResult1.convertTo(blendResult1, CV_8UC3);

        Mat foreMaskForFeather;
        smoothMaskWeightEdge(vForegroundMasks[i](foreRectPosition), foreMaskForFeather, 16);
        blender2->prepare(dstROI);
        blender2->feed(frameS(foreRectPosition), foreMaskForFeather, foreRectPosition.tl());
        blender2->feed(panoS, backgroundMask, Point(0, 0));
        blender2->blend(blendResult2, blendMask2);
        blendResult2.convertTo(blendResult2, CV_8UC3);

        const Mat kernel3 = getStructuringElement(MORPH_CROSS, Size(15, 15));
        Mat foreMaskForSeamlessClone;
        dilate(vForegroundMasks[i], foreMaskForSeamlessClone, kernel3);
        Point2f center = (foreRectPosition.tl() + foreRectPosition.br()) * 0.5;
        cvSeamlessClone(vImages[i], pano, foreMaskForSeamlessClone, center, blendResult3, NORMAL_CLONE);
//        blender2->normalClone(pano, vImages[i], vForegroundMasks[i], panoMaskToClone, blendResult2, NORMAL_CLONE);

        const string txt1 = "/home/vance/output/ms/第" + to_string(i) + "帧单个前景融合-多频段.jpg";
        const string txt2 = "/home/vance/output/ms/第" + to_string(i) + "帧单个前景融合-羽化.jpg";
        const string txt3 = "/home/vance/output/ms/第" + to_string(i) + "帧单个前景融合-泊松.jpg";
        imwrite(txt1, blendResult1);
        imwrite(txt2, blendResult2);
        imwrite(txt3, blendResult3);
        imshow("blendResult1(multiband)", blendResult1);
        imshow("blendResult2(feather)", blendResult2);
        imshow("blendResult2(poisson)", blendResult3);
        waitKey(30);
    }

    exit(0);
#endif

    /// calc the result in different gaps
    int maxFores = 8, minFores = 3;  // 前景最多存在9个, 最少3个
    vector<Mat> vImgsToProcess = vImages;
//    vector<int> vIdxToProcess{0, 1, 2, 3, 4, 5, 6, 7, 8};
//    vector<vector<int>> vvIdxPerIter;
//    vvIdxPerIter.emplace_back(vector<int>{0, 3, 6});           // 3
//    vvIdxPerIter.emplace_back(vector<int>{0, 3, 6, 8});
//    vvIdxPerIter.emplace_back(vector<int>{0, 2, 4, 6, 8});
//    vvIdxPerIter.emplace_back(vector<int>{0, 1, 3, 4, 6, 7});  // 6
//    vvIdxPerIter.emplace_back(vector<int>{0, 1, 3, 4, 6, 7, 8});
//    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
//    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8});   // 9

    vector<int> vIdxToProcess{0, 1, 2, 3, 4, 5, 6, 7};
    vector<vector<int>> vvIdxPerIter;
    vvIdxPerIter.emplace_back(vector<int>{0, 3, 6});           // 3
    vvIdxPerIter.emplace_back(vector<int>{0, 2, 5, 7});
    vvIdxPerIter.emplace_back(vector<int>{0, 2, 4, 6, 7});
    vvIdxPerIter.emplace_back(vector<int>{0, 2, 3, 4, 5, 7});  // 6
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 4, 5, 6, 7});
    vvIdxPerIter.emplace_back(vector<int>{0, 1, 2, 3, 4, 5, 6, 7});

    timer.stop();
    TIMER("系统初始化耗时(s): " << timer.getTimeSec());

    /// main loop
    for (int k = minFores; k <= maxFores; ++k) {  // k为前景数量
        timer.start();

        // 1.前景拼接融合
        const vector<int>& vIdxThisIter = vvIdxPerIter[k - minFores];
        const Mat pano = vImgsToProcess.front().clone();
        const Rect disRoi(0, 0, pano.cols, pano.rows);

        blender->prepare(disRoi);
        for (size_t j = 0, jend = vIdxThisIter.size(); j < jend; ++j) {
            const int& imgIdx = vIdxThisIter[j];
            const Mat& frame = vImages[imgIdx];
            const Mat& foreMask = vForegroundMasks[imgIdx];

            // blender
            Mat frame_S, foreMaskSmooth;
            frame.convertTo(frame_S, CV_16SC3);
            smoothMaskWeightEdge(foreMask, foreMaskSmooth, 16);  // 过渡边缘
            blender->feed(frame_S, /*foreMask*/foreMaskSmooth, Point(0, 0));
        }

        Mat allForeground, allForeground_S, allForegroundMask;
        blender->blend(allForeground_S, allForegroundMask);
        allForeground_S.convertTo(allForeground, CV_8UC3);

        imshow("allForeground", allForeground);
        imshow("allForeground mask", allForegroundMask);
        string txt1 = "/home/vance/output/ms/" + to_string(k) + "个前景拼接融合结果-" + strMode1 + ".jpg";
        imwrite(txt1, allForeground);
//        waitKey(0);
//        continue;

        // 2.前背景融合. 把pano和前景对应的mask区域的稍微缩收/扩张, 设置平滑的权重过渡, 然后再融合.
        //! 不能用多频段融合.
#if USE_OPENCV_BLENDER
        detail::FeatherBlender* blender2 = new detail::FeatherBlender();
#else
//        ms::cvSeamlessCloning* blender2 = new ms::cvSeamlessCloning();
        ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender(0.1, false);
        const string strMode2 = "羽化";
#endif

        blender2->prepare(disRoi);

        Mat foregroundMaskFinal, backgroundMaskFinal, maskFinal;
        smoothMaskWeightEdge(allForegroundMask, foregroundMaskFinal, 10);  // 过渡边缘
        bitwise_not(foregroundMaskFinal, backgroundMaskFinal);
        hconcat(foregroundMaskFinal, backgroundMaskFinal, maskFinal);
        imshow("foreground & background maskFinal", maskFinal);

        Mat fmf, fmfOut;
        cvtColor(foregroundMaskFinal, fmf, COLOR_GRAY2BGR);
        hconcat(allForeground, fmf, fmfOut);
//        string txt0 = "/home/vance/output/前景与掩模-" + strMode1 + "-" + to_string(k) + ".jpg";
//        imwrite(txt0, fmfOut);

        Mat pano_S, result, resultMask;
        pano.convertTo(pano_S, CV_16SC3);
        blender2->feed(pano_S, backgroundMaskFinal, Point(0, 0));
        blender2->feed(allForeground_S, foregroundMaskFinal, Point(0, 0));
        blender2->blend(result, resultMask);
        result.convertTo(result, CV_8U);

//        namedWindow("blend result", WINDOW_GUI_EXPANDED);
//        hconcat(fmfOut, result, result);
        imshow("blend result", result);

        string txt2 = "/home/vance/output/ms/" + to_string(k) + "个前景最终结果-" + strMode1 + "+" + strMode2 + ".jpg";
        imwrite(txt2, result);

        waitKey(300);

        timer.stop();
        TIMER(k << "个前景, 算法整体耗时(s): " << timer.getTimeSec());
    }

    waitKey(0);
    return 0;
}
