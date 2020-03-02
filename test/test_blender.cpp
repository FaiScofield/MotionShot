#include "utility.h"
#include <iostream>
#include <map>
#include <set>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo.hpp>
//#include <opencv2/stitching.hpp>
//#include "ImageBlender/PoissionBlender.h"
#include "ImageBlender/cvBlenders.h"
#include "MotionDetector/OpticalFlower.h"


#define USE_FLOW_WEIGHT 0
#define GET_FOREGROUND_FROM_GT  1
#define AVE_BACKGROUND  0

using namespace std;
using namespace cv;
using namespace ms;

enum BlenderType { NO, FEATHER, MULTI_BAND, POISSON_BAND };

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder     f| |LASIESTA dataset folder}"
               "{blender    b|multiband|valid blend type: \"feather\", \"multiband\", \"poission\"}"
               "{delta      d|10|interval from frame to frame for foreground. If delta = 0, output "
               "5~15 images}"
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

//    detail::Blender* blender = nullptr;
        ms::cvBlender* blender = nullptr;
    BlenderType blenderType;
    if (str_blender == "feather") {
//        blender = dynamic_cast<detail::Blender*>(new detail::FeatherBlender());
        blender = new ms::cvFeatherBlender(1.0, true);
        blenderType = FEATHER;
    } else if (str_blender == "multiband") {
//        blender = dynamic_cast<detail::Blender*>(new detail::MultiBandBlender(false, 3, CV_32F));
        blender = new ms::cvMultiBandBlender(false, 5, CV_32F);
        blenderType = MULTI_BAND;
    } /*else if (str_blender == "poission") {
        blender = dynamic_cast<detail::Blender*>(new ms::PoissionBlender()); // TODO
    }*/
    else {
        cerr << "[Error] Unknown blender type for " << str_blender << endl;
        exit(-1);
    }
    //    PoissionBlender* blender = new PoissionBlender();
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

    vector<Mat> vImages, vGTsColor, vMasks;
    ReadImageSequence_lasiesta(str_folder, vImages, vGTsColor, start, end - start);
    if (vImages.empty() || vGTsColor.empty())
        exit(-1);
    int N = vImages.size();
    assert(N > 3);
    assert((delta == 0) || (delta > 1 && delta < N));
    assert(vImages.size() == vGTsColor.size());

    vMasks.reserve(N);
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m) {
        Mat mask;
        cvtColor(m, mask, COLOR_BGR2GRAY);
        vMasks.push_back(mask);
    });

    /// calc the result in different gaps
    int maxFores = 8, minFores = 3;  // 前景最多存在8个, 最少3个
    vector<Mat> vImgsToProcess;
    vector<int> vIdxToProcess;
    vector<vector<int>> vvIdxPerIter;
    extractImagesToStitch(vImages, vImgsToProcess, vIdxToProcess, vvIdxPerIter, minFores, maxFores);

    timer.stop();
    TIMER("系统初始化耗时(s): " << timer.getTimeSec());

    //! TODO  计算出来的光流对应的图像和序号还没弄好!!!
#if USE_FLOW_WEIGHT
    timer.start();

    // get flow mask
    cout << "计算光流中, 需要一段时间... " << endl;
    OpticalFlower* optFlower = new OpticalFlower;
    vector<Mat> vFlows;
    std::map<int, Mat> mFlows;
    optFlower->apply(vImgsToProcess, vFlows);
    for (size_t i = 0; i < vFlows.size(); ++i)
        mFlows.emplace(vIdxToProcess[i], vFlows[i]);

    timer.stop();
    TIMER("光流计算耗时(s): " << timer.getTimeSec() / timer.getCounter());
#endif

    const int th_minBlobArea = 0.05 * vImages[0].size().area();
    Ptr<BackgroundSubtractorMOG2> subtractor = createBackgroundSubtractorMOG2(vImgsToProcess.size(), 16.0, false);

#if AVE_BACKGROUND
    // 计算平均背景
    Mat aveBackground = Mat::zeros(vImages[0].size(), CV_32FC3);
    for (size_t i = 0; i < vImgsToProcess.size(); ++i) {
        Mat frameF;
        vImgsToProcess[i].convertTo(frameF, CV_32FC3);
        //add(aveBackground, frameF, aveBackground);
        accumulate(frameF, aveBackground);
    }
    multiply(aveBackground, 1./vImgsToProcess.size(), aveBackground);
    aveBackground.convertTo(aveBackground, CV_8UC3);

    imshow("aveBackground", aveBackground);
    Mat maskAve;
    subtractor->apply(aveBackground, maskAve);
    imshow("maskAve", maskAve);
//    waitKey(0);
#endif

    for (int k = minFores; k <= maxFores; ++k) { // k为前景数量
        timer.start();

        // 1.获取前景区域
        const vector<int>& vIdxThisIter = vvIdxPerIter[k - minFores];
        vector<Rect> vBlobs;
        vector<bool> vValid;
        vector<Mat> vForeMaskPrecise;
        for (size_t i = 0; i < vIdxThisIter.size(); ++i) {
            const int& imgIdx = vIdxThisIter[i];
//            const auto it = std::find(vIdxToProcess.begin(), vIdxToProcess.end(), imgIdx);
//            const int idx = std::distance(vIdxToProcess.begin(), it);
//            const Mat& frame = vImgsToProcess[idx];
            const Mat& frame = vImages[imgIdx];
            vector<vector<Point>> contours;
#if GET_FOREGROUND_FROM_GT
            findContours(vMasks[imgIdx], contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                vBlobs.push_back(Rect());
                vValid.push_back(0);
                continue;
            }
            Mat frameOut = vImages[imgIdx].clone();
            drawContours(frameOut, contours, -1, Scalar(0, 255, 0), 2);

            Rect blob = boundingRect(contours[0]);
            vBlobs.push_back(blob);
            vValid.push_back(1);
            rectangle(frameOut, blob, Scalar(0, 0, 255), 2);
//            imshow("contour & blob", frameOut);
#else   // get foreground form BackgroundSubtractor
            Mat diff;
            subtractor->apply(frame, diff, 0);
            Mat kernel1 = getStructuringElement(MORPH_RECT, Size(7, 7));
            Mat kernel2 = getStructuringElement(MORPH_RECT, Size(5, 5));
            morphologyEx(diff, diff, MORPH_OPEN, kernel1);
            morphologyEx(diff, diff, MORPH_CLOSE, kernel2);

            findContours(diff, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                vBlobs.push_back(Rect());
                vValid.push_back(0);
                continue;
            }
            Mat frameOut = vImages[imgIdx].clone();
            drawContours(frameOut, contours, -1, Scalar(0, 255, 0), 2);

            int maxArea = 0, maxAreaIdx;
            Rect maxBlob;
            for (size_t j = 0; j < contours.size(); ++j) {
                Rect blob = boundingRect(contours[j]);
                if (blob.area() < th_minBlobArea)
                    continue;
                if (blob.area() > maxArea) {
                    maxAreaIdx = j;
                    maxBlob = blob;
                }
            }
            vBlobs.push_back(maxBlob);
            vValid.push_back(1);

            Mat maxBlobMask = Mat::zeros(frame.size(), CV_8UC1);
            maxBlobMask(maxBlob).setTo(255);
            bitwise_and(maxBlobMask, diff, maxBlobMask);
            vForeMaskPrecise.push_back(maxBlobMask);
            rectangle(frameOut, maxBlob, Scalar(0, 0, 255), 2);
            imshow("contour & blob", frameOut);
            imshow("maxBlobMask", maxBlobMask);
#endif
            // 得到稍精确的前景掩模
            Mat diff, diffColor;
            subtractor->apply(frame, diff, -1);
            Mat kernel1 = getStructuringElement(MORPH_RECT, Size(7, 7));
            Mat kernel2 = getStructuringElement(MORPH_RECT, Size(9, 9));
            morphologyEx(diff, diff, MORPH_OPEN, kernel1);
            morphologyEx(diff, diff, MORPH_CLOSE, kernel2);
            cvtColor(diff, diffColor, COLOR_GRAY2BGR);

            Mat maxBlobMask = Mat::zeros(frame.size(), CV_8UC1);
            maxBlobMask(blob).setTo(255);
            bitwise_and(maxBlobMask, diff, maxBlobMask);

            Mat tmp1 = frame.clone(), tmp2;
            bitwise_not(maxBlobMask, tmp2);
            tmp1.setTo(0, tmp2);
            hconcat(frameOut, diffColor, tmp2);
//            imshow("contour + blob & diff & ForeMaskPrecise", tmp2);

            // 腐蚀contours, 去除边缘白边
            Mat kernel3 = getStructuringElement(MORPH_RECT, Size(5, 5));
            erode(maxBlobMask, maxBlobMask, kernel3);
            Mat tmp3 = frame.clone(), tmp4;
            bitwise_not(maxBlobMask, tmp4);
            tmp3.setTo(0, tmp4);
            hconcat(tmp2, tmp3, tmp3);
            imshow("contour & blob & ForeMaskPrecise erode", tmp3);

            // 过渡边缘
            smoothMaskWeightEdge(maxBlobMask, maxBlobMask, 3);
            vForeMaskPrecise.push_back(maxBlobMask);

            waitKey(2000);

#if AVE_BACKGROUND
#else
            // 利用最后一帧的diff更新第一帧的精确mask
            if (i == vIdxThisIter.size() - 1) {
                Mat blobMask = Mat::zeros(frame.size(), CV_8UC1);
                blobMask(vBlobs[0]).setTo(255);
                bitwise_and(blobMask, diff, blobMask);
                erode(blobMask, blobMask, kernel3);
                smoothMaskWeightEdge(blobMask, blobMask, 5);
                vForeMaskPrecise[0] = blobMask;
//                imshow("first blobMask", blobMask);
//                waitKey(0);
            }
#endif
        }
        destroyAllWindows();

        // 2.前景拼接融合
        const Mat pano = vImgsToProcess.back().clone();
        const Rect dis_roi(0, 0, pano.cols, pano.rows);
        blender->prepare(dis_roi);
        for (size_t j = 0; j < vBlobs.size(); ++j) {
            if (!vValid[j])
                continue;

            const Rect& blob = vBlobs[j];
            const size_t& imgIdx = vIdxThisIter[j];
            const Mat& blobMask = vForeMaskPrecise[j];

            // blender
            Mat imgInput;
            vImages[imgIdx].convertTo(imgInput, CV_16SC3);
            Mat mask = Mat::zeros(pano.size(), CV_8UC1);
            mask(blob).setTo(20);
            add(mask, blobMask, mask);
            threshold(mask, mask, 255, 255, THRESH_TRUNC);

//            Mat tmp1, tmp2;
//            cvtColor(mask, tmp1, COLOR_GRAY2BGR);
//            hconcat(vImages[imgIdx], tmp1, tmp2);
//            imshow("imgInput & mask", tmp2);


#if USE_FLOW_WEIGHT
            Mat tmp, tmp2;
            hconcat(mask, mFlows[imgIdx], tmp);
            imshow("mask & flowmask", tmp);

            Mat maskPlusFlow1, maskPlusFlow2, thMask1, thMask2;
            bitwise_and(mFlows[imgIdx], mask, maskPlusFlow1);
            normalize(maskPlusFlow1, maskPlusFlow2, 0, 255, NORM_MINMAX);
            threshold(maskPlusFlow2, thMask1, 80, 255, THRESH_BINARY);
            bitwise_not(thMask1, thMask2);
            bitwise_and(maskPlusFlow2, thMask2, maskPlusFlow2);
            add(thMask1, maskPlusFlow2, maskPlusFlow2);
            hconcat(maskPlusFlow1, maskPlusFlow2, tmp2);
            imshow("Enhenced maskFlow weight", tmp2);
            string txt = "EnhencedMaskFlow-" + to_string(j) + ".jpg";
            imwrite(txt, tmp2);
            waitKey(1000);

            blender->feed(imgInput, maskPlusFlow2, Point(0, 0));
#else
            blender->feed(imgInput, /*blobMask*/mask, Point(0, 0));
#endif
//            waitKey(0);
        }
//        waitKey(0);
        destroyAllWindows();

        Mat foreground_f, foregroundMask, foreground;
        blender->blend(foreground_f, foregroundMask);
        foreground_f.convertTo(foreground, CV_8U);
//        imshow("foregroundMask", foregroundMask);

        Mat tmp1, tmp2;
        cvtColor(foregroundMask, tmp1, COLOR_GRAY2BGR);
        hconcat(foreground, tmp1, tmp2);
//        imshow("foreground and mask", tmp2);
//        string txt1 = "/home/vance/output/前景融合-" + strMode1 + "-" +to_string(k) + ".jpg";
//        imwrite(txt1, tmp2);
//        waitKey(0);

        // 3.前背景融合
        //! TODO 把pano和前景对应的mask区域的稍微缩收/扩张, 设置具体的权重值, 然后再融合.
        //! (不能用羽化, 羽化不能自定义权重) 目前看多频段的效果还不如羽化!
//        detail::FeatherBlender* blender2 = new detail::FeatherBlender();
        BlenderType blenderType2 = FEATHER;
        ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender();

        blender2->prepare(dis_roi);

        Mat foregroundMask_final, backgroundMast_final, maskFinal;
        smoothMaskWeightEdge(foregroundMask, foregroundMask_final, 7); // 过渡边缘
        bitwise_not(foregroundMask_final, backgroundMast_final);

        hconcat(foregroundMask_final, backgroundMast_final, maskFinal);
        Mat fmf, fmfOut;
        cvtColor(foregroundMask_final, fmf, COLOR_GRAY2BGR);
        hconcat(foreground, fmf, fmfOut);
//        string txt0 = "/home/vance/output/前景与掩模-" + strMode1 + "-" + to_string(k) + ".jpg";
//        imwrite(txt0, fmfOut);

        Mat panof;
        pano.convertTo(panof, CV_16SC3);
        blender2->feed(panof, backgroundMast_final, Point(0, 0));
        blender2->feed(foreground_f, foregroundMask_final, Point(0, 0));
        Mat result, resultMask;
        blender2->blend(result, resultMask);
        result.convertTo(result, CV_8U);

        hconcat(fmfOut, result, result);
        imshow("blend result", result);


        string strMode2;
        if (blenderType2 == FEATHER)
            strMode2 = "羽化";
        else if (blenderType2 == MULTI_BAND)
            strMode2 = "多频带";
        string txt2 = "/home/vance/output/前背景融合-" + strMode1 + "+" + strMode2 + "-" + to_string(k) + ".jpg";
        imwrite(txt2, result);

        timer.stop();
        TIMER("间隔" << delta << "帧, Blending 耗时(s): " << timer.getTimeSec() / timer.getCounter());
    }

    waitKey(0);
    return 0;
}
