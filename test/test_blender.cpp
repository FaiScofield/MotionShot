#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/photo.hpp>
#include "utility.h"
//#include "ImageBlender/PoissionBlender.h"
#include "ImageBlender/cvBlenders.h"

using namespace std;
using namespace cv;
using namespace ms;

enum BlenderType { NO, FEATHER, MULTI_BAND };

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder     f| |LASISESTA dataset folder}"
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

    detail::Blender* blender = nullptr;
//    ms::cvBlender* blender = nullptr;
    BlenderType blenderType;
    if (str_blender == "feather") {
        blender = dynamic_cast<detail::Blender*>(new detail::FeatherBlender());
//        blender = new ms::cvFeatherBlender();
        blenderType = FEATHER;
    } else if (str_blender == "multiband") {
        blender = dynamic_cast<detail::Blender*>(new detail::MultiBandBlender(false, 2, CV_32F));
//        blender = new ms::cvMultiBandBlender();
        blenderType = MULTI_BAND;
    } /*else if (str_blender == "poission") {
        blender = dynamic_cast<detail::Blender*>(new ms::PoissionBlender()); // TODO
    }*/ else {
        cerr << "[Error] Unknown blender type for " << str_blender << endl;
        exit(-1);
    }
//    PoissionBlender* blender = new PoissionBlender();

    int start = parser.get<int>("start");
    int end = parser.get<int>("end");
    int delta = parser.get<int>("delta");
    cout << " - start index = " << start << endl;
    cout << " - end index = " << end << endl;
    cout << " - delta = " << delta << endl;

    vector<Mat> vImages, vGTsColor, vMasks;
    ReadImageSequence_lasisesta(str_folder, vImages, vGTsColor, start, end - start);
    if (vImages.empty() || vGTsColor.empty())
        exit(-1);
    int N = vImages.size();
    assert((delta == 0) || (delta > 1 && delta < N));
    assert(vImages.size() == vGTsColor.size());

    vMasks.reserve(N);
    for_each(vGTsColor.begin(), vGTsColor.end(), [&](Mat& m){
        Mat mask;
        cvtColor(m, mask, COLOR_BGR2GRAY);
        vMasks.push_back(mask);
    });

    timer.stop();
    cout << "[Timer] Time cost in system initilization: " << timer.getTimeSec() / timer.getCounter() << endl;

    /// get foreground and blend
    vector<int> vDelta;
    if (delta == 0) {
        for (int d = 3; d < 11; ++d)
            vDelta.push_back(N / d);
    } else {
        vDelta.push_back(delta);
    }

    for (size_t d = 0; d < vDelta.size(); ++d) {
        // get blobs
        const int k = vDelta[d];
        vector<Rect> vBlobs;
        vector<bool> vValid;
        for (int i = 0; i < N; ++i) {
            if (i % k != 0)
                continue;

            vector<vector<Point>> contours;
            findContours(vMasks[i], contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            if (contours.empty()) {
                vBlobs.push_back(Rect());
                vValid.push_back(0);
                continue;
            }
            Mat frameOut = vImages[i].clone();
            drawContours(frameOut, contours, -1, Scalar(0, 255, 0), 2);

            Rect blob = boundingRect(contours[0]);
            vBlobs.push_back(blob);
            vValid.push_back(1);
            rectangle(frameOut, blob, Scalar(0, 0, 255), 2);
            imshow("contour & blob", frameOut);
            waitKey(100);
        }
        destroyAllWindows();

        /// 前景拼接融合
        const Mat pano = vImages[N - N % k].clone();
        Mat panos, panosMask;
        pano.convertTo(panos, CV_16SC3);
        panosMask.create(pano.size(), CV_8UC1);
        panosMask.setTo(255);

        Rect dis_roi(0, 0, pano.cols, pano.rows);
//        blender->prepare(dis_roi);
//        vector<Point> corners(vBlobs.size(), Point(0, 0));
//        vector<Size> sizes(vBlobs.size(), pano.size());;
//        blender->prepare(corners, sizes);

        Mat foreground_f, foregroundMask, foreground;
        blender->prepare(dis_roi);
        vector<Mat> maskWeight;
        for (size_t j = 0; j < vBlobs.size(); ++j) {
            if (!vValid[j])
                continue;

            const Rect& blob = vBlobs[j];
            const int imgIdx = j * k;

            // blender
            Mat imgInput;
            vImages[imgIdx].convertTo(imgInput, CV_16SC3);
            Mat mask = Mat::zeros(pano.size(), CV_8U);
            mask(blob).setTo(255);

            blender->feed(imgInput, mask, Point(0,0));
            if (j > 0) {
                Mat forej, foreMaskj, forejU;
                blender->blend(forej, foreMaskj);
                forej.convertTo(forejU, CV_8UC3);
                imshow("forej", forejU);

                if (j == vBlobs.size() - 1) {
                    foreground_f = forej.clone();
                    foregroundMask = foreMaskj.clone();
                    foreground_f.convertTo(foreground, CV_8UC3);
                } else {
                    blender->prepare(dis_roi);
                    blender->feed(forej, foreMaskj, Point(0,0));
                }
            }

            //! TODO
//            Mat mw;
//            blender->calcMaskWeight(mw);
//            maskWeight.push_back(mw);

            waitKey(2000);
        }
        destroyAllWindows();

//        Mat foreground_f, foregroundMask, foreground;
//        blender->blend(foreground_f, foregroundMask);
//        // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
//        // so convert it to avoid user confusing
//        foreground_f.convertTo(foreground, CV_8U);

        Mat tmp1, tmp2;
        cvtColor(foregroundMask, tmp1, COLOR_GRAY2BGR);
        hconcat(foreground, tmp1, tmp2);
        imshow("foreground and mask", tmp2);
        string mode;
        if (blenderType == FEATHER)
            mode = "羽化";
        else if (blenderType == MULTI_BAND)
            mode = "多频带";
        string txt1 = "/home/vance/output/前景时序融合-" + mode + "-" +to_string(k) + ".jpg";
        imwrite(txt1, foreground);

//        /// 前背景融合
////        ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender();
////        ms::cvMultiBandBlender* blender2 = new ms::cvMultiBandBlender(false, 2, CV_32F);
//        detail::FeatherBlender* blender2 = new detail::FeatherBlender();
//        BlenderType blenderType2 = FEATHER;
////        detail::MultiBandBlender* blender2 = new detail::MultiBandBlender(false, 2, CV_32F);
////        BlenderType blenderType2 = MULTI_BAND;
//        blender2->prepare(dis_roi);
//        Mat panoMask(pano.size(), CV_8U);
//        panoMask.setTo(255);
//        Mat panof;
//        pano.convertTo(panof, CV_16SC3);
//        blender2->feed(panof, panoMask, Point(0,0));
//        blender2->feed(foreground_f, foregroundMask, Point(0,0));
//        Mat result, resultMask;
//        blender2->blend(result, resultMask);
//        result.convertTo(result, CV_8U);
//        imshow("blend result", result);
//        string mode2;
//        if (blenderType2 == FEATHER)
//            mode2 = "羽化";
//        else if (blenderType2 == MULTI_BAND)
//            mode2 = "多频带";
//        string txt2 = "/home/vance/output/前背景融合-" + mode + "+" + mode2 + "-" + to_string(k) + ".jpg";
//        imwrite(txt2, result);

//        // diff
//        Mat diffFore, hist;
//        Mat foreGray, panoGray;
//        cvtColor(foreground, foreGray, COLOR_BGR2GRAY);
//        cvtColor(pano, panoGray, COLOR_BGR2GRAY);
////        bitwise_and(panoGray, foregroundMask, panoGray);
////        absdiff(foreGray, panoGray, diffFore);
////        drawhistogram(diffFore, hist);
////        Mat tmpGray, tmp3;
////        vconcat(foreGray, panoGray, tmpGray);
////        vconcat(tmpGray, diffFore, tmp3);
////        imshow("fore and back ground diff", tmp3);
////        imshow("diffFore hist", hist);
////        double maxV, minV;
////        minMaxLoc(diffFore, &minV, &maxV);
////        cout << "min / max diff value = " << minV << " / " << maxV << endl;

////        Mat panoHist;
////        drawhistogram(panoGray, panoHist);
////        imshow("panoHist", panoHist);

//        Mat resultPlus = pano.clone();
//        resultPlus.setTo(Vec3b(0,0,0), foregroundMask);
//        resultPlus += foreground;
//        imshow("resultPlus", resultPlus);
//        imwrite(string("/home/vance/output/前背景融合-" + mode + "+叠加-" + to_string(k) + ".jpg"), resultPlus);
    }

    waitKey(0);
    return 0;
}
