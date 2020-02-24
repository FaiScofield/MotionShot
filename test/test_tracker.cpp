#include "utility.h"
#include "MotionTracker.h"
#include "OpticalFlower.h"
#include "utility.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/stitching.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/tracking.hpp>
//#include <opencv2/gpu/gpu.hpp>  // gpu::calcOpticalFlowBM()

using namespace std;
using namespace cv;
using namespace ms;

InputType g_type;

int main(int argc, char *argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
               "{type      t|VIDEO|value input type: VIDEO, LASSESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
               "{size      s|5|min blob size}"
               "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
               "{start     a|0|start index for image sequence}"
               "{num       n|0|number to process for image sequence}"
               "{flip      p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
               "{rotate    r|-1|rotate image for type VIDEO, r = cv::RotateFlags(0, 1, 2)}"
               "{write     w|false|write result sequence to a dideo}"
               "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    const String str_type = parser.get<String>("type");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    int start = parser.get<int>("start");
    int num = parser.get<int>("num");
    double scale = parser.get<double>("scale");
    int flip = parser.get<int>("flip");
    int rotate = parser.get<int>("rotate");
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;

    InputType inputType;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
    } else if (str_type == "lasiesta" || str_type == "LASSESTA") {
        inputType = LASIESTA;
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    //// read images
    vector<Mat> vImages, vGTs;
    if (inputType == LASIESTA) {
        ReadImageSequence_lasiesta(str_folder, vImages, vGTs, start, num);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages, start, num);
         scale = 0.15;
    } else if (inputType == VIDEO) {
        ReadImageSequence_video(str_folder, vImages, start, num);
        scale = 0.4;
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);
    cout << " - start = " << max(0, start) << endl;
    cout << " - num = " << vImages.size() << endl;

//    vector<vector<int>> vvImgsPerIter;
//    extractImagesToStitch(vImages, vvImgsPerIter);

    TickMeter timer;
    timer.start();

/*
    /// get pano
    vector<Mat> toStitch(5);
    const int delta = vImages.size() / 5;
    for (int i = 0; i < 5; ++i)
        toStitch[i] = vImages[i * delta];

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
    Stitcher::Status status = stitcher->stitch(toStitch, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    tm.stop();
    double time = tm.getTimeSec() / tm.getCounter();
    cout << " - Time cost in stitching = " << time << "s" << endl;
    cout << " - Image size = " << vImages[0].size() << endl;
    cout << " - Pano size = " << pano.size()  << endl;

    imshow("pano", pano);
    waitKey(1000);
*/

    /*
    OpticalFlower* optFlower = new OpticalFlower();
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        const Mat& frame = vImages[i];
        Mat mask, weight, tmp;
        optFlower->apply(frame, mask);
        if (mask.empty())
            continue;
        weight = optFlower->getWeightMask();
        hconcat(mask, weight, tmp);
        imshow("mask & weight", tmp);
        waitKey(30);
    }
*/

//    Mat pano = vImages.back();

    timer.start();


    /// track
#ifdef USE_OPENCV4
    Ptr<DISOpticalFlow> optFlower = DISOpticalFlow::create();
//    Ptr<FarnebackOpticalFlow> optFlower = FarnebackOpticalFlow::create();
#else
    Ptr<DualTVL1OpticalFlow> optFlower = DualTVL1OpticalFlow::create();
    Ptr<FarnebackOpticalFlow> optFlower2 = FarnebackOpticalFlow::create();
#endif

    vector<Mat> vFlows(vImages.size());
    Mat frameLast, frameCurr;
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        cvtColor(vImages[i], frameCurr, COLOR_BGR2GRAY);
        if (i == 0) {
            frameLast = frameCurr.clone();
            continue;
        }

        Mat flowi;
        if (i == 1) {
            optFlower->calc(frameCurr, frameLast, flowi);
            multiply(flowi, -1, flowi);
            vFlows[0] = flowi;
        }

        optFlower->calc(frameLast, frameCurr, flowi);
        vFlows[i] = flowi;

        int binSize = 5;
        Mat flowColor, flowGray, hist, histGraph;
        //flowToColor(flowi, flowColor);
        drawFlowAndHist(flowi, flowGray, hist, histGraph, 1, binSize);

        // 对flow进行二值化
//        double maxValue, minValue;
//        int minLoc, maxLoc;
//        minMaxLoc(SparseMat(hist), &minValue, &maxValue, &minLoc, &maxLoc);
////        cout << "max loc: " << maxLoc << endl;

//        int th1 = maxLoc;
//        while (--th1 >= 0) {
//            if (hist.at<float>(th1) < 0.2) {
////                cout << "break loc1 = " << th1 << endl;
//                break;
//            }
//        }
//        int th2 = maxLoc;
//        while (++th2 < 256) {
//            if (hist.at<float>(th2) < 0.2) {
////                cout << "break loc2 = " << th2 << endl;
//                break;
//            }
//        }
//        Mat dst = flowGray.clone();
//        for (int y = 0; y < flowGray.rows; ++y) {
//            const uchar* fg_row = flowGray.ptr<uchar>(y);
//            uchar* dst_row = dst.ptr<uchar>(y);
//            for (int x = 0; x < flowGray.cols; ++x) {
//                if (fg_row[x] > th1 * binSize  && fg_row[x] < th2 * binSize);
//                    dst_row[x] = 0;
////                if (fg_row[x] < th1 * binSize || fg_row[x] > th2 * binSize);
////                    dst_row[x] = 255;
//            }
//        }

        Mat dst;
        threshold(flowGray, dst, 80, 255, THRESH_BINARY);
        imshow("dst threshold", dst);
//        waitKey(30);

//        showFlow(flowi, flowColor);
//        imshow("flowColor", flowColor);
//        cvtColor(flowColor, flowGray, COLOR_BGR2GRAY);
//        bitwise_not(flowGray, flowGray);
//        imshow("flowGray", flowGray);

//        drawhistogram(flowGray, hist, Mat(), 5);
//        imshow("flowHist", hist);

//        Mat th1, th2, th3, tmp1, tmp2, tmp3;
//        threshold(flowGray, th1, 100, 255, THRESH_TOZERO);
//        threshold(flowGray, th2, 0, 255, THRESH_OTSU);
//        threshold(flowGray, th3, 0, 255, THRESH_TRIANGLE);
//        bitwise_not(th1, th1);
//        bitwise_not(th2, th2);
//        bitwise_not(th3, th3);
//        vconcat(th1, th2, tmp1);
//        vconcat(tmp1, th3, tmp2);
//        imshow("threshold", tmp2);;

//        waitKey(800);

        frameLast = frameCurr.clone();
    }

    timer.stop();
    TIMER("光流计算耗时(s): " << timer.getTimeSec());
    waitKey(0);
//    auto bs = createBackgroundSubtractorMOG2(500, 100, false);
//    new OpticalFlower();
//    BS_MOG2_CV be(bs.get());
//    MotionTracker tracker;
//    tracker.setBackgroundSubtractor(dynamic_cast<BaseMotionDetector*>(&be));
////    tracker.SetBackgroundSubtractor(dynamic_cast<BaseBackgroundSubtractor*>(&fd));
//    tracker.setMinBlobSize(Size(5, 5));
//    tracker.setPano(pano);

//    Mat frame, gray, mask, mask_gt, output;
//    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
//        Mat& frame = vImages[i];

//        tracker.substractBackground(frame, mask);
//        tracker.detectBlocks();
//        tracker.matchObject();
//        tracker.displayObjects("Objects");
//        tracker.displayDetail("Details");

//        if (waitKey(300) == 27)
//            break;
//    }


    destroyAllWindows();
    return 0;
}
