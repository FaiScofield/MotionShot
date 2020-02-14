#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/photo.hpp>
#include "utility.h"
#include "ImageBlender/PoissionBlender.h"


using namespace std;
using namespace cv;
using namespace ms;

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

//    detail::Blender* blender = nullptr;
//    if (str_blender == "feather") {
//        blender = dynamic_cast<detail::Blender*>(new detail::FeatherBlender());
//    } else if (str_blender == "multiband") {
//        blender = dynamic_cast<detail::Blender*>(new detail::MultiBandBlender());
//    } else if (str_blender == "poission") {
//        blender = dynamic_cast<detail::Blender*>(new ms::PoissionBlender()); // TODO
//    } else {
//        cerr << "[Error] Unknown blender type for " << str_blender << endl;
//        exit(-1);
//    }
    PoissionBlender* blender = new PoissionBlender();

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

    for (int d = 0; d < vDelta.size(); ++d) {
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
//            waitKey(0);
        }

//        overlapRoi();

        // blend with blobs
        const Mat pano = vImages[N - N % k].clone();
        Mat foreground = Mat::zeros(pano.size(), CV_8UC3);
        Mat foregroundMask = Mat::zeros(pano.size(), CV_8UC1);
        for (int j = 0; j < vBlobs.size(); ++j) {
            if (!vValid[j])
                continue;

            const Rect& blob = vBlobs[j];
            int imgIdx = j * k;
            Mat frame_s, dst_mask;

//            vImages[imgIdx].convertTo(frame_s, CV_16SC3);
            Point center = (blob.tl() + blob.br()) * 0.5;
            blender->feed(pano, Mat(), center);
            blender->blend(vImages[imgIdx], vMasks[imgIdx]);
            Mat result = blender->getResult();
            imshow("blend result", result);


            bitwise_and(foregroundMask, vMasks[imgIdx], foregroundMask);
            result.rowRange(blob.tl().y, blob.br().y).colRange(blob.tl().x, blob.br().x).
                    copyTo(foreground.rowRange(blob.tl().y, blob.br().y).colRange(blob.tl().x, blob.br().x));
            imshow("foreground", foreground);

            waitKey(0);
        }
        string filename = "/home/vance/output/foreground-d" + to_string(k) + ".jpg";
        imwrite(filename, foreground);
    }
//    blender->prepare(Rect(0, 0, src.cols, src.rows));
//    blender->feed(src_s, mask, Point(0, 0));
//    blender->feed(targ_s, mask, Point(0, 0));
//    blender->blend(result_s, result_mask);
//    Mat result, result2;
//    result_s.convertTo(result, CV_8U);
//    result_mask.convertTo(result2, CV_8UC3);

//    Mat result;
//    // NORMAL_CLONE, MIXED_CLONE, 黑白单色MONOCHROME_TRANSFER
//    seamlessClone(src, targ, mask, placed, result, NORMAL_CLONE);

//    imshow("source image", src);
//    imshow("target image", targ);
//    imshow("blending result", result);
//    imwrite("/home/vance/output/blending_result.bmp", result);

    waitKey(0);
    return 0;
}
