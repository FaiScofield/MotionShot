#include "utility.h"
#include "MotionTracker.h"
#include "BS_MOG2_CV.h"
#include "OpticalFlower.h"
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
               "{type      t|VIDEO|value input type: VIDEO, LASISESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASISESTA/HUAWEI/VIDEO}"
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
    } else if (str_type == "lasisesta" || str_type == "LASISESTA") {
        inputType = LASISESTA;
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    //// read images
    vector<Mat> vImages, vGTs;
    if (inputType == LASISESTA) {
        ReadImageSequence_lasisesta(str_folder, vImages, vGTs, start, num);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages, start, num);
         scale = 0.15;
    } else if (inputType == VIDEO) {
        ReadImageSequence_video(str_folder, vImages, start, num);
        scale = 0.4;
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);
    cout << " - start = " << start << endl;
    cout << " - num = " << num << endl;

    TickMeter tm;
    tm.start();

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

    /// track
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
