#include "utility.h"
#include "ImageStitcher/ImageStitcher.h"
#include <opencv2/stitching.hpp>

using namespace ms;
using namespace std;
using namespace cv;
namespace cvd = cv::detail;

const int delta = 3;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
        "{type      t|VIDEO|value input type: LASIESTA, HUAWEI, SEQUENCE}"
        "{folder    f| |data folder or video file for type LASIESTA/HUAWEI/SEQUENCE/VIDEO}"
        "{suffix    x|jpg|image suffix for SEQUENCE}"
        "{begin     a|0|start index for image sequence}"
        "{end       e|-1|end index for image sequence}"
        "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_folder = parser.get<String>("folder");
    const int beginIdx = parser.get<int>("begin");
    const int endIdx = parser.get<int>("end");
    const int num = endIdx - beginIdx + 1;
    INFO(" - type = " << str_type);
    INFO(" - folder = " << str_folder);
    INFO(" - beginIdx = " << beginIdx);
    INFO(" - endIdx = " << endIdx);

    vector<Mat> vImages;
    if (str_type == "video" || str_type == "VIDEO") {
        ReadImageSequence_video(str_folder, vImages, beginIdx, num);
    }  else if (str_type == "lasiesta" || str_type == "LASIESTA") {
        vector<Mat> gts;
        ReadImageSequence_lasiesta(str_folder, vImages, gts, beginIdx, num);
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        ReadImageSequence_huawei(str_folder, vImages, beginIdx, num);
    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
        const String str_suffix = parser.get<String>("suffix");
        INFO(" - suffix = " << str_suffix);
        ReadImageSequence(str_folder, str_suffix, vImages, beginIdx, num);
    } else {
        ERROR("[Error] Unknown input type for " << str_type);
        return -1;
    }
    INFO(" - num of input images = " << vImages.size());
    if (num < 2) {
        ERROR("Too less images input!");
        exit(-1);
    }

    /// start stitching
    INFO(endl << "\t Stitching... This will take a while...");

    TickMeter timer;
    timer.start();

    /// 1. opencv stitcher
    Ptr<cv::Stitcher> stitcher1 = cv::Stitcher::create(Stitcher::PANORAMA);
    stitcher1->setWarper(makePtr<PlaneWarper>());
    stitcher1->setWaveCorrection(false);
//    stitcher1->setExposureCompensator(makePtr<cvd::NoExposureCompensator>());
//    stitcher1->setSeamFinder(makePtr<cvd::NoSeamFinder>());

    Mat pano1;
    Stitcher::Status status = stitcher1->stitch(vImages, pano1);
    if (status != Stitcher::OK) {
        ERROR("Can't stitch images (cv), error code = " << status);
        return -1;
    }

    timer.stop();
    TIMER(" - Time cost in stitching 1 = " << timer.getTimeSec() << "s");
    INFO(" - Image size = " << vImages[0].size());
    INFO(" - Pano size = " << pano1.size());

    NamedLargeWindow("Result Pano CV");
    imshow("Result Pano CV", pano1);
    imwrite("/home/vance/output/result_pano_cv.jpg", pano1);
    INFO("Saving result1 to /home/vance/output/result_pano_cv.jpg");
    timer.start();

    /// 2. custom stitcher
    Ptr<ImageStitcher> stitcher2 = ImageStitcher::create(ImageStitcher::ORB, ImageStitcher::BF);
    stitcher2->setScales(0.25, 0.1, 1.);
    stitcher2->setWaveCorrection(false);
//    stitcher2->setRistResolutions(0.6, 0.1, 0);

    Mat pano2;
    ImageStitcher::Status status2 = stitcher2->stitch(vImages, pano2);
    if (status2 != ImageStitcher::OK) {
        ERROR("Can't stitch2 images (custom), error code = " << status);
        return -1;
    }

    timer.stop();
    TIMER(" - Time cost in stitching 2 = " << timer.getTimeSec() << "s");

    NamedLargeWindow("Result Pano Custom");
    imshow("Result Pano Custom", pano2);
    imwrite("/home/vance/output/result_pano_custom.jpg", pano2);
    INFO("Saving result2 to /home/vance/output/result_pano_custom.jpg");

    waitKey(0);

    return 0;
}
