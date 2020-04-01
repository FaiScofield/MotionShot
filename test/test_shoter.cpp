#include "utility.h"
#include "MotionShoter/MotionShoter.h"

using namespace ms;
using namespace std;
using namespace cv;

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
    INFO(" - resolution of images  = " << vImages[0].size());
    if (num < 2) {
        ERROR("Too less images input!");
        exit(-1);
    }

    /// mainloop
    INFO(endl << "\t Shoting... This will take a while...");

    TickMeter timer;
    timer.start();

    MotionShot::Status status = MotionShot::OK;
    Ptr<MotionShot> motionShoter = makePtr<MotionShot>();

    status = motionShoter->setInputs(vImages);
    if (status != MotionShot::OK) {
        ERROR("Input data error! status code = " << status);
        exit(-1);
    }
    status = motionShoter->run();
    if (status != MotionShot::OK) {
        ERROR("Runing error! status code = " << status);
        exit(-1);
    }

    timer.stop();
    TIMER(" - Time cost in motion shot: " << timer.getTimeSec() << "s");

//    Mat pano;
//    motionShoter->getResult(pano);

//    NamedLargeWindow("Result Pano");
//    imshow("Result Pano", pano);
//    imwrite("/home/vance/output/result_pano_shoter.jpg", pano);
//    INFO("Saving result to /home/vance/output/result_pano_shoter.jpg");
//    waitKey(0);

    return 0;
}
