/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "BS_MOG2_CV.h"
#include "BGDifference.h"
#include "FramesDifference.h"
#include "Vibe.h"
#include "ViBePlus.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
        "{type      t|DATASET|input type for one of VIDEO, DATASET}"
        "{folder    f|  |data folder or video file for type DATASET or VIDEO}"
        "{detector  d|fd|detector for ont of bgd, fd, mog2, vibe, vibe+}"
        "{delta     a|2|delta for fd detector}"
        "{size      s|3|board size for fd detector}"
        "{showGT    gt|false|if show ground for type DATASET}"
        "{help      h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    InputType inputType;
    const String str_type = parser.get<String>("type");
    if (str_type == "dataset" || str_type == "DATASET") {
        inputType = DATASET;
    } else if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    String str_folder = parser.get<String>("folder");
    if (str_folder.find_last_of('/') == String::npos)
        str_folder = str_folder.substr(0, str_folder.size() - 1);

    String str_gtFolder;
    bool showGT = parser.get<bool>("showGT");
    if (showGT) {
        str_gtFolder = str_folder + "-GT/";
        boost::filesystem::path path(str_gtFolder);
        if (path.empty()) {
            cerr << "[Error] Ground truth folder doesn't exist: " << str_gtFolder << endl;
            str_gtFolder.clear();
            showGT = false;
        }
    }

    int delta = -1, size = -1;
    BaseMotionDetector* detector;
    const String str_detector = parser.get<String>("detector");
    if (str_detector == "bgd") {
        // TODO
        //detector = dynamic_cast<BaseMotionDetector*>(new BGDiff());
    } else if (str_detector == "fd") {
        delta = parser.get<int>("delta");
        size = parser.get<int>("size");
        detector = dynamic_cast<BaseMotionDetector*>(new FramesDifference(delta, size));
    } else if (str_detector == "mog2") {
        Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, true);
        detector = dynamic_cast<BaseMotionDetector*>(new BS_MOG2_CV(bs.get()));
    } /*else if (str_detector == "vibe") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBe(20, 2, 20, 16));
    } else if (str_detector == "vibe+") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBePlus(20, 2, 20, 16));
    }*/ else {
        cerr << "[Error] Unknown input detector for " << str_detector << endl;
        return -1;
    }

    /// input datas
    vector<Mat> vImages, vMaskGTs;
    if (inputType == DATASET) {
        ReadImageSequence_lasisesta(str_folder, vImages, vMaskGTs);
    } else if (inputType == VIDEO) {
        ReadImagesFromVideo(str_folder, vImages);
    }

    /// detect moving frontground
    Mat frame, mask, mask_gt, output;
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        const Mat& frame = vImages[i];
        detector->apply(frame, mask);
        if (!mask.empty()) {
            Mat mask_color, tmp;
            cvtColor(mask, mask_color, COLOR_GRAY2BGR);
            hconcat(frame, mask_color, tmp);
            if (showGT)
                hconcat(tmp, vMaskGTs[i], output);
            else
                output = tmp;

            imshow("result", output);
            if (waitKey(30) == 27)
                break;
        }
    }

    destroyAllWindows();
    return 0;
}
