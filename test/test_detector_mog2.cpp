#include "utility.h"
#include "BS_MOG2_CV.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace ms;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cerr << "Arguments: <video_file> (or <data_path>)" << endl;
        exit(-1);
    }

    InputType inputType;
    string dataPath(argv[1]); // "不要以/结尾"
    boost::filesystem::path path(dataPath);
    if (boost::filesystem::is_directory(path))
        inputType = DATASET;
    else
        inputType = VIDEO;

    VideoCapture vc;
    vector<string> vImags, vMaskGTs;
    if (inputType == VIDEO) {
        vc.open(dataPath);
        if (!vc.isOpened()) {
            cerr << "Unable to open video file: " << dataPath << endl;
            exit(-1);
        }
    } else {
        assert(inputType == DATASET);

        string gtPath = dataPath + "-GT/";
        ReadImageFiles(dataPath, vImags);
        ReadImageGTFiles(gtPath, vMaskGTs);
    }

    bool detectShadows = true;
    Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, detectShadows);
    BS_MOG2_CV extractor(bs.get());

    Mat frame, mask, mask_gt, output;
    if (inputType == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            extractor.apply(frame, mask);
            if (!mask.empty()) {
                Mat mask_color;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_color, output);
                imshow("MOG2 Method by Opencv", output);
                if (waitKey(50) == 27)
                    break;
            }
            vc >> frame;
        }
    } else {
        for (size_t i = 0, iend = vImags.size(); i < iend; ++i) {
            frame = imread(vImags[i], CV_LOAD_IMAGE_COLOR);
            mask_gt = imread(vMaskGTs[i], CV_LOAD_IMAGE_COLOR);
            if (frame.empty()) {
                cerr << "Empty image for " << vImags[i] << endl;
                continue;
            }
            extractor.apply(frame, mask);
            if (!mask.empty()) {
                Mat mask_color, tmp;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_gt, tmp);
                hconcat(tmp, mask_color, output);
                imshow("MOG2 Method by Opencv", output);
                if (waitKey(50) == 27)
                    break;
            }
        }
    }

    destroyAllWindows();
    return 0;
}
