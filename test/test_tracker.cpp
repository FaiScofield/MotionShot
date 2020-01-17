#include "utility.h"
#include "MotionTracker.h"
#include "BS_MOG2_CV.h"
#include "FramesDifference.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace ms;

InputType g_type;

int main(int argc, char *argv[])
{
    if (argc < 2) {
        cerr << "Arguments: <video_file> (or <data_path>)" << endl;
        exit(-1);
    }

    string dataPath(argv[1]); // "不要以/结尾"
    boost::filesystem::path path(dataPath);
    if (boost::filesystem::is_directory(path))
        g_type = DATASET;
    else
        g_type = VIDEO;

    VideoCapture vc;
    vector<string> vImags, vMaskGTs;
    if (g_type == VIDEO) {
        vc.open(dataPath);
        if (!vc.isOpened()) {
            cerr << "Unable to open video file: " << dataPath << endl;
            exit(-1);
        }
    } else {
        assert(g_type == DATASET);

        string gtPath = dataPath + "-GT/";
        ReadImageFiles(dataPath, vImags);
        ReadImageGTFiles(gtPath, vMaskGTs);
    }

    auto bs = createBackgroundSubtractorMOG2(500, 100, false);
    BS_MOG2_CV be(bs.get());

//    FramesDifference fd;

    MotionTracker tracker;
    tracker.setBackgroundSubtractor(dynamic_cast<BaseMotionDetector*>(&be));
//    tracker.SetBackgroundSubtractor(dynamic_cast<BaseBackgroundSubtractor*>(&fd));
    tracker.setMinBlobSize(Size(5, 5));

    Mat frame, gray, mask, mask_gt, output;
    if (g_type == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            tracker.substractBackground(frame, mask);
            tracker.detectBlocks();
            tracker.matchObject();
            tracker.displayObjects("Objects");
            tracker.displayDetail("Detial");

            if (waitKey(300) == 27)
                break;

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
            tracker.substractBackground(frame, mask);
            tracker.detectBlocks();
            tracker.matchObject();
            tracker.displayObjects("Objects");
            tracker.displayDetail("Details");

            if (waitKey(300) == 27)
                break;
        }
    }

    destroyAllWindows();
    return 0;
}
