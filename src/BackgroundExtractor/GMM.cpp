#include "utility.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace ms;

#define USE_MOG2    1
#define USE_KNN     0
#define USE_MORP    0

InputType g_type;

int main(int argc, char** argv)
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

    bool detectShadows = true;
#if USE_MOG2
    Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, detectShadows);
#elif USE_KNN
    Ptr<BackgroundSubtractorKNN> bs = createBackgroundSubtractorKNN(20, 16.0, detectShadows);
#endif

    Mat frame, mask, mask_gt, output;
    if (g_type == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            bs->apply(frame, mask);
            if (!mask.empty()) {
            #if USE_MORP
                erode(mask, mask, Mat());   // 腐蚀
                erode(mask, mask, Mat());   // 腐蚀
                dilate(mask, mask, Mat());  // 膨胀
                dilate(mask, mask, Mat());  // 膨胀
                dilate(mask, mask, Mat());  // 膨胀
                erode(mask, mask, Mat());   // 腐蚀
            #endif
                Mat mask_color;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_color, output);
                imshow("MOG Method", output);
                if (waitKey(30) == 27)
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
            bs->apply(frame, mask);
            if (!mask.empty()) {
            #if USE_MORP
                erode(mask, mask, Mat());   // 腐蚀
                erode(mask, mask, Mat());   // 腐蚀
                dilate(mask, mask, Mat());  // 膨胀
                dilate(mask, mask, Mat());  // 膨胀
                dilate(mask, mask, Mat());  // 膨胀
                erode(mask, mask, Mat());   // 腐蚀
            #endif
                Mat mask_color, tmp;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_gt, tmp);
                hconcat(tmp, mask_color, output);
                imshow("MOG Method", output);
                if (waitKey(30) == 27)
                    break;
            }
        }
    }

    destroyAllWindows();
    return 0;
}
