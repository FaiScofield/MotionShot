/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "utility.h"
#include "FramesDifference.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;
using namespace ms;

InputType g_type;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        cerr << "Arguments: <video_file> (or <data_path>) [delta]" << endl;
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

    int delta = 2;  // 二帧差/三帧差
    if (argc > 2)
        delta = atoi(argv[2]);
    if (delta < 2)
        delta = 2;
    cerr << " - set delta to " << delta << endl;

    FramesDifference fd(delta, 3);

    Mat frame, mask, mask_gt, output;
    if (g_type == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            fd.apply(frame, mask);
            if (!mask.empty()) {
                Mat mask_color;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_color, output);
                imshow("Frame Difference Method", output);
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

            Mat gray;
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            fd.apply(gray, mask);
            if (!mask.empty()) {
                Mat mask_color, tmp;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_gt, tmp);
                hconcat(tmp, mask_color, output);
                imshow("Frame Difference Method", output);
                if (waitKey(30) == 27)
                    break;
            }
        }
    }

    destroyAllWindows();
    return 0;
}
