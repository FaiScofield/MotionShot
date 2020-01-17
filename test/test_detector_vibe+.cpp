/*=================================================================
 * Extract Background & Foreground Model by ViBe+ Algorithm using OpenCV Library.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/

#include "ViBePlus.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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

    string dataPath(argv[1]);  // "不要以/结尾"
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

    ViBePlus vibeplus;
    vibeplus.init();

    Mat frame, gray, mask, mask_gt, output;
    if (g_type == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            vibeplus.FrameCapture(gray);
            vibeplus.Run();
            mask = vibeplus.getSegModel();

            if (!mask.empty()) {
                Mat mask_color;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_color, output);
                imshow("ViBe+ Method", output);
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

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            vibeplus.FrameCapture(gray);
            vibeplus.Run();
            mask = vibeplus.getSegModel();

            if (!mask.empty()) {
                Mat mask_color, tmp;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                hconcat(frame, mask_gt, tmp);
                hconcat(tmp, mask_color, output);
                imshow("ViBe+ Method", output);
                if (waitKey(30) == 27)
                    break;
            }
        }
    }


    return 0;
}
