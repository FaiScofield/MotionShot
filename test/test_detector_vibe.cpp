/*=================================================================
 * Extract Background & Foreground Model by ViBe Algorithm using OpenCV Library.
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

#include "utility.h"
#include "Vibe.h"
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
        g_type = LASIESTA;
    else
        g_type = VIDEO;

    VideoCapture vc;
    vector<Mat> vImags, vMaskGTs;
    if (g_type == VIDEO) {
        vc.open(dataPath);
        if (!vc.isOpened()) {
            cerr << "Unable to open video file: " << dataPath << endl;
            exit(-1);
        }
    } else {
        assert(g_type == LASIESTA);

        string gtPath = dataPath + "-GT/";
        ReadImagesFromFolder_lasiesta(dataPath, vImags);
        ReadGroundtruthFromFolder_lasiesta(gtPath, vMaskGTs);
    }

    ViBe vibe;

    bool firstFrame = true;
    Mat frame, gray, mask, mask_gt, output;
    if (g_type == VIDEO) {
        vc >> frame;
        while (!frame.empty()) {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
            if (firstFrame) {
                vibe.init(gray);
                vibe.ProcessFirstFrame(gray);
                firstFrame = false;
            } else {
                vibe.Run(gray);
                mask = vibe.getFGModel();
                if (!mask.empty()) {
                    Mat mask_color;
                    cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                    hconcat(frame, mask_color, output);
                    imshow("ViBe Method", output);
                    if (waitKey(30) == 27)
                        break;
                }
            }
            vc >> frame;
        }
    } else {
        for (size_t i = 0, iend = vImags.size(); i < iend; ++i) {
            frame = vImags[i];
            if (frame.empty()) {
                cerr << "Empty image for " << vImags[i] << endl;
                continue;
            }

            cvtColor(frame, gray, COLOR_BGR2GRAY);
            if (firstFrame) {
                vibe.init(gray);
                vibe.ProcessFirstFrame(gray);
                firstFrame = false;
            } else {
                vibe.Run(gray);
                mask = vibe.getFGModel();
                if (mask.empty())
                    continue;

                // find contours
                Mat frame_contours = frame.clone();
                vector<vector<Point>> contours, contoursFilter;
                findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
                if (contours.empty())
                    continue;

                contoursFilter.reserve(contours.size());
                const int th_minArera = 0.001 * frame.size().area();
                for (size_t i = 0; i < contours.size(); ++i) {
                    if (contours[i].size() > th_minArera)
                        contoursFilter.push_back(contours[i]);
                }
                drawContours(frame_contours, contours, -1, Scalar(2500, 0, 0), 1);
                drawContours(frame_contours, contoursFilter, -1, Scalar(0, 255, 0), 2);

                // calculate blobs
                vector<Rect> blobs;
                int maxArea = 0;
                for (int i = 0, iend = contours.size(); i < iend; ++i) {
                    const Rect blobi = boundingRect(contours[i]);
                    if (blobi.area() < th_minArera)
                        continue;
                    if (blobi.area() > maxArea)
                        maxArea = blobi.area();
                    blobs.push_back(blobi);
                }
        //        cout << " - max blob area: " << maxArea << endl;

                Mat diff_blobs, tmp, output;
                cvtColor(mask, diff_blobs, COLOR_GRAY2BGR);
                for (int i = 0, iend = blobs.size(); i < iend; ++i) {
                    rectangle(diff_blobs, blobs[i], Scalar(0, 255, 0), 1);
                    string txt = to_string(i) + "-" + to_string(blobs[i].area());
                    putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0, 0, 255));

                    //  const Point tl = blobs[i].tl();
                    //  const Point br = blobs[i].br();
                    //  mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
                }

                Mat mask_color, tmp1, tmp2;
                cvtColor(mask, mask_color, COLOR_GRAY2BGR);
                vconcat(frame, mask_color, tmp1);
                vconcat(frame_contours, diff_blobs, tmp2);
                hconcat(tmp1, tmp2, output);
                imshow("ViBe Method", output);
                if (waitKey(100) == 27)
                    break;

            }
        }
    }

    return 0;
}
