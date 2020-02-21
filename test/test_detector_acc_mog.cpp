/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "ImageStitcher/ImageStitcher.h"
#include "MotionShoter/utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#define WATERSHED 0

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
               "{type      t|VIDEO|value input type: VIDEO, LASISESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASISESTA/HUAWEI/VIDEO}"
               "{size      s|5|board size for fd detector}"
               "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
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

    vector<Mat> vImages, vGTs;
    if (inputType == LASISESTA) {
        ReadImageSequence_lasisesta(str_folder, vImages, vGTs);
        scale = 1;
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages);
        scale = 0.1;
    } else if (inputType == VIDEO) {
        ReadImagesFromVideo(str_folder, vImages);
        scale = 0.3;
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);

    /// 先拼接
    ImageStitcher* stitcher = new ImageStitcher();
    Mat pano;
    stitcher->stitch(vImages.front(), vImages.back(), pano);

//    vector<Mat> vHomographies;
//    vHomographies.reserve(vImages.size());
//    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
//        vHomographies.push_back(stitcher->computeHomography(vImages[i], pano));
//    }

    /// 再背景建模
    Ptr<BackgroundSubtractorMOG2> detector = createBackgroundSubtractorMOG2(500, 100.0, true);
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25,
                       Size(vImages[0].cols, vImages[0].rows * 2));
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        const Mat& frame = vImages[i];
        Mat diff, frameGray;
//        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        Mat frameWarped, warpResult;
        warpPerspective(frame, frameWarped, vHomographies[i], pano.size()); //! TODO 解决像素为0的区域的背景问题

        cout << frameWarped.size() << ", pano: " << pano.size() << endl;
        vconcat(frameWarped, pano, warpResult);
        imshow("warpResult", warpResult);

        cvtColor(frameWarped, frameGray, COLOR_BGR2GRAY);
        detector->apply(frameGray, diff, 0);
        if (diff.empty()) {
            cerr << "First Frame! i = " << i << endl;
            continue;
        }

        // find contours
        Mat frame_contours = frame.clone();
        vector<vector<Point>> contours;
#if WATERSHED
        vector<Vec4i> hierarchy;
        findContours(diff, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        if (contours.empty())
            continue;

        Mat markers = Mat::zeros(frame.size(), CV_32S);
        int idx = 0, compCount = 0;
        for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
            drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);
        if (compCount == 0)
            continue;

        vector<Vec3b> colorTab;
        for (int j = 0; j < compCount; j++) {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);

            colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
        }

        watershed(frame, markers);

        // paint the watershed image
        Mat wshed(markers.size(), CV_8UC3);
        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int index = markers.at<int>(i, j);
                if (index == -1)
                    wshed.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
                else if (index <= 0 || index > compCount)
                    wshed.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                else
                    wshed.at<Vec3b>(i, j) = colorTab[index - 1];
            }
        }

        wshed = wshed * 0.5 + frame * 0.5;
        imshow("watershed transform", wshed);
//        waitKey(0);
#else
//        findContours(diff, contours, RETR_EXTERNAL,
//                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE
//        if (contours.empty())
//            continue;

//        drawContours(frame_contours, contours, -1, Scalar(0, 255, 0), 2);
#endif

        // calculate blobs
//        vector<Rect> blobs;
//        int maxArea = 0;
//        for (int i = 0, iend = contours.size(); i < iend; ++i) {
//            Rect blobi = boundingRect(contours[i]);
//            //            if (blobi.area() < 10000)
//            //                continue;
//            if (blobi.area() > maxArea)
//                maxArea = blobi.area();
//            blobs.push_back(blobi);
//        }
//        cout << " - max blob area: " << maxArea << endl;

//        Mat diff_blobs, tmp, output;
//        cvtColor(diff, diff_blobs, COLOR_GRAY2BGR);
//        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
//            rectangle(diff_blobs, blobs[i], CV_RGB(0, 255, 0), 1);
//            string txt = to_string(i) + "-" + to_string(blobs[i].area());
//            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0, 0, 255));

//            //            const Point tl = blobs[i].tl();
//            //            const Point br = blobs[i].br();
//            //            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
//        }

//        vconcat(frame_contours, diff_blobs, output);
        imshow("result", diff/*output*/);
//        if (write)
//            writer.write(output);
        if (waitKey(0) == 27)
            break;
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
