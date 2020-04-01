/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "BGDifference.h"
#include "BS_MOG2_CV.h"
#include "FramesDifference.h"
#include "OpticalFlower.h"
#include "ViBePlus.h"
#include "Vibe.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#define WATERSHED 0
#define USE_MEDIAN_FILTER_BACKGROUND    1

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
            "{type      t|VIDEO|value input type: VIDEO, LASSESTA, HUAWEI}"
            "{folder    f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
            "{detector  d|fd|value detector: bgd, fd, mog2, vibe, vibe+, flow}"
            "{size      s|5|board size for fd detector}"
            "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
            "{suffix    x|jpg|image suffix for SEQUENCE}"
            "{begin     a|0|start index for image sequence}"
            "{end       e|-1|end index for image sequence}"
            "{write     w|false|write result sequence to a dideo}"
            "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_detector = parser.get<String>("detector");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    double scale = parser.get<double>("scale");
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;
    cout << " - detector = " << str_detector << endl;

    vector<Mat> vImages, vGTs;
    InputType inputType;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
        scale = 0.3;
        ReadImageSequence_video(str_folder, vImages, beginIdx, endIdx);
    } else if (str_type == "lasiesta" || str_type == "LASSESTA") {
        inputType = LASIESTA;
        scale = 1;
        ReadImageSequence_lasiesta(str_folder, vImages, vGTs, beginIdx, endIdx);
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
        scale = 0.1;
        ReadImageSequence_huawei(str_folder, vImages, beginIdx, endIdx);
    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
        inputType = SEQUENCE;
        scale = 0.5;
        String suffix = parser.get<String>("suffix");
        ReadImageSequence(str_folder, suffix, vImages, beginIdx, endIdx);
    } else {
        ERROR("[Error] Unknown input type for " << str_type);
        return -1;
    }
    cout << " - Image size input = " << vImages[0].size() << endl;
    ResizeFlipRotateImages(vImages, scale);


    BaseMotionDetector* detector;
    if (str_detector == "bgd") {
        // TODO
        // detector = dynamic_cast<BaseMotionDetector*>(new BGDiff());
    } else if (str_detector == "fd") {
        int size = parser.get<int>("size");
        cout << " - kernel size = " << size << endl;
        assert(size > 2);
        detector = dynamic_cast<BaseMotionDetector*>(new FramesDifference(2, size, 10));
    } else if (str_detector == "mog2") {
        Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, true);
        detector = dynamic_cast<BaseMotionDetector*>(new BS_MOG2_CV(bs.get()));
    }/* else if (str_detector == "flow") {
        Ptr<FarnebackOpticalFlow> fof = FarnebackOpticalFlow::create();
        detector = dynamic_cast<BaseMotionDetector*>(new OpticalFlower(fof.get()));
    }*/ /*else if (str_detector == "vibe") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBe(20, 2, 20, 16));
    } else if (str_detector == "vibe+") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBePlus(20, 2, 20, 16));
    }*/
    else {
        ERROR("Unknown input detector for " << str_detector);
        return -1;
    }

#if USE_MEDIAN_FILTER_BACKGROUND
    // 把所有输入图像做一个中值滤波, 获得一个不变的背景
    const size_t N = vImages.size();
    Mat medianPano = Mat::zeros(vImages[0].size(), CV_8UC3);
    vector<Mat> vImgs_Y(N); // 每副图像的Y域分量
    for (size_t i = 0; i < N; ++i) {
        Mat imgYUV;
        cvtColor(vImages[i], imgYUV, COLOR_BGR2YUV);
        vector<Mat> channels;
        split(imgYUV, channels);
        vImgs_Y[i] = channels[0];
    }

    // 中值滤波
    for (int y = 0; y < vImages[0].rows; ++y) {
        Vec3b* imgRow = medianPano.ptr<Vec3b>(y);

        for(int x = 0; x < vImages[0].cols; ++x) {
            vector<pair<uchar, uchar>> vLumarAndIndex;
            for (size_t imgIdx = 0; imgIdx < N; ++imgIdx)
                vLumarAndIndex.emplace_back(vImgs_Y[imgIdx].at<uchar>(y, x), imgIdx);

            sort(vLumarAndIndex.begin(), vLumarAndIndex.end()); // 根据亮度中值决定此像素的值由哪张图像提供
            uchar idx = vLumarAndIndex[N/2].second;
            imgRow[x] = vImages[idx].at<Vec3b>(y, x);
        }
    }
    imwrite("/home/vance/output/ms/fixBackground(medianBlur).jpg", medianPano);
    detector->setFixedBackground(true);
    Mat tmpMask;
    detector->apply(medianPano, tmpMask);   // 喂第一帧
#endif

    /// detect moving frontground
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25,
                       Size(vImages[0].cols, vImages[0].rows * 2));
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        const Mat& frame = vImages[i];
        Mat diff, frameGray;
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);
        detector->apply(frame, diff);
        if (diff.empty())
            continue;
        NamedLargeWindow("mask");
        imshow("mask", diff);

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
        findContours(diff, contours, RETR_EXTERNAL,
                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE
        if (contours.empty())
            continue;

        drawContours(frame_contours, contours, -1, Scalar(0, 255, 0), 2);
#endif

        // calculate blobs
        vector<Rect> blobs;
        int maxArea = 0;
        for (int i = 0, iend = contours.size(); i < iend; ++i) {
            Rect blobi = boundingRect(contours[i]);
            if (blobi.area() < 5000)
                continue;
            if (blobi.area() > maxArea)
                maxArea = blobi.area();
            blobs.push_back(blobi);
        }
        cout << " - max blob area: " << maxArea << endl;

        Mat diff_blobs, tmp, output;
        cvtColor(diff, diff_blobs, COLOR_GRAY2BGR);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(diff_blobs, blobs[i], CV_RGB(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0, 0, 255));

            //            const Point tl = blobs[i].tl();
            //            const Point br = blobs[i].br();
            //            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }

        vconcat(frame_contours, diff_blobs, tmp);
        output = tmp;

        NamedLargeWindow("result");
        imshow("result", output);
        if (write)
            writer.write(output);
        if (waitKey(0) == 27)
            break;
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
