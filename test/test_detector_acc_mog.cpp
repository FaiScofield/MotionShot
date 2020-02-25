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

#define STATIC_SCENE    1
#define SAVE_OUTPUT_TO_VIDEO   0

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
               "{type      t|VIDEO|value input type: VIDEO, LASIESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
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
    } else if (str_type == "lasiesta" || str_type == "LASSESTA") {
        inputType = LASIESTA;
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    vector<Mat> vImages, vGTs;
    if (inputType == LASIESTA) {
        ReadImageSequence_lasiesta(str_folder, vImages, vGTs);
        scale = 1;
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages);
        scale = 0.1;
    } else if (inputType == VIDEO) {
        ReadImagesFromVideo(str_folder, vImages);
        scale = 0.3;
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);

    vector<Mat> vImgsToProcess;
    vector<size_t> vIdxToProcess;
    vector<vector<size_t>> vvIdxPerIter;

    /// 先拼接
    Mat pano, panoGray, warpedMask1;

#if STATIC_SCENE
    vImgsToProcess = vImages;
    pano = vImages.back();
#else
    extractImagesToStitch(vImages, vImgsToProcess, vIdxToProcess, vvIdxPerIter, 10, 10);

    ImageStitcher* stitcher = new ImageStitcher();
    stitcher->stitch(vImgsToProcess.front(), vImgsToProcess.back(), pano,warpedMask1);

    vector<Mat> vHomographies;  //? 计算和pano的变换? 还是和基准帧的变换?
    vHomographies.reserve(vImgsToProcess.size());
    for (size_t i = 0, iend = vImgsToProcess.size(); i < iend; ++i) {
        vHomographies.push_back(stitcher->computeHomography(vImgsToProcess[i], pano));
    }
#endif

    cvtColor(pano, panoGray, COLOR_BGR2GRAY);
    cout << " - paono size = " << pano.size() << endl;

    /// 再背景建模
    Ptr<BackgroundSubtractorMOG2> detector = createBackgroundSubtractorMOG2(vImgsToProcess.size(), 25.0, false); //! TODO. 输入背景

#if SAVE_OUTPUT_TO_VIDEO
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25,
                       Size(vImages[0].cols, vImages[0].rows * 2));
#endif
    for (size_t i = 0, iend = vImgsToProcess.size(); i < iend; ++i) {
        const Mat& frame = vImgsToProcess[i];
        Mat frameGray;
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        Mat diff, background;
        Mat frameWarped, frameWarpedGray, warpResult;
#if STATIC_SCENE
        detector->apply(frame/*Gray*/, diff, 0);
        detector->getBackgroundImage(background);
        frameWarped = frame;
        imshow("background", background);
#else
        warpPerspective(frame, frameWarped, vHomographies[i], pano.size()); //! TODO 解决像素为0的区域的背景问题

        cout << "warp size: " << frameWarped.size() << ", pano: " << pano.size() << endl;
        vconcat(frameWarped, pano, warpResult);
        imshow("warpResult", warpResult);

        cvtColor(frameWarped, frameWarpedGray, COLOR_BGR2GRAY);
        detector->apply(frameWarpedGray, diff);
        detector->getBackgroundImage(background);
#endif
        if (diff.empty()) {
            cerr << "   First Frame! i = " << i << endl;
            continue;
        }
        // 形态学操作, 去除噪声
        Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(diff, diff, MORPH_OPEN, kernel1);
        morphologyEx(diff, diff, MORPH_CLOSE, kernel2);

        // find contours
        Mat frame_contours = frameWarped.clone();
        vector<vector<Point>> contours, contoursFilter; //! TODO contoursFilter
        findContours(diff, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (contours.empty())
            continue;

        contoursFilter.reserve(contours.size());
        const int th_minArera = 0.05 * pano.size().area();
        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > th_minArera * 0.2)
                contoursFilter.push_back(contours[i]);
        }
        drawContours(frame_contours, contours, -1, Scalar(250, 0, 0), 1);
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
        cvtColor(diff, diff_blobs, COLOR_GRAY2BGR);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(diff_blobs, blobs[i], Scalar(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0, 0, 255));

            //  const Point tl = blobs[i].tl();
            //  const Point br = blobs[i].br();
            //  mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }

        vconcat(frame_contours, diff_blobs, output);
        imshow("result", /*diff*/output);
        if (waitKey(300) == 27)
            break;

#if SAVE_OUTPUT_TO_VIDEO
        if (write)
            writer.write(output);
    }
    writer.release();
#else
    }
#endif

    destroyAllWindows();
    return 0;
}
