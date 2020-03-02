/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "ImageStitcher/ImageStitcher.h"
#include "MotionShoter/utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#define STATIC_SCENE 1
#define SAVE_OUTPUT_TO_VIDEO 0

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{type       t|VIDEO|value input type: VIDEO, LASIESTA, HUAWEI}"
               "{folder     f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
               "{subtractor s|knn|value detector: knn, mog2; (contrib:) mog, gmg, cnt, gsoc, lsbp}"
               "{begin      b|0|start index for image sequence}"
               "{end        e|-1|end index for image sequence}"
               "{scale      a|1.0|scale to resize image, 0.15 for type HUAWEI}"
               "{flip       p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
               "{rotate     r|-1|rotate image for type VIDEO, r = cv::RotateFlags(0, 1, 2)}"
               "{write      w|false|write result sequence to a dideo}"
               "{help       h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_subtractor = parser.get<String>("subtractor");
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

    // read images
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
    vector<Mat> vImages, vGTs;
    if (inputType == LASIESTA) {
        ReadImageSequence_lasiesta(str_folder, vImages, vGTs, beginIdx, endIdx);
        scale = 1;
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages, beginIdx, endIdx);
        scale = 0.1;
    } else if (inputType == VIDEO) {
        ReadImageSequence_video(str_folder, vImages, beginIdx, endIdx);
        scale = 0.3;
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);

    // generate subtractor
    Ptr<BackgroundSubtractor> subtractor;
    if (str_subtractor == "mog2") {
        subtractor = createBackgroundSubtractorMOG2(vImages.size(), 16, false);
    } else if (str_subtractor == "knn") {
        subtractor = createBackgroundSubtractorKNN(vImages.size(), 400, false);
    } else if (str_subtractor == "mog") {
        subtractor = bgsegm::createBackgroundSubtractorMOG(vImages.size());
    } else if (str_subtractor == "gmg") {
        subtractor = bgsegm::createBackgroundSubtractorGMG(1);
    } else if (str_subtractor == "cnt") {
        subtractor = bgsegm::createBackgroundSubtractorCNT();
    } else if (str_subtractor == "gsoc") {
        subtractor = bgsegm::createBackgroundSubtractorGSOC();
    } else if (str_subtractor == "lsbp") {
        subtractor = bgsegm::createBackgroundSubtractorLSBP();
    } else {
        cerr << "[Error] Unknown input detector for " << str_subtractor << endl;
        return -1;
    }

    Mat pano, panoGray;
    vector<Mat> vImgsToProcess;
#if STATIC_SCENE
    vImgsToProcess = vImages;
    pano = vImages.front();
    cvtColor(pano, panoGray, COLOR_BGR2GRAY);
    cout << " - paono size = " << pano.size() << endl;
#else  //! TODO
    vector<size_t> vIdxToProcess;
    vector<vector<size_t>> vvIdxPerIter;

    extractImagesToStitch(vImages, vImgsToProcess, vIdxToProcess, vvIdxPerIter, 10, 10);

    ImageStitcher* stitcher = new ImageStitcher();
    stitcher->stitch(vImgsToProcess.front(), vImgsToProcess.back(), pano, warpedMask1);

    vector<Mat> vHomographies;  //? 计算和pano的变换? 还是和基准帧的变换?
    vHomographies.reserve(vImgsToProcess.size());
    for (size_t i = 0, iend = vImgsToProcess.size(); i < iend; ++i) {
        vHomographies.push_back(stitcher->computeHomography(vImgsToProcess[i], pano));
    }
#endif

    /// 先拼接
#if SAVE_OUTPUT_TO_VIDEO
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 25,
                       Size(vImages[0].cols, vImages[0].rows * 2));
#endif
    // 试试把所有输入图像做一个均/中值滤波 获得一个不变的背景
    Mat aveBackground = Mat::zeros(pano.size(), CV_32FC3);
    for (size_t i = 0, iend = vImgsToProcess.size(); i < iend; ++i) {
        Mat framef;
        vImgsToProcess[i].convertTo(framef, CV_32FC3, 1. / 255.1);
        add(aveBackground, framef, aveBackground);
    }
    normalize(aveBackground, aveBackground, 1.0, 0.0, NORM_MINMAX);
    imshow("fixBackground", aveBackground);
    // waitKey(0);
    Mat firstMask;
    subtractor->apply(aveBackground, firstMask, 0);

    const int th_minArera = 0.05 * pano.size().area();
    for (size_t i = 0, iend = vImgsToProcess.size(); i < iend; ++i) {
        const Mat& frame = vImgsToProcess[i];
        Mat frameGray;
        cvtColor(frame, frameGray, COLOR_BGR2GRAY);

        Mat diff, background;
        Mat frameWarped, frameWarpedGray, warpResult;
#if STATIC_SCENE
        frameWarped = frame;
        subtractor->apply(frame, diff, 0);
//        subtractor->getBackgroundImage(background);
//        imshow("background", background);
#else
        warpPerspective(frame, frameWarped, vHomographies[i], pano.size());  //! TODO 解决像素为0的区域的背景问题

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
        morphologyEx(diff, diff, MORPH_CLOSE, kernel1);
        morphologyEx(diff, diff, MORPH_OPEN, kernel2);

        // find contours
        Mat frame_contours = frameWarped.clone();
        vector<vector<Point>> contours, contoursFilter;  //! TODO contoursFilter
        findContours(diff, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (contours.empty())
            continue;

        contoursFilter.reserve(contours.size());

        for (size_t i = 0; i < contours.size(); ++i) {
            if (contours[i].size() > th_minArera * 0.1)
                contoursFilter.push_back(contours[i]);
        }
        drawContours(frame_contours, contours, -1, Scalar(250, 0, 0), 1);
        drawContours(frame_contours, contoursFilter, -1, Scalar(0, 255, 0), 2);

        // calculate blobs
        vector<Rect> blobs;
        int maxArea = 0;
        for (int i = 0, iend = contours.size(); i < iend; ++i) {
            const Rect blobi = boundingRect(contours[i]);
            if (blobi.area() < th_minArera || blobi.area() >= frame.size().area())
                continue;
            if (blobi.area() > maxArea)
                maxArea = blobi.area();
            blobs.push_back(blobi);
        }
        if (blobs.empty())
            continue;
        //  cout << " - max blob area: " << maxArea << endl;

        Mat diff_blobs, tmp, output;
        cvtColor(diff, diff_blobs, COLOR_GRAY2BGR);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(diff_blobs, blobs[i], Scalar(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0, 0, 255));
        }
        vconcat(frame_contours, diff_blobs, output);
        imshow("result", /*diff*/ output);

        //! TODO grabCut
//        Mat cutMask, cutBgdModel, cutFgdModel;
//        // cutMask = Mat::zeros(frame.size(), CV_8UC1);
//        // cutMask(blobs[0]).setTo(Scalar(GC_PR_FGD));
//        grabCut(frame, cutMask, blobs[0], cutBgdModel, cutFgdModel, 3, GC_INIT_WITH_RECT);
//        imshow("cutMask orignal", cutMask * 50);
//        Mat cutBinMask = Mat(cutMask.size(), CV_8UC1);
//        Mat tmpMask, cutFrame;
//        compare(cutMask, 50, tmpMask, CMP_EQ);
//        cutBinMask.setTo(Scalar(255), tmpMask);
//        frame.copyTo(cutFrame, cutBinMask);
//        rectangle(cutFrame, blobs[0], Scalar(0, 0, 255), 1);
//        imshow("cutFrame", cutFrame);
//        imshow("cutMask", cutBinMask);


        if (waitKey(100) == 27)
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
