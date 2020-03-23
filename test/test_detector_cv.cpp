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
#include <map>

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
               "{type       t|VIDEO|value input type: VIDEO, LASIESTA, HUAWEI, SEQUENCE}"
               "{folder     f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
               "{suffix     x|jpg|image suffix for SEQUENCE}"
               "{subtractor s|knn|value detector: knn, mog2; (contrib:) mog, gmg, cnt, gsoc, lsbp}"
               "{begin      a|0|start index for image sequence}"
               "{end        e|-1|end index for image sequence}"
               "{scale      a|1.0|scale to resize image, 0.15 for type HUAWEI}"
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

    // read images
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
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
    resizeFlipRotateImages(vImages, scale);

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
        ERROR("Unknown input detector for " << str_subtractor);
        return -1;
    }

    Mat pano, panoGray;
    vector<Mat> vImgsToProcess;
#if STATIC_SCENE
    vImgsToProcess = vImages;
    pano = vImages.front();
    cvtColor(pano, panoGray, COLOR_BGR2GRAY);
    cout << " - paono size = " << pano.size() << endl;

    const size_t N = vImgsToProcess.size();
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
    // 把所有输入图像做一个中值滤波, 获得一个不变的背景
    Mat medianPano = Mat::zeros(pano.size(), CV_8UC3);
    vector<Mat> vImgs_Y(N); // 每副图像的Y域分量
    for (size_t i = 0; i < N; ++i) {
        Mat imgYUV;
        cvtColor(vImgsToProcess[i], imgYUV, COLOR_BGR2YUV);
        vector<Mat> channels;
        split(imgYUV, channels);
        vImgs_Y[i] = channels[0];
    }

    // 中值滤波
    for (int y = 0; y < pano.rows; ++y) {
        Vec3b* imgRow = medianPano.ptr<Vec3b>(y);

        for(int x = 0; x < pano.cols; ++x) {
            vector<pair<uchar, uchar>> vLumarAndIndex;
            for (size_t i = 0; i < N; ++i)
                vLumarAndIndex.emplace_back(vImgs_Y[i].at<uchar>(y, x), i);

            sort(vLumarAndIndex.begin(), vLumarAndIndex.end()); // 根据亮度中值决定此像素的值由哪张图像提供
            uchar idx = vLumarAndIndex[N/2].second;
            imgRow[x] = vImgsToProcess[idx].at<Vec3b>(y, x);
        }
    }
    imwrite("/home/vance/output/ms/fixBackground(medianBlur).jpg", medianPano);

    // waitKey(0);
    Mat firstMask;
    subtractor->apply(medianPano, firstMask, 0);

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
        namedLargeWindow("orignal diff");
        imshow("orignal diff", diff);
        const string fileName = "/home/vance/output/ms/rough_fore_" + to_string(i + 1) + ".jpg";
        imwrite(fileName, diff);


//        threshold(diff, diff, 200, 255, THRESH_BINARY);

        // 形态学操作, 去除噪声
        Mat kernel1 = getStructuringElement(MORPH_RECT, Size(11, 11));
        Mat kernel2 = getStructuringElement(MORPH_RECT, Size(11, 11));
        morphologyEx(diff, diff, MORPH_OPEN, kernel2);
        morphologyEx(diff, diff, MORPH_CLOSE, kernel1); //

        namedLargeWindow("filter diff");
        imshow("filter diff", diff);

        // find contours
        Mat frame_contours = frameWarped.clone();
        vector<vector<Point>> contours, contoursFilter;  //! TODO contoursFilter
        findContours(diff, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if (contours.empty())
            continue;

        contoursFilter.reserve(contours.size());

        for (size_t j = 0; j < contours.size(); ++j) {
            if (contours[j].size() > th_minArera * 0.1)
                contoursFilter.push_back(contours[j]);
        }
        drawContours(frame_contours, contours, -1, Scalar(250, 0, 0), 1);
        drawContours(frame_contours, contoursFilter, -1, Scalar(0, 255, 0), 2);

        // calculate blobs
        vector<Rect> blobs;
        int maxArea = 0;
        for (int j = 0, iend = contours.size(); j < iend; ++j) {
            const Rect blobi = boundingRect(contours[j]);
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
        for (int j = 0, iend = blobs.size(); j < iend; ++j) {
            rectangle(diff_blobs, blobs[j], Scalar(0, 255, 0), 1);
            string txt = to_string(j) + "-" + to_string(blobs[j].area());
            putText(diff_blobs, txt, blobs[j].tl(), 1, 1., Scalar(0, 0, 255));
        }
        vconcat(frame_contours, diff_blobs, output);
        namedLargeWindow("result");
        imshow("result", /*diff*/ output);

        const string outName = "/home/vance/output/ms/fore-" + to_string(i + 1) + ".jpg";
        imwrite(outName, output);

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


        if (waitKey(10) == 27)
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
