/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "ImageStitcher/ImageStitcher.h"
#include "OpticalFlower.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/features2d.hpp>

#define BLNED_IMAGE     0
#define GET_FOREGROUND_FROM_GT  1   // for Lasiesta dataset

using namespace cv;
using namespace std;
using namespace ms;

typedef vector<Point> Contour;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder    f| |LASIESTA dataset folder}"
               "{detector  d|AKAZE|Feature detector for sparse flow tracking}"
               "{start     s|0|start index for image sequence}"
               "{end       e|-1|end index for image sequence}"
               "{write     w|false|write result sequence to a dideo}"
               "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    TickMeter timer;
    timer.start();

    String str_detector = parser.get<String>("detector");
    String str_folder = parser.get<String>("folder");
    if (str_folder[str_folder.size() - 1] == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    int start = parser.get<int>("start");
    int end = parser.get<int>("end");
    cout << " - folder = " << str_folder << endl;
    cout << " - feature = " << str_detector << endl;
    cout << " - start index = " << start << endl;
    cout << " - end index = " << end << endl;

    Ptr<FeatureDetector> featureDetector;
    if (str_detector == "ORB") {
        featureDetector = ORB::create();
    } else if (str_detector == "MSER") {
        featureDetector = MSER::create();
    } else if (str_detector == "GFTTD") {
        featureDetector = GFTTDetector::create();
    } else if (str_detector == "KAZE") {
        featureDetector = KAZE::create();
    } else if (str_detector == "AKAZE") {
        featureDetector = AKAZE::create();
    } else if (str_detector == "BLOB") {
        featureDetector = SimpleBlobDetector::create();
    } else {
        cerr << "[Error] Unknown feature type for " << str_detector << endl;
        exit(-1);
    }
//    Ptr<> featureDetector = nullptr;

    /// read images
    vector<Mat> vImages, vGTsColor;
    ReadImageSequence_lasiesta(str_folder, vImages, vGTsColor, start, end - start);
    assert(!vImages.empty() && !vGTsColor.empty());

    timer.stop();
    TIMER("系统初始化耗时(s): " << timer.getTimeSec());

    /// get pano
    const Mat pano = vImages.back();
    Mat panoGray;
    cvtColor(pano, panoGray, COLOR_BGR2GRAY);

    /// detect moving frontground
    Ptr<BackgroundSubtractorMOG2> backgroundSubtractor = createBackgroundSubtractorMOG2(vImages.size(), 25.0, false);
    Ptr<SparsePyrLKOpticalFlow> opticalFlowCalculator = SparsePyrLKOpticalFlow::create();
//    Ptr<AKAZE> featureDetector = AKAZE::create();

//    const bool write = parser.get<bool>("write");
//    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M','J','P','G'), 10,
//                       Size(vImages[0].cols * 2, vImages[0].rows * 2));

    timer.start();

    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(7, 7));
    const int th_minBlobArea = 0.05 * pano.size().area();

    vector<vector<Rect>> vvBlobs;
    vector<vector<Contour>> vvContours;
    vvBlobs.reserve(vImages.size() - 1);

    Mat lastFrame, currentFrame, currentMask, lastMask;
    Mat currentDesc, lastDesc;
    vector<KeyPoint> vCurrentKPs, vLastKPs;
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        cvtColor(vImages[i], currentFrame, COLOR_BGR2GRAY);

        // 1.get foreground mask
#if GET_FOREGROUND_FROM_GT
        cvtColor(vGTsColor[i], currentMask, COLOR_BGR2GRAY);
#else   // get foreground form BackgroundSubtractor
        backgroundSubtractor->apply(currentFrame, currentMask, 0);
        morphologyEx(currentMask, currentMask, MORPH_OPEN, kernel1);   // 开操作去除噪点
        morphologyEx(currentMask, currentMask, MORPH_CLOSE, kernel2);  // 关操作补孔洞
#endif
        // 2.get foreground blobs
        vector<Contour> vContours, vContoursFilter;
        vector<Rect> vBlobs;
        findContours(currentMask, vContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (vContours.empty()) {
            vvBlobs.push_back(vBlobs);
            vvContours.push_back(vContours);
            continue;
        }
        Mat frameOut = vImages[i].clone();
        drawContours(frameOut, vContours, -1, Scalar(0, 255, 0), 2);

        int maxArea = 0;
        Rect maxBlob;
        for (size_t j = 0; j < vContours.size(); ++j) {
            Rect blob = boundingRect(vContours[j]);
            if (blob.area() > th_minBlobArea) {
                vBlobs.push_back(blob);
                vContoursFilter.push_back(vContours[j]);
                rectangle(frameOut, blob, Scalar(255, 0, 0), 2);
                if (blob.area() > maxArea) {
                    maxArea = blob.area();
                    maxBlob = blob;
                }
            }
        }
        vvBlobs.push_back(vBlobs);
        vvContours.push_back(vContoursFilter);
        rectangle(frameOut, maxBlob, Scalar(0, 0, 255), 2);
        const string txt = "area:" + to_string(maxBlob.area());
        putText(frameOut, txt, maxBlob.tl(), 1, 1., Scalar(0,0,255));

        Mat maskColor;
        cvtColor(currentMask, maskColor, COLOR_GRAY2BGR);
        hconcat(frameOut, maskColor, frameOut);

//        Mat maxBlobMask = Mat::zeros(frame.size(), CV_8UC1);
//        maxBlobMask(maxBlob).setTo(255);
//        bitwise_and(maxBlobMask, diff, maxBlobMask);
//        vForeMaskPrecise.push_back(maxBlobMask);
//        rectangle(frameOut, maxBlob, Scalar(0, 0, 255), 2);
//        imshow("contour & blob", frameOut);
//        imshow("maxBlobMask", maxBlobMask);

        imshow("contour & blob", frameOut);

        // 3.calculate optical flow
        Mat maskk = Mat::zeros(currentFrame.size(), CV_8UC1);
        for (size_t k = 0; k < vBlobs.size(); ++k)
            maskk(vBlobs[k]).setTo(255);
//        featureDetector->detectAndCompute(currentFrame, maskk, vCurrentKPs, currentDesc);
//        Mat featureShow = vImages[i].clone();
//        drawKeypoints(featureShow, vCurrentKPs, featureShow, Scalar(0,255,0));
//        imshow("featureShow", featureShow);
        if (i == 0) {
            lastFrame = currentFrame.clone();
            lastMask = currentMask.clone();
            lastDesc = currentDesc.clone();
            vLastKPs = vCurrentKPs;
            continue;
        }

//        featureDetector->

        vector<uchar> status;
        vector<float> err;
        vector<Point2f> vKPsLast, vKPsCurr;
        KeyPoint::convert(vLastKPs, vKPsLast);
         goodFeaturesToTrack(lastFrame, vKPsLast, 2000, 0.05, 2, maskk);
        opticalFlowCalculator->calc(lastFrame, currentFrame, vKPsLast, vKPsCurr, status);
//        calcOpticalFlowPyrLK(lastFrame, currentFrame, vKPsLast, vKPsCurr, status, err);
//        calcOpticalFlowBM();
        Mat flow = vImages[i].clone();
        for (size_t j = 0, iend = status.size(); j < iend; j++) {
            if (status[j]) {
                circle(flow, vKPsLast[j], 2, Scalar(255,0,0), -1);
                circle(flow, vKPsCurr[j], 2, Scalar(0,255,0), -1);
                line(flow, vKPsLast[j], vKPsCurr[j], Scalar(0,255,0));
            }
        }
        imshow("flow", flow);

        if (waitKey(1500) == 27)
            break;
        lastFrame = currentFrame.clone();
        lastMask = currentMask.clone();
        lastDesc = currentDesc.clone();
        vLastKPs = vCurrentKPs;
    }
    timer.stop();
    TIMER("Cost time in sparse flow: " << timer.getTimeSec());

//    writer.release();
    destroyAllWindows();


//    Ptr<Tracker> tracker = Tracker::create("KCF");

#if BLNED_IMAGE
    /// stitch image
    timer.start();
    vector<Mat> vToStitch;
    vToStitch.reserve(10);
    for (size_t i = 0; i < N - 1; ++i) {
        if (i % 10 != 0)
            continue;
        vToStitch.push_back(vImages[i]);
    }
    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
    Stitcher::Status status = stitcher->stitch(vToStitch, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    timer.stop();
    cout << " - image size: " << vImages[0].cols << " x " << vImages[0].rows << endl;
    cout << " - pano size: " << pano.cols << " x " << pano.rows << endl;
    cout << "[Timer] Cost time in stitching images: " << timer.getTimeSec()  << endl;


    /// get foregrounds
    assert(vBlobs1.size() == N - 1);

    timer.start();
    int delta = 20;
    vector<Mat> vForegroundMasks;
    vector<Mat> vPanoMasks;
    vector<Mat> vHomographies;
    vector<size_t> vImageIndex;
    vForegroundMasks.reserve(N / delta);
    vPanoMasks.reserve(N / delta);
    vHomographies.reserve(N / delta);
    vImageIndex.reserve(N / delta);
    ms::ImageStitcher msStitcher;
    for (size_t i = 0; i < N - 2; ++i) {
        if ((i + 1) % delta != 0)
            continue;

//        if (vBlobs.size() != 1)
//            continue;

        Mat Hi = msStitcher.computeHomography(vImages[i], pano);
        vHomographies.push_back(Hi);
        vImageIndex.push_back(i);

        Mat mask = Mat::zeros(vImages[i].size(), CV_8UC1);
        vector<Rect> blobs = vBlobs1[i - 1];
        for (int j = 0, iend = blobs.size(); j < iend; ++j) {
            const Point tl = blobs[j].tl();
            const Point br = blobs[j].br();
            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }
        Mat mask_pano;
        warpPerspective(mask, mask_pano, Hi, pano.size());
        vForegroundMasks.push_back(mask);
        vPanoMasks.push_back(mask_pano);

        // show mask
        Mat dst, imgGray;
        cvtColor(vImages[i], imgGray, COLOR_BGR2GRAY);
        bitwise_and(imgGray, mask, dst);
        imshow("mask dst", dst);
        waitKey(30);
    }
    vImageIndex.pop_back(); // 去掉最后2帧, 前景太多
    vImageIndex.pop_back();
    timer.stop();
    cout << "[Timer] Cost time in calc H: " << timer.getTimeSec()  << endl;


    /// blending
//    detail::MultiBandBlender* blender = new detail::MultiBandBlender();
    vector<Mat> vBlendImgs;
    Mat foreground = Mat::zeros(pano.size(), CV_8UC3);
    Mat foregroundMask = Mat::zeros(pano.size(), CV_8UC1);
    for (size_t i = 0; i < vImageIndex.size(); ++i) {
        const size_t idx = vImageIndex[i];
        Mat blendImage = pano.clone();
        Mat warped, tmp;
        warpPerspective(vImages[idx], warped, vHomographies[i], pano.size());
//        vconcat(warped, pano, tmp);
//        imshow("img_warped & pano", tmp);

        Mat fi;
        if (vBlobs1[idx - 1].empty())
            continue;
        const Point tl = vBlobs1[idx - 1][0].tl();
        const Point br = vBlobs1[idx - 1][0].br();
        Point p = (tl + br) * 0.5;
        if (p.x < 10 || p.y < 10 || p.x > pano.cols - 10 || p.y > pano.rows - 10)
            continue;
        seamlessClone(vImages[idx], blendImage, vForegroundMasks[i], p, blendImage, NORMAL_CLONE);

        string idxOut = "/home/vance/output/blend_" + to_string(vImageIndex[i]) + ".jpg";
        imshow("blend", blendImage);
        imwrite(idxOut, blendImage);
        waitKey(1000);

        bitwise_and(foregroundMask, vPanoMasks[i], foregroundMask);
        blendImage.rowRange(tl.y, br.y).colRange(tl.x, br.x).copyTo(foreground.rowRange(tl.y, br.y).colRange(tl.x, br.x));
        imshow("foreground", foreground);
        vBlendImgs.push_back(blendImage);
    }
    imwrite("/home/vance/output/foreground.jpg", foreground);
    Mat result;
    Point po(pano.cols/2, pano.rows/2);
    seamlessClone(foreground, pano, foregroundMask, po, result, NORMAL_CLONE);
    imshow("result", result);
    imwrite("/home/vance/output/result.jpg", result);
#endif

    return 0;
}


//int main(int argc, char **argv) {

//    // images, note they are CV_8UC1, not CV_8UC3
//    cv::Mat img1 = imread(file_1, 0);
//    cv::Mat img2 = imread(file_2, 0);

//    // key points, using GFTT here.
//    vector<cv::KeyPoint> kp1;
//    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
//    detector->detect(img1, kp1);

//    // now lets track these key points in the second image
//    // first use single level LK in the validation picture
//    vector<cv::KeyPoint> kp2_single;
//    vector<bool> success_single;
//    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, false);

//    cv::Mat img2_single;
//    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
//    for (int i = 0; i < kp2_single.size(); i++) {
//        if (success_single[i]) {
//            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
//        }
//    }

//    cv::imshow("tracked single level (forward)", img2_single);
//    cv::imwrite("tracked_single_forward.png", img2_single);

//    kp2_single.clear();
//    success_single.clear();
//    img2_single.release();

//    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single, true);
//    cv::cvtColor(img2, img2_single, CV_GRAY2BGR);
//    for (int i = 0; i < kp2_single.size(); i++) {
//        if (success_single[i]) {
//            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
//        }
//    }

//    cv::imshow("tracked single level (inverse)", img2_single);
//    cv::imwrite("tracked_single_inverse.png", img2_single);

//    // then test multi-level LK
//    vector<cv::KeyPoint> kp2_multi;
//    vector<bool> success_multi;
//    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, false);

//    cv::Mat img2_multi;
//    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
//    for (int i = 0; i < kp2_multi.size(); i++) {
//        if (success_multi[i]) {
//            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
//        }
//    }

//    cv::imshow("tracked multi level (forward)", img2_multi);
//    cv::imwrite("tracked_multi_forward.png", img2_multi);

//    kp2_multi.clear();
//    success_multi.clear();
//    img2_multi.release();

//    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
//    cv::cvtColor(img2, img2_multi, CV_GRAY2BGR);
//    for (int i = 0; i < kp2_multi.size(); i++) {
//        if (success_multi[i]) {
//            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
//        }
//    }

//    cv::imshow("tracked multi level (inverse)", img2_multi);
//    cv::imwrite("tracked_multi_inverse.png", img2_multi);


//    // use opencv's flow for validation
//    vector<cv::Point2f> pt1, pt2;
//    for (auto &kp: kp1) pt1.push_back(kp.pt);
//    vector<uchar> status;
//    vector<float> error;
//    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error, cv::Size(8, 8));

//    cv::Mat img2_CV;
//    cv::cvtColor(img2, img2_CV, CV_GRAY2BGR);
//    for (int i = 0; i < pt2.size(); i++) {
//        if (status[i]) {
//            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
//            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
//        }
//    }

//    // show the result
//    cv::imshow("tracked by opencv", img2_CV);
//    cv::imwrite("tracked_by_opencv.png", img2_CV);

//    cv::waitKey(0);
//    cv::destroyAllWindows();

//    return 0;
//}
