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
#include <opencv2/stitching.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/photo.hpp>

#define USE_CAMSHIT     0
#define USE_MEANSHIT    0
#define BLNED_IMAGE     0

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{type      t|VIDEO|value input type: VIDEO, LASSESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
               "{dense     d|true|dense flow (otherwise sparse flow)}"
               "{size      s|5|kernel size for morphology operators}"
               "{showGT    g|false|if show ground for type DATASET}"
               "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
               "{flip      p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
               "{rotate    r|-1|rotate image for type VIDEO, r = RotateFlags(0, 1, 2)}"
               "{write     w|false|write result sequence to a dideo}"
               "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    TickMeter timer;
    timer.start();

    const String str_type = parser.get<String>("type");
    String str_folder = parser.get<String>("folder");
    if (str_folder[str_folder.size() - 1] == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    double scale = parser.get<double>("scale");
    int flip = parser.get<int>("flip");
    int rotate = parser.get<int>("rotate");
    bool showGT = parser.get<bool>("showGT");
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;

    InputType inputType;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
        showGT = false;
    } else if (str_type == "lasiesta" || str_type == "LASIESTA") {
        inputType = LASIESTA;
        cout << " - showGT = " << showGT << endl;
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
        showGT = false;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    //// read images
    vector<Mat> vImages, vGTs;
    if (inputType == LASIESTA) {
        ReadImageSequence_lasiesta(str_folder, vImages, vGTs);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages);
        // scale = 0.15;
    } else if (inputType == VIDEO) {
        ReadImagesFromVideo(str_folder, vImages);
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);
    timer.stop();
    cout << "[Timer] Cost time in reading datas: " << timer.getTimeSec()  << endl;

    /// get pano
//    timer.start();

//    vector<Mat> toStitch(5);
//    const int delta = vImages.size() / 5;
//    for (int i = 0; i < 5; ++i)
//        toStitch[i] = vImages[i * delta];

//    Mat pano;
//    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
//    Stitcher::Status status = stitcher->stitch(toStitch, pano);
//    if (status != Stitcher::OK) {
//        cerr << "Can't stitch images, error code = " << int(status) << endl;
//        system("pause");
//        return -1;
//    }
//    timer.stop();
//    double time = timer.getTimeSec() ;
//    cout << " - Time cost in stitching = " << time << "s" << endl;
//    cout << " - Image size = " << vImages[0].size() << endl;
//    cout << " - Pano size = " << pano.size()  << endl;
    const Mat pano = vImages.back();
    Mat panoGray;
    cvtColor(pano, panoGray, COLOR_BGR2GRAY);

    /// detect moving frontground
    bool dense = parser.get<bool>("dense");
#ifdef USE_OPENCV4
    Ptr<DISOpticalFlow> detector1 = DISOpticalFlow::create();
    Ptr<VariationalRefinement> detector2 = VariationalRefinement::create();
//    Ptr<FarnebackOpticalFlow> detector2 = FarnebackOpticalFlow::create();
#else
    Ptr<DualTVL1OpticalFlow> detector1 = DualTVL1OpticalFlow::create();
    Ptr<FarnebackOpticalFlow> detector2 = FarnebackOpticalFlow::create();
#endif
//    const bool write = parser.get<bool>("write");
//    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M','J','P','G'), 10,
//                       Size(vImages[0].cols * 2, vImages[0].rows * 2));

    int size = parser.get<int>("size");
    cout << " - kernel size = " << size << endl;
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    Mat lastFrame, currentFrame;
    vector<vector<Rect>> vBlobs1, vBlobs2;
    vBlobs1.reserve(vImages.size() - 1);
    vBlobs2.reserve(vImages.size() - 1);
    double t1 = 0, t2 = 0;
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        cvtColor(vImages[i], currentFrame, COLOR_BGR2GRAY);
        if (i == 0) {
            currentFrame.copyTo(lastFrame);
            continue;
        }

        /* detector1 method */
        timer.start();
        Mat flow1, flow1_uv[2], mag1, ang1, hsv1, hsv_split1[3], bgr1;
        detector1->calc(lastFrame/*panoGray*/, currentFrame, flow1); // get flow type CV_32FC2
        split(flow1, flow1_uv);
        multiply(flow1_uv[1], -1, flow1_uv[1]);
        cartToPolar(flow1_uv[0], flow1_uv[1], mag1, ang1, true); // 笛卡尔转极坐标系
        normalize(mag1, mag1, 0, 1, NORM_MINMAX);
        hsv_split1[0] = ang1;
        hsv_split1[1] = mag1;
        hsv_split1[2] = Mat::ones(ang1.size(), ang1.type());
        merge(hsv_split1, 3, hsv1);
        cvtColor(hsv1, bgr1, COLOR_HSV2BGR);    // bgr1 type = CV_32FC3
        Mat rgbU;
        bgr1.convertTo(rgbU, CV_8UC3, 255, 0);
        timer.stop();
        t1 += timer.getTimeSec() ;

        // 二值化
        Mat mask1(bgr1.size(), CV_8UC1, Scalar(255)), grayU1;
        cvtColor(rgbU, grayU1, COLOR_BGR2GRAY);
        mask1 = mask1 - grayU1;
        threshold(mask1, mask1, 20, 255, THRESH_BINARY); // THRESH_OTSU, THRESH_BINARY
        erode(mask1, mask1, kernel);
        dilate(mask1, mask1, kernel);
        dilate(mask1, mask1, kernel);

        // find contours
        Mat outContours1 = vImages[i].clone();
        vector<vector<Point>> contours1;
        findContours(mask1, contours1, RETR_EXTERNAL,
                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE
        drawContours(outContours1, contours1, -1, Scalar(0, 255, 0), 2);

        // calculate blobs
        vector<Rect> blobs1;
        const auto th = 0.5 * currentFrame.size().area();
        for (int i = 0, iend = contours1.size(); i < iend; ++i) {
            Rect blobi = boundingRect(contours1[i]);
            if (blobi.area() > th) // 删去太大的前景
                continue;
            blobs1.push_back(blobi);
        }
        vBlobs1.push_back(blobs1);

        for (int i = 0, iend = blobs1.size(); i < iend; ++i) {
            rectangle(outContours1, blobs1[i], Scalar(0, 0, 255), 1);
            string txt = to_string(i) + "-" + to_string(blobs1[i].area());
            putText(outContours1, txt, blobs1[i].tl(), 1, 1., Scalar(0,0,255));
        }

        //! TODO 利用前景运动的连续性做约束去除无效的前景

        /* detector2 method */
        timer.start();
        Mat flow2, flow2_uv[2], mag2, ang2, hsv2, hsv_split2[3], bgr2;
        detector2->calc(lastFrame/*panoGray*/, currentFrame, flow2);
        split(flow2, flow2_uv);
        multiply(flow2_uv[1], -1, flow2_uv[1]);
        cartToPolar(flow2_uv[0], flow2_uv[1], mag2, ang2, true);
        normalize(mag2, mag2, 0, 1, NORM_MINMAX);
        hsv_split2[0] = ang2;
        hsv_split2[1] = mag2;
        hsv_split2[2] = Mat::ones(ang2.size(), ang2.type());
        merge(hsv_split2, 3, hsv2);
        cvtColor(hsv2, bgr2, COLOR_HSV2BGR);
        Mat rgbU2;
        bgr2.convertTo(rgbU2, CV_8UC3,  255, 0);
        timer.stop();
        t2 += timer.getTimeSec() ;

        // 二值化
        Mat mask2(bgr2.size(), CV_8UC1, Scalar(255)), grayU2;
        cvtColor(rgbU2, grayU2, COLOR_BGR2GRAY);
        mask2 = mask2 - grayU2;
        threshold(mask2, mask2, 20, 255, THRESH_BINARY); // THRESH_OTSU, THRESH_BINARY
        erode(mask2, mask2, kernel);
        dilate(mask2, mask2, kernel);
        dilate(mask2, mask2, kernel);

        // find contours
        Mat outContours2 = vImages[i].clone();
        vector<vector<Point>> contours2;
        findContours(mask2, contours2, RETR_EXTERNAL,
                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE
        drawContours(outContours2, contours2, -1, Scalar(0, 255, 0), 2);

        // calculate blobs
        vector<Rect> blobs2;
        for (int i = 0, iend = contours2.size(); i < iend; ++i) {
            Rect blobi = boundingRect(contours2[i]);
            if (blobi.area() > th) // 删去太大的前景
                continue;
            blobs2.push_back(blobi);
        }
        vBlobs2.push_back(blobs2);

        for (int i = 0, iend = blobs2.size(); i < iend; ++i) {
            rectangle(outContours2, blobs2[i], Scalar(0, 0, 255), 1);
            string txt = to_string(i) + "-" + to_string(blobs2[i].area());
            putText(outContours2, txt, blobs2[i].tl(), 1, 1., Scalar(0,0,255));
        }

        // show hist
//        Mat hist;
//        drawhistogram(grayU, hist);
//        imshow("histogram", hist);

        // show
        Mat tmp1, tmp2, out;
        vconcat(outContours1, outContours2, tmp1);
        vconcat(rgbU, rgbU2, tmp2);
        hconcat(tmp1, tmp2, out);
        putText(out, to_string(i), Point(20, 20), 1, 2., Scalar(0,0,255));
        imshow("result", out);

//        if (write)
//            writer.write(out);

        if (waitKey(50) == 27)
            break;

        currentFrame.copyTo(lastFrame);
    }
    cout << "[Timer] Cost time in dense flow (DIS/Farneback): " << t1 << "/" << t2 << endl;

//    writer.release();
    destroyAllWindows();

#if USE_CAMSHIT || USE_MEANSHIT
    // test Tracking
    bool init = false;
    int channels[] = {0};
    float range_[] = {0, 180};
    const float* range[] = {range_};
    Rect track_window;
    Mat roi_hist;
    // Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    TermCriteria term_crit(TermCriteria::EPS | TermCriteria::COUNT, 10, 1);
    for (size_t i = 0; i < N - 1; ++i) {
        const Mat& frame = vImages[i + 1];
        if (!init) {
            if (vBlobs[i].size() != 1)
                continue; // TODO
            if (vBlobs[i][0].area() < 300)
                continue;
            if (vBlobs[i][0].area() > 0.5 * frame.size().area())
                continue;

            cout << "start idx = " << i << endl;
            track_window = vBlobs[i][0];
            Mat roi = frame(track_window).clone();
            Mat tmp = frame.clone();
            rectangle(tmp, track_window, Scalar(0, 0, 255));
            imshow("roi", tmp);
            waitKey(0);
            Mat hsv_roi, mask;
            cvtColor(roi, hsv_roi, COLOR_BGR2HSV);
            inRange(hsv_roi, Scalar(0, 60, 32), Scalar(180, 255, 255), mask);

            int histSize[] = {180};
            calcHist(&hsv_roi, 1, channels, mask, roi_hist, 1, histSize, range);
            normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
            init = true;
        }
        Mat hsv, dst, trackingResult;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        calcBackProject(&hsv, 1, channels, roi_hist, dst, range);

        // apply meanshift to get the new location
        trackingResult = frame.clone();
#if USE_CAMSHIT
        RotatedRect rot_rect = CamShift(dst, track_window, term_crit);
        Point2f points[4];
        rot_rect.points(points);
        for (int i = 0; i < 4; i++)
            line(trackingResult, points[i], points[(i+1)%4], 255, 2);
#elif USE_MEANSHIT
        meanShift(dst, track_window, term_crit);
        rectangle(trackingResult, track_window, 255, 2);
#endif
        imshow("tracking result", trackingResult);
        int keyboard = waitKey(50);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
#endif

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
