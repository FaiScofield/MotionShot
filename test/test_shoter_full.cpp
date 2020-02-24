#include <iostream>
#include "utility.h"
#include "ImageStitcher/ImageStitcher.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <string>

using namespace ms;
using namespace std;
using namespace cv;

const double g_scale = 0.15;    // 0.15, 1.
#define concat hconcat          // hconcat, vconcat

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
        "{type      t|sequence|input type for one of VIDEO, LASSESTA, HUAWEI, SEQUENCE, TWO_IMAGES}"
        "{folder    f| |data folder or video file for type LASSESTA/HUAWEI dataset or VIDEO or SEQUENCE}"
        "{suffix    s|jpg|image suffix for type SEQUENCE}"
        "{img1      1| |the first image for type TWO_IMAGES}"
        "{img2      2| |the second image for type TWO_IMAGES}"
        "{start     a|0|start index for image sequence}"
        "{num       n|0|number to process for image sequence}"
        "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
        "{delta     d|1|delta frames to stitch between image sequence}"
        "{flip      p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
        "{rotate    r|-1|rotate image for type VIDEO, r = cv::RotateFlags(0, 1, 2)}"
        "{help      h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    InputType inputType;
    const String str_type = parser.get<String>("type");
    cout << " - type = " << str_type << endl;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
    }  else if (str_type == "lasiesta" || str_type == "LASSESTA") {
        inputType = LASIESTA;
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
        inputType = SEQUENCE;
    } else if (str_type == "two_images" || str_type == "TWO_IMAGES") {
        inputType = TWO_IMAGES;
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    String str_folder = parser.get<String>("folder");
    cout << " - folder = " << str_folder << endl;
    int start = parser.get<int>("start");
    int num = parser.get<int>("num");
    int delta = parser.get<int>("delta");
    double scale = parser.get<double>("scale");
    int flip = parser.get<int>("flip");
    int rotate = parser.get<int>("rotate");
    assert(delta >= 1);

    //// read images
    vector<Mat> vImages, toStitch;
    Mat image1, image2;
    String str_suffix;
    if (inputType == LASIESTA) {
        vector<Mat> gts;
        ReadImageSequence_lasiesta(str_folder, vImages, gts, start, num);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages, start, num);
        scale = 0.15;
    } else if (inputType == VIDEO) {
        ReadImageSequence_video(str_folder, vImages, start, num);
    } else if (inputType == SEQUENCE) {
        str_suffix = parser.get<String>("suffix");
        cout << " - suffix = " << str_suffix << endl;
        ReadImageSequence(str_folder, str_suffix, vImages, start, num);
    } else {
        assert(inputType == TWO_IMAGES);
        Mat img1 = imread(parser.get<String>("img1"));
        Mat img2 = imread(parser.get<String>("img2"));
        if (img1.empty() || img2.empty()) {
            cerr << "[Error] Empty image for img1 or img2!" << endl;
            return -1;
        }
        vImages.push_back(img1);
        vImages.push_back(img2);
    }
    if (num <= 0)
        num = vImages.size();
    else
        num = min(num, static_cast<int>(vImages.size()));
    cout << " - start = " << start << endl;
    cout << " - num = " << num << endl;
    cout << " - delta = " << delta << endl;
    cout << " - scale = " << scale << endl;
    cout << " - flip flag = " << flip << endl;
    cout << " - rotate flag = " << rotate << endl;
    assert(scale > 0);
    if (num < 2) {
        cerr << "Too less imagesinput!" << endl;
        exit(-1);
    }
    resizeFlipRotateImages(vImages, scale, flip, rotate);

    // toStitch
    image1 = vImages.front();
    image2 = vImages.back();
    toStitch.push_back(image1);
    toStitch.push_back(image2);

    Mat fl;
    hconcat(image1, image2, fl);
    imshow("first and last images", fl);

    ////////// 1. image stitching
    cout << "Stitching... This will take a while..." << endl;
    cout << " - Tatal image count = " << toStitch.size() << endl;
    TickMeter tm;
    tm.start();

    Mat pano;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
    Stitcher::Status status = stitcher->stitch(toStitch, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    tm.stop();
    double time = tm.getTimeSec() / tm.getCounter();
    cout << " - Time cost in stitching = " << time << "s" << endl;
    cout << " - Image size = " << image1.cols << " x " << image1.rows << endl;
    cout << " - Pano size = " << pano.cols << " x " << pano.rows << endl;
    imshow("Result Pano", pano);

    cout << " - workScale = " << stitcher->workScale() << endl;
    cout << " - seamEstimationResol = " << stitcher->seamEstimationResol() << endl;
    cout << " - registrationResol = " << stitcher->registrationResol() << endl;
    cout << " - compositingResol = " << stitcher->compositingResol() << endl;

    ////////// 2. DF with pano
    // get Homographies and ROIs
    vector<Rect> vBlobs;
    vector<Mat> vMasks;
    ms::ImageStitcher msStitcher;
    vector<Mat> vHomographies;
    vHomographies.reserve(num/10);
//    vector<Rect2i> vImageROIs, vPanoROIs;
//    vImageROIs.reserve(num/10);
//    vPanoROIs.reserve(num/10);
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < num - 1; ++i) {
        if (i % 10 != 0)
            continue;

        Mat& imgi = vImages[i];
        Mat Hi = msStitcher.computeHomography(imgi, pano);
        vHomographies.push_back(Hi);

        Mat panof, img_warped, warpf, tmp, diff;
        warpPerspective(imgi, img_warped, vHomographies[i], pano.size());
        vconcat(img_warped, pano, tmp);
        imshow("img_warped & pano", tmp);

        img_warped.convertTo(warpf, CV_32FC1);
        pano.convertTo(panof, CV_32FC1);
        absdiff(panof, warpf, diff);
        threshold(diff, diff, 30, 255, THRESH_BINARY);
        diff.convertTo(diff, CV_8UC1);
        cv::erode(diff, diff, kernel);   // 腐蚀
        cv::dilate(diff, diff, kernel);  // 膨胀
        cv::dilate(diff, diff, kernel);
        cv::erode(diff, diff, kernel);
        imshow("diff", diff);

        // find contours
        vector<vector<Point>> contours;
        findContours(diff, contours, RETR_EXTERNAL,
                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE

        Mat contOutput = pano.clone();
        drawContours(contOutput, contours, -1, Scalar(0, 255, 0), -1);
        vector<Rect> blobs;
        int maxArea = 0;
        for (int i = 0, iend = contours.size(); i < iend; ++i) {
            Rect blobi = boundingRect(contours[i]);
//            if (blobi.area() < 10000)
//                continue;
//            if (blobi.area()  == 19968)
//                continue;
//            if (blobi.area()  == 20176)
//                continue;
            if (blobi.area() > maxArea)
                maxArea = blobi.area();
            blobs.push_back(blobi);
        }
        cout << "blobs size: " << blobs.size() << endl;
        vBlobs.push_back(blobs.front());
        for (auto it = blobs.begin(), iend = blobs.end(); it != iend; it++) {
            int th = max(int(maxArea * 0.2), 2000);
            if (it->area() < th)
                blobs.erase(it);
        }

        Mat blobOutput = pano.clone();
        Mat mask = Mat::zeros(imgi.size(), CV_8UC1);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(blobOutput, blobs[i], CV_RGB(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(blobOutput, txt, blobs[i].tl(), 1, 1., Scalar(0,0,255));

            const Point tl = blobs[i].tl();
            const Point br = blobs[i].br();
            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }
        Mat tmp4;
        concat(diff, mask, tmp4);
        imshow("diff and mask i", tmp4);
        vMasks.push_back(mask);

        Mat tmp5;
        concat(contOutput, blobOutput, tmp5);
        imshow("contours and blobs", tmp5);

        waitKey(100);
    }

    ////////// 3. blending
    Mat blendImage = pano.clone();
    for (size_t i = 0; i < vMasks.size(); ++i) {
        if (i % 10 != 0)
            continue;

        Mat blend, tmp1, tmp2;
        Point p = (vBlobs[i].tl() + vBlobs[i].br()) * 0.5;
        seamlessClone(vImages[i], blendImage, vMasks[i], p, blendImage, MIXED_CLONE);
        cvtColor(vMasks[i], tmp1, COLOR_GRAY2BGR);
        vconcat(tmp1, blend, tmp2);
        imshow("blend", tmp2);
        waitKey(0);
    }
    imshow("blending result", blendImage);
    imwrite("/home/vance/output/blendImage.bmp", blendImage);


    waitKey(0);

    return 0;
}
