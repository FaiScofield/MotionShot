#include <iostream>
#include "ImageStitcher/ImageStitcher.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <string>

using namespace std;
using namespace cv;

const double g_scale = 1.;    // 0.15

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Parameters: <image_sequence_folder> <image_count>" << endl;
        exit(-1);
    }

    const string prefix(argv[1]);
    const int N = atoi(argv[2]);

    vector<Mat> vImages;
    vImages.reserve(N);
    for (int i = 0; i < N; ++i) {
        string fi = prefix + to_string(i) + ".bmp";    // jpg
        Mat imgi = imread(fi, IMREAD_COLOR);
        if (!imgi.empty()) {
            Mat tmp;
            resize(imgi, tmp, Size(imgi.cols * g_scale, imgi.rows * g_scale));
            vImages.push_back(tmp);
        } else {
            cerr << "Empty image for " << fi << endl;
        }
    }
    if (vImages.empty()) {
        cerr << "Empty folder!" << endl;
        exit(-1);
    }

    ////////// 1. image stitching
    int M = N > 10 ? N / 2 : N;
    vector<Mat> toStitch;
    toStitch.reserve(M);
    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0)
            toStitch.push_back(vImages[i]);
    }

    Mat pano = vImages.back().clone();

    // get Homographies and ROIs
    ms::ImageStitcher msStitcher;
    vector<Mat> vHomographies;
    vHomographies.reserve(N - 1);
    vector<Rect2i> vImageROIs, vPanoROIs;
    vImageROIs.reserve(N - 1);
    vPanoROIs.reserve(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        Mat& img = vImages[i];
        Mat Hli = msStitcher.computeHomography(img, vImages.back());
        vHomographies.push_back(Hli);
        // cout << "H " << i << " to last = " << endl << Hli << endl;

        int tl_x = cvRound(Hli.at<double>(0, 2));
        int tl_y = cvRound(Hli.at<double>(1, 2));

        int roi_tl_x = tl_x < 0 ? -tl_x : 0;
        int roi_tl_y = tl_y < 0 ? -tl_y : 0;
        int roi_br_x = tl_x + img.cols > pano.cols ? pano.cols - tl_x : img.cols;
        int roi_br_y = tl_y + img.rows > pano.rows ? pano.rows - tl_y : img.rows;
        Rect2i imgRoi(Point2i(roi_tl_x, roi_tl_y), Point2i(roi_br_x, roi_br_y));
        vImageROIs.push_back(imgRoi);

        int pano_roi_tl_x = tl_x < 0 ? 0 : tl_x;
        int pano_roi_tl_y = tl_y < 0 ? 0 : tl_y;
        int pano_roi_br_x = tl_x + img.cols > pano.cols ? pano.cols : tl_x + img.cols;
        int pano_roi_br_y = tl_y + img.rows > pano.rows ? pano.rows : tl_y + img.rows;
        Rect2i panoRoi(Point2i(pano_roi_tl_x, pano_roi_tl_y), Point2i(pano_roi_br_x, pano_roi_br_y));
        vPanoROIs.push_back(panoRoi);

        assert(imgRoi.area() == panoRoi.area());
    }

    ////////// 2. DF with pano
    vector<Mat> vMasks;
    vector<Rect> vBlobs;
    Mat pano_gray, img_gray, img_warped;
    cvtColor(pano, pano_gray, COLOR_BGR2GRAY);
    pano_gray.convertTo(pano_gray, CV_32FC1);
    for (int i = 0; i < N - 1; ++i) {
        cvtColor(vImages[i], img_gray, COLOR_BGR2GRAY);
        warpPerspective(img_gray, img_warped, vHomographies[i], pano.size());
        Mat tmp1, tmp2, tmp3;
        cvtColor(img_warped, tmp1, COLOR_GRAY2BGR);
        rectangle(tmp1, vPanoROIs[i], Scalar(0, 0, 255), 2);
        tmp2 = pano.clone();
        rectangle(tmp2, vPanoROIs[i], Scalar(0, 0, 255), 2);
        vconcat(tmp1, tmp2, tmp3);
        imshow("img_warped & pano", tmp3);

        img_warped.convertTo(img_warped, CV_32FC1);
        Mat img_roi, pano_roi, diff;
        img_roi = Mat(img_warped, vPanoROIs[i]);
        pano_roi = Mat(pano_gray, vPanoROIs[i]);
        absdiff(pano_roi, img_roi, diff);
        threshold(diff, diff, 15, 255, THRESH_BINARY);
        diff.convertTo(diff, CV_8UC1);
        cv::erode(diff, diff, cv::Mat());   // 腐蚀
        cv::dilate(diff, diff, cv::Mat());  // 膨胀
        cv::dilate(diff, diff, cv::Mat());
        cv::erode(diff, diff, cv::Mat());
        //imshow("diff", diff);

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
            if (blobi.area() < 10000)
                continue;
            if (blobi.area()  == 19968)
                continue;
            if (blobi.area()  == 20176)
                continue;
            if (blobi.area() > maxArea)
                maxArea = blobi.area();
            blobs.push_back(blobi);
        }
        cout << "blobs size: " << blobs.size() << endl;
        vBlobs.push_back(blobs.front());
//        for (auto it = blobs.begin(), iend = blobs.end(); it != iend; it++) {
//            int th = max(int(maxArea * 0.2), 200);
//            if (it->area() < th)
//                blobs.erase(it);
//            if (it->area() == 19968)
//                blobs.erase(it);
//        }

        Mat blobOutput = pano.clone();
        Mat mask = Mat::zeros(img_gray.size(), CV_8UC1);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(blobOutput, blobs[i], CV_RGB(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(blobOutput, txt, blobs[i].tl(), 1, 1., Scalar(0,0,255));

            const Point tl = blobs[i].tl();
            const Point br = blobs[i].br();
            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }
        Mat tmp4;
        vconcat(diff, mask, tmp4);
        imshow("diff and mask i", tmp4);
        vMasks.push_back(mask);

        Mat tmp5;
        vconcat(contOutput, blobOutput, tmp5);
        imshow("contours and blobs", tmp5);

        waitKey(50);
    }

    ////////// 3. blending
    Mat blendImage = pano.clone();
    for (int i = 0; i < vMasks.size(); ++i) {
        if (i % 10 != 0)
            continue;

        Mat blend, tmp1, tmp2;
        Point p = (vBlobs[i].tl() + vBlobs[i].br()) * 0.5;
        seamlessClone(vImages[i], blendImage, vMasks[i], p, blendImage, MIXED_CLONE);
//        cvtColor(vMasks[i], tmp1, COLOR_GRAY2BGR);
//        vconcat(tmp1, blend, tmp2);
//        imshow("blend", tmp2);
//        waitKey(0);
    }
    imshow("blending result", blendImage);
    imwrite("/home/vance/output/blendImage.bmp", blendImage);


    waitKey(0);

    return 0;
}
