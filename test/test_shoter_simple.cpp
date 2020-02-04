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
        string fi = prefix + to_string(i + 1) + ".bmp";
        Mat imgi = imread(fi, IMREAD_COLOR);
        if (!imgi.empty())
            vImages.push_back(imgi);
        else
            cerr << "Empty image for " << fi << endl;
    }
    if (vImages.empty()) {
        cerr << "Empty folder!" << endl;
        exit(-1);
    }

    ////////// 1. image stitching
    vector<Mat> toStitch(2);
    toStitch[0] = vImages.front();
    toStitch[1] = vImages.back();

    Mat pano;
    Stitcher stitcher = Stitcher::createDefault(false);
//    Stitcher::Status status = stitcher.stitch(toStitch, pano);

    Stitcher::Status status = stitcher.estimateTransform(toStitch);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    status = stitcher.composePanorama(pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }

    Mat pano_before;
    hconcat(toStitch[0], toStitch[1], pano_before);
    imshow("Front & back image", pano_before);
//    imshow("Stitched Image", pano);
    imwrite("/home/vance/output/pano.bmp", pano);

    cout << " - workScale = " << stitcher.workScale() << endl;
    cout << " - seamEstimationResol = " << stitcher.seamEstimationResol() << endl;
    cout << " - registrationResol = " << stitcher.registrationResol() << endl;
    cout << " - compositingResol = " << stitcher.compositingResol() << endl;
    cout << " - pano size = " << pano.cols << " x " << pano.rows << endl;


    // get Homographies and ROIs
    ms::ImageStitcher is;
    vector<Mat> vHomographies;
    vHomographies.reserve(N - 1);
    vector<Rect2i> vImageROIs, vPanoROIs;
    vImageROIs.reserve(N - 1);
    vPanoROIs.reserve(N - 1);
    for (int i = 1; i < N; ++i) {
        Mat& img = vImages[i];
        Mat H1i = is.computeHomography(img, vImages[0]);
        vHomographies.push_back(H1i);
        // cout << "H " << i << " = " << endl << H1i << endl;

        int tl_x = cvRound(H1i.at<double>(0, 2));
        int tl_y = cvRound(H1i.at<double>(1, 2));

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

    // DF with pano
    Mat pano_gray, img_gray, img_warped;
    cvtColor(pano, pano_gray, COLOR_BGR2GRAY);
    pano_gray.convertTo(pano_gray, CV_32FC1);
    for (int i = 1; i < N; ++i) {
        cvtColor(vImages[i], img_gray, COLOR_BGR2GRAY);
        warpPerspective(img_gray, img_warped, vHomographies[i - 1], pano.size());
        Mat tmp1, tmp2, tmp3;
        cvtColor(img_warped, tmp1, COLOR_GRAY2BGR);
        rectangle(tmp1, vPanoROIs[i - 1], Scalar(0, 0, 255), 2);
        tmp2 = pano.clone();
        rectangle(tmp2, vPanoROIs[i - 1], Scalar(0, 0, 255), 2);
        vconcat(tmp1, tmp2, tmp3);
        imshow("img_warped & pano", tmp3);

        img_warped.convertTo(img_warped, CV_32FC1);
        Mat img_roi, pano_roi, diff;
        img_roi = Mat(img_warped, vPanoROIs[i - 1]);
        pano_roi = Mat(pano_gray, vPanoROIs[i - 1]);
        absdiff(pano_roi, img_roi, diff);
        threshold(diff, diff, 30, 255, CV_THRESH_BINARY);
        diff.convertTo(diff, CV_8UC1);
        imshow("diff", diff);
        //threshold(diff, diff, 25, 255, CV_THRESH_BINARY);

        waitKey(500);
    }


//    std::vector<Point> corners(toStitch.size());
//    std::vector<UMat> masks_warped(toStitch.size());
//    std::vector<UMat> images_warped(toStitch.size());
//    std::vector<Size> sizes(toStitch.size());
//    std::vector<UMat> masks(toStitch.size());

//    std::vector<detail::CameraParams> cameras = stitcher.cameras();
//    detail::SphericalWarper warper = detail::SphericalWarper(1.f);

//    // Prepare image masks
//    for (size_t i = 0; i < toStitch.size(); ++i) {
//        masks[i].create(toStitch[i].size(), CV_8U);
//        masks[i].setTo(Scalar::all(255));
//    }

//    // Warp images and their masks
//    for (size_t i = 0; i < toStitch.size(); ++i) {
//        Mat_<float> K;
//        cameras[i].K().convertTo(K, CV_32F);
//        cout << K << endl;

//        corners[i] = warper.warp(toStitch[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
//        sizes[i] = images_warped[i].size();

//        warper.warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
//    }
//    imshow("warp1", images_warped[0]);
//    imshow("warp2", images_warped[1]);

    ////////// 2. motion detection


    waitKey(0);

    return 0;
}
