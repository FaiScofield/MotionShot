#include "ImageStitcher/ImageStitcher.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

const double g_scale = 0.15;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Parameters: <image1> <image2> or <image_sequence_folder> <image_count>" << endl;
        exit(-1);
    }

    vector<Mat> vImages;

    Mat image1 = imread(argv[1]);
    Mat image2 = imread(argv[2]);
    if (image1.empty() || image2.empty()) {
        const string prefix(argv[1]);
        const int N = atoi(argv[2]);
        if (N < 2) {
            cerr << "Need more images" << endl;
            exit(-1);
        }

        vImages.reserve(N);
        for (int i = 0; i < N; ++i) {
            // string fi = prefix + to_string(i + 1) + ".bmp";
            string fi = prefix + to_string(i) + ".jpg";
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

        image1 = vImages.front();
        image2 = vImages.back();
    } else {
        Mat tmp1, tmp2;
        resize(image1, tmp1, Size(image1.cols * g_scale, image1.rows * g_scale));
        resize(image2, tmp2, Size(image2.cols * g_scale, image2.rows * g_scale));
        vImages.push_back(image1);
        vImages.push_back(image2);
    }

    Mat fl;
    hconcat(image1, image2, fl);
    imshow("first and last images", fl);
    imwrite("/home/vance/output/first_and_last.bmp", fl);

//    int m = vImages.size() / 3;
//    vector<Mat> toStitch;
//    toStitch.reserve(m);
//    for (int k = 0; k < vImages.size(); ++k) {
//        if (k % 3 == 0 )
//            toStitch.push_back(vImages[k]);
//    }

    TickMeter tm;
    tm.start();

    Mat pano;
    // Stitcher stitcher = Stitcher::createDefault(false);
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, false);
    Stitcher::Status status = stitcher->stitch(vImages/*toStitch*/, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    tm.stop();
    double time = tm.getTimeSec() / tm.getCounter();

//    cout << "Tatal time in image stitching = " << time << "s" << endl;
//    cout << "Image size = " << image1.cols << " x " << image1.rows << endl;
//    cout << "Pano size = " << pano.cols << " x " << pano.rows << endl;

//    imshow("Result Pano", pano);
//    imwrite("/home/vance/output/result_pano_cv.bmp", pano);
/*
    // test
    ms::ImageStitcher is;
    Mat H21 = is.computeHomography(image1, image2);
    // 获取变换后的图像左上角与右下角坐标, 确定画布 大小
    const Mat tl_mat = (Mat_<double>(3, 1) << 0, 0, 1);
    const Mat tr_mat = (Mat_<double>(3, 1) << image1.cols, 0, 1);
    const Mat bl_mat = (Mat_<double>(3, 1) << 0, image1.rows, 1);
    const Mat br_mat = (Mat_<double>(3, 1) << image1.cols, image1.rows, 1);
    Mat tlw_mat = H21 * tl_mat;
    Mat trw_mat = H21 * tr_mat;
    Mat blw_mat = H21 * bl_mat;
    Mat brw_mat = H21 * br_mat;
    Point2d tlw(tlw_mat.at<double>(0) / tlw_mat.at<double>(2), tlw_mat.at<double>(1) / tlw_mat.at<double>(2));
    Point2d trw(trw_mat.at<double>(0) / trw_mat.at<double>(2), trw_mat.at<double>(1) / trw_mat.at<double>(2));
    Point2d blw(blw_mat.at<double>(0) / blw_mat.at<double>(2), blw_mat.at<double>(1) / blw_mat.at<double>(2));
    Point2d brw(brw_mat.at<double>(0) / brw_mat.at<double>(2), brw_mat.at<double>(1) / brw_mat.at<double>(2));
    cout << "4 conners of warped image is " << tlw << ", " << trw << ", " << blw << ", " << brw << endl;

    const double minx = min(min(tlw.x, blw.x), 0.);
    const double miny = min(min(tlw.y, trw.y), 0.);
    const double maxx = max(max(trw.x, brw.x), (double)image1.cols);
    const double maxy = max(max(blw.y, brw.y), (double)image1.rows);
    cout << "canvas rect = " << minx << ", " << miny << ", " << maxx << ", " << maxy << endl;

    Mat H21_offset = H21.clone();
    H21_offset.at<double>(0, 2) -= tlw.x;
    H21_offset.at<double>(0, 2) -= tlw.y;
    cout << "H21 + offset = " << endl << H21_offset << endl;

    Mat img_warped, img_warped_gray;
    Size canvasSize(maxx - minx, maxy - miny);
    warpPerspective(image1, img_warped, H21_offset, canvasSize);
    cvtColor(img_warped, img_warped_gray, COLOR_BGR2GRAY);
//    imshow("img_warped", img_warped);

    Mat pano2 = img_warped.clone();
    int offset_x = cvFloor(abs(minx));
    int offset_y = cvFloor(abs(miny));
    image2.copyTo(
        pano2.rowRange(offset_y, offset_y + image2.rows).colRange(offset_x, offset_x + image2.cols));
    imshow("Result Pano2", pano2);
    imwrite("/home/vance/output/result_pano_H.bmp", pano2);


    waitKey(0);
*/
    return 0;
}
