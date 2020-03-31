#include "utility.h"
#include "ImageStitcher/ImageStitcher.h"

//#define USE_CUSTOM_STITCHER 0

//#if !USE_CUSTOM_STITCHER
#include <opencv2/stitching.hpp>
//#include <opencv2/stitching/detail/blenders.hpp>
//#endif


using namespace ms;
using namespace std;
using namespace cv;
namespace cvd = cv::detail;

const int delta = 3;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
        "{type      t|VIDEO|value input type: LASIESTA, HUAWEI, SEQUENCE}"
        "{folder    f| |data folder or video file for type LASIESTA/HUAWEI/SEQUENCE/VIDEO}"
        "{suffix    x|jpg|image suffix for SEQUENCE}"
        "{begin     a|0|start index for image sequence}"
        "{end       e|-1|end index for image sequence}"
        "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_folder = parser.get<String>("folder");
    const int beginIdx = parser.get<int>("begin");
    const int endIdx = parser.get<int>("end");
    const int num = endIdx - beginIdx + 1;
    INFO(" - type = " << str_type);
    INFO(" - folder = " << str_folder);
    INFO(" - beginIdx = " << beginIdx);
    INFO(" - endIdx = " << endIdx);

    vector<Mat> vImages;
    if (str_type == "video" || str_type == "VIDEO") {
        ReadImageSequence_video(str_folder, vImages, beginIdx, num);
    }  else if (str_type == "lasiesta" || str_type == "LASIESTA") {
        vector<Mat> gts;
        ReadImageSequence_lasiesta(str_folder, vImages, gts, beginIdx, num);
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        ReadImageSequence_huawei(str_folder, vImages, beginIdx, num);
    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
        const String str_suffix = parser.get<String>("suffix");
        INFO(" - suffix = " << str_suffix);
        ReadImageSequence(str_folder, str_suffix, vImages, beginIdx, num);
    } else {
        ERROR("[Error] Unknown input type for " << str_type);
        return -1;
    }
    INFO(" - num of input images = " << vImages.size());
    if (num < 2) {
        ERROR("Too less images input!");
        exit(-1);
    }

    /// start stitching
    INFO(endl << "\tStitching... This will take a while...");

    TickMeter timer;
    timer.start();

    /// 1. opencv stitcher
    Ptr<cv::Stitcher> stitcher1 = cv::Stitcher::create(Stitcher::PANORAMA);
    stitcher1->setWarper(makePtr<PlaneWarper>());
    stitcher1->setExposureCompensator(makePtr<cvd::NoExposureCompensator>());

    Mat pano1;
    Stitcher::Status status = stitcher1->stitch(vImages, pano1);
    if (status != Stitcher::OK) {
        ERROR("Can't stitch images (cv), error code = " << status);
        return -1;
    }

    timer.stop();
    INFO(" - Time cost in stitching = " << timer.getTimeSec() << "s");
    INFO(" - Image size = " << vImages[0].size());
    INFO(" - Pano size = " << pano1.size());

    NamedLargeWindow("Result Pano CV");
    imshow("Result Pano CV", pano1);
    imwrite("/home/vance/output/result_pano_cv.jpg", pano1);
    INFO("Saving result1 to /home/vance/output/result_pano_cv.jpg");


    /// 2. custom stitcher
    Ptr<ImageStitcher> stitcher2 = ImageStitcher::create(ImageStitcher::ORB, ImageStitcher::BF);
    stitcher2->setScales(0.25);

    Mat pano2;
    ImageStitcher::Status status2 = stitcher2->stitch(vImages, pano2);
    if (status2 != ImageStitcher::OK) {
        ERROR("Can't stitch2 images (custom), error code = " << status);
        return -1;
    }

//    Mat H21 = is.computeHomography(image1, image2);

//    // 获取变换后的图像左上角与右下角坐标, 确定画布 大小
//    const Mat tl_mat = (Mat_<double>(3, 1) << 0, 0, 1);
//    const Mat tr_mat = (Mat_<double>(3, 1) << image1.cols, 0, 1);
//    const Mat bl_mat = (Mat_<double>(3, 1) << 0, image1.rows, 1);
//    const Mat br_mat = (Mat_<double>(3, 1) << image1.cols, image1.rows, 1);
//    Mat tlw_mat = H21 * tl_mat;
//    Mat trw_mat = H21 * tr_mat;
//    Mat blw_mat = H21 * bl_mat;
//    Mat brw_mat = H21 * br_mat;
//    Point2d tlw(tlw_mat.at<double>(0) / tlw_mat.at<double>(2), tlw_mat.at<double>(1) / tlw_mat.at<double>(2));
//    Point2d trw(trw_mat.at<double>(0) / trw_mat.at<double>(2), trw_mat.at<double>(1) / trw_mat.at<double>(2));
//    Point2d blw(blw_mat.at<double>(0) / blw_mat.at<double>(2), blw_mat.at<double>(1) / blw_mat.at<double>(2));
//    Point2d brw(brw_mat.at<double>(0) / brw_mat.at<double>(2), brw_mat.at<double>(1) / brw_mat.at<double>(2));
//    INFO("4 conners of warped image is " << tlw << ", " << trw << ", " << blw << ", " << brw);

//    const double minx = min(min(tlw.x, blw.x), 0.);
//    const double miny = min(min(tlw.y, trw.y), 0.);
//    const double maxx = max(max(trw.x, brw.x), (double)image1.cols);
//    const double maxy = max(max(blw.y, brw.y), (double)image1.rows);
//    INFO("canvas rect = " << minx << ", " << miny << ", " << maxx << ", " << maxy);

//    Mat H21_offset = H21.clone();
//    H21_offset.at<double>(0, 2) -= tlw.x;
//    H21_offset.at<double>(0, 2) -= tlw.y;
//    INFO("H21 + offset = " << endl << H21_offset);

//    Mat img_warped, img_warped_gray;
//    Size canvasSize(maxx - minx, maxy - miny);
//    warpPerspective(image1, img_warped, H21_offset, canvasSize);
//    cvtColor(img_warped, img_warped_gray, COLOR_BGR2GRAY);
//    imshow("img_warped", img_warped);

//    Mat pano2 = img_warped.clone();
//    int offset_x = cvFloor(abs(minx));
//    int offset_y = cvFloor(abs(miny));
//    image2.copyTo(
//        pano2.rowRange(offset_y, offset_y + image2.rows).colRange(offset_x, offset_x + image2.cols));

    NamedLargeWindow("Result Pano Custom");
    imshow("Result Pano Custom", pano2);
    imwrite("/home/vance/output/result_pano_custom.jpg", pano2);
    INFO("Saving result2 to /home/vance/output/result_pano_custom.jpg");

    waitKey(0);

    return 0;
}
