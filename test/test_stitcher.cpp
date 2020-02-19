#include "utility.h"
#include "ImageStitcher/ImageStitcher.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <boost/filesystem.hpp>

using namespace ms;
using namespace std;
using namespace cv;
namespace bf = boost::filesystem;

const int delta = 3;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
        "{type      t|VIDEO|value input type: VIDEO, LASISESTA, HUAWEI, SEQUENCE, TWO_IMAGES}"
        "{folder    f| |data folder or video file for type LASISESTA/HUAWEI/SEQUENCE/VIDEO}"
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
    }  else if (str_type == "lasisesta" || str_type == "LASISESTA") {
        inputType = LASISESTA;
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
    if (inputType == LASISESTA) {
        vector<Mat> gts;
        ReadImageSequence_lasisesta(str_folder, vImages, gts, start, num);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages, start, num);
        //scale = 0.15;
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
    if (inputType == TWO_IMAGES) {
        assert(vImages.size() == 2);
        toStitch = vImages;
    } else {
        toStitch.reserve(num / delta);
        for (int i = 0; i < num; ++i) {
//            imshow("sequence", vImages[i]);
//            waitKey(25);
            if (i % delta == 0)
                toStitch.push_back(vImages[i]);
        }
    }
    destroyAllWindows();
    if (toStitch.size() < 2) {
        cerr << "Too less images or too big delta(" << delta << ")!" << endl;
        exit(-1);
    }

    Mat fl;
    hconcat(image1, image2, fl);
//    namedWindow("first and last images", WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
    imshow("first and last images", fl);
    imwrite("/home/vance/output/first_and_last.bmp", fl);

    /// start stitching
    cout << "Stitching... This will take a while..." << endl;
    cout << " - Tatal image count = " << toStitch.size() << endl;

    TickMeter tm;
    tm.start();

    Mat pano;
//    ImageStitcher* stitcher = new ImageStitcher();
//    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
    // stitcher->setWarper(); //! TODO 投影方式修改
    Stitcher::Status status = stitcher->stitch(toStitch, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }
    tm.stop();
    double time = tm.getTimeSec() / tm.getCounter();
    cout << " - Time cost in stitching = " << time << "s" << endl;
    cout << " - Image size = " << image1.size() << endl;
    cout << " - Pano size = " << pano.size() << endl;

//    namedWindow("Result Pano", WINDOW_FREERATIO | WINDOW_GUI_EXPANDED);
    imshow("Result Pano", pano);
    string fileOut = "/home/vance/output/result_pano_cv_" + to_string(vImages.size()) + "-"
                     + to_string(toStitch.size()) + ".bmp";
    imwrite(fileOut, pano);

    // test
    ms::ImageStitcher is;
    Mat pano2;
    if (!is.stitch(image1, image2, pano2))
        return -1;

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
//    cout << "4 conners of warped image is " << tlw << ", " << trw << ", " << blw << ", " << brw << endl;

//    const double minx = min(min(tlw.x, blw.x), 0.);
//    const double miny = min(min(tlw.y, trw.y), 0.);
//    const double maxx = max(max(trw.x, brw.x), (double)image1.cols);
//    const double maxy = max(max(blw.y, brw.y), (double)image1.rows);
//    cout << "canvas rect = " << minx << ", " << miny << ", " << maxx << ", " << maxy << endl;

//    Mat H21_offset = H21.clone();
//    H21_offset.at<double>(0, 2) -= tlw.x;
//    H21_offset.at<double>(0, 2) -= tlw.y;
//    cout << "H21 + offset = " << endl << H21_offset << endl;

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
    imshow("Result Pano2", pano2);
    imwrite("/home/vance/output/result_pano_H.bmp", pano2);


    waitKey(0);

    return 0;
}
