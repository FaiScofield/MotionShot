/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "OpticalFlower.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/videoio/videoio.hpp>
#include <opencv2/video/video.hpp>

using namespace cv;
using namespace std;
using namespace ms;

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{type      t|VIDEO|value input type: VIDEO, LASISESTA, HUAWEI}"
               "{folder    f| |data folder or video file for type LASISESTA/HUAWEI/VIDEO}"
               "{dense     d|true|dense }"
               "{size      s|5|kernel size for Binary Image}"
               "{showGT    g|false|if show ground for type DATASET}"
               "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
               "{flip      p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
               "{rotate    r|-1|rotate image for type VIDEO, r = RotateFlags(0, 1, 2)}"
               "{write     w|false|write result sequence to a dideo}"
               "{help      h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
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
    } else if (str_type == "lasisesta" || str_type == "LASISESTA") {
        inputType = LASISESTA;
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
    if (inputType == LASISESTA) {
        ReadImageSequence_lasisesta(str_folder, vImages, vGTs);
    } else if (inputType == HUAWEI) {
        ReadImageSequence_huawei(str_folder, vImages);
        // scale = 0.15;
    } else if (inputType == VIDEO) {
        ReadImagesFromVideo(str_folder, vImages);
    }
    // scale
    const size_t N = vImages.size();
    if (abs(scale - 1) > 1e-9) {
        cout << " - scale = " << scale << endl;
        vector<Mat> vImgResized(N);
        Size imgSize = vImages[0].size();
        imgSize.width *= scale;
        imgSize.height *= scale;
        for (size_t i = 0; i < N; ++i) {
            Mat imgi;
            resize(vImages[i], imgi, imgSize);
            vImgResized[i] = imgi;
        }
        vImages.swap(vImgResized);
    }
    // flip or rotate
    if (flip != 0) {
        cout << " - flip = " << flip << endl;
        vector<Mat> vImgFlipped(N);
        for (size_t i = 0; i < N; ++i) {
            Mat imgi;
            cv::flip(vImages[i], imgi, flip);
            vImgFlipped[i] = imgi;
        }
        vImages.swap(vImgFlipped);
    } else if (rotate >= 0) {
        cout << " - rotate = " << rotate << endl;
        vector<Mat> vImgRotated(N);
        for (size_t i = 0; i < N; ++i) {
            Mat imgi;
            cv::rotate(vImages[i], imgi, rotate);
            vImgRotated[i] = imgi;
        }
        vImages.swap(vImgRotated);
    }

    /// detect moving frontground
    bool dense = parser.get<bool>("dense");
#ifdef USE_OPENCV4
    Ptr<DISOpticalFlow> detector1 = DISOpticalFlow::create();
    Ptr<FarnebackOpticalFlow> detector2 = FarnebackOpticalFlow::create();
#else
    Ptr<VariationalRefinement> detector1 = VariationalRefinement::create();
    Ptr<FarnebackOpticalFlow> detector2 = FarnebackOpticalFlow::create();
#endif
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M','J','P','G'), 10,
                       Size(vImages[0].cols * 2, vImages[0].rows * 2));

    int size = parser.get<int>("size");
    cout << " - kernel size = " << size << endl;
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    Mat lastFrame, currentFrame;
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        cvtColor(vImages[i], currentFrame, COLOR_BGR2GRAY);
        if (i == 0) {
            currentFrame.copyTo(lastFrame);
            continue;
        }

        Mat flow1, flow1_uv[2], mag1, ang1, hsv1, hsv_split1[3], bgr1;
        detector1->calc(lastFrame, currentFrame, flow1); // get flow type CV_32FC2
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

        // test
        Mat mask(bgr1.size(), CV_8UC1, Scalar(255)), grayU;
        cvtColor(rgbU, grayU, COLOR_BGR2GRAY);
        mask = mask - grayU;
        threshold(mask, mask, 30, 255, THRESH_BINARY); // THRESH_OTSU, THRESH_BINARY
        erode(mask, mask, kernel);
        dilate(mask, mask, kernel);
        dilate(mask, mask, kernel);
        Mat mask_color;
        cvtColor(mask, mask_color, COLOR_GRAY2BGR);
//        imshow("DISOpticalFlow", mask);

        /*Farneback*/
        Mat flow2, flow2_uv[2], mag2, ang2, hsv2, hsv_split2[3], bgr2;
        detector2->calc(lastFrame, currentFrame, flow2);
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
//        imshow("FlowFarneback", rgbU2);

        // find contours
        Mat frame_contours = vImages[i].clone();
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL,
                     CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE
        drawContours(frame_contours, contours, -1, Scalar(0, 255, 0), 2);

        // calculate blobs
        vector<Rect> blobs;
        int maxArea = 0;
        for (int i = 0, iend = contours.size(); i < iend; ++i) {
            Rect blobi = boundingRect(contours[i]);
//            if (blobi.area() < 10000)
//                continue;
            if (blobi.area() > maxArea)
                maxArea = blobi.area();
            blobs.push_back(blobi);
        }
//        cout << " - max blob area: " << maxArea << endl;

        Mat diff_blobs, tmp3, output;
        cvtColor(mask, diff_blobs, COLOR_GRAY2BGR);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(diff_blobs, blobs[i], Scalar(0, 255, 0), 1);
            rectangle(frame_contours, blobs[i], Scalar(0, 0, 255), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0,0,255));

//            const Point tl = blobs[i].tl();
//            const Point br = blobs[i].br();
//            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }
//        vconcat(frame_contours, diff_blobs, tmp3);
//        if (showGT && !vGTs.empty())
//            vconcat(tmp3, vGTs[i], output);
//        else
//            output = tmp3;
//        imshow("result", output);

        // show
        Mat tmp1, tmp2, out;
        vconcat(frame_contours, diff_blobs, tmp1);
        vconcat(rgbU, rgbU2, tmp2);
        hconcat(tmp1, tmp2, out);
        imshow("result", out);
//        waitKey(50);

        if (write)
            writer.write(out);

        if (waitKey(50) == 27)
            break;

        currentFrame.copyTo(lastFrame);
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
