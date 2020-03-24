/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "MotionDetector/BS_MOG2_CV.h"
#include "MotionDetector/FramesDifference.h"
#include "MotionShoter/utility.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h>
#include <jsoncpp/json/json.h>
#include <curl/curl.h>

#define USE_CVMAT_FOR_AIP   1
#define USE_MEDIAN_FILTER_BACKGROUND    1

using namespace cv;
using namespace std;
using namespace ms;

const string app_id = "18679385";
const string aip_key = "12yptwaZPOxoGBfPR0PGYT43";
const string secret_key = "sx8w8l1dzlD2Gt0QAKgZxItRB3uE8DZz";
aip::Bodyanalysis client(app_id, aip_key, secret_key);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
            "{type      t|VIDEO|value input type: VIDEO, LASSESTA, HUAWEI}"
            "{folder    f| |data folder or video file for type LASSESTA/HUAWEI/VIDEO}"
            "{detector  d|fd|value detector: bgd, fd, mog2, vibe, vibe+, flow}"
            "{size      s|5|board size for fd detector}"
            "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
            "{suffix    x|jpg|image suffix for SEQUENCE}"
            "{begin     a|0|start index for image sequence}"
            "{end       e|-1|end index for image sequence}"
            "{write     w|false|write result sequence to a dideo}"
            "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_detector = parser.get<String>("detector");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    double scale = parser.get<double>("scale");
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
    INFO(" - type = " << str_type);
    INFO(" - folder = " << str_folder);
    INFO(" - detector = " << str_detector);

    vector<Mat> vImages;
    InputType inputType;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
        scale = 0.3;
        ReadImageSequence_video(str_folder, vImages, beginIdx, endIdx);
    } else if (str_type == "lasiesta" || str_type == "LASSESTA") {
        inputType = LASIESTA;
        scale = 1;
        vector<Mat> vGTs;
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
    INFO(" - Image size input = " << vImages[0].size());
    resizeFlipRotateImages(vImages, scale);

    BaseMotionDetector* detector;
    if (str_detector == "fd") {
        int size = parser.get<int>("size");
        INFO(" - kernel size = " << size);
        assert(size > 2);
        detector = dynamic_cast<BaseMotionDetector*>(new FramesDifference(2, size, 10));
    } else if (str_detector == "mog2") {
        Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, true);
        detector = dynamic_cast<BaseMotionDetector*>(new BS_MOG2_CV(bs.get()));
    } else {
        ERROR("Unknown input detector for " << str_detector);
        return -1;
    }

    // 把所有输入图像做一个中值滤波, 获得一个不变的背景
    const size_t N = vImages.size();
    Mat medianPano = Mat::zeros(vImages[0].size(), CV_8UC3);
    vector<Mat> vImgs_Y(N); // 每副图像的Y域分量
    for (size_t i = 0; i < N; ++i) {
        Mat imgYUV;
        cvtColor(vImages[i], imgYUV, COLOR_BGR2YUV);
        vector<Mat> channels;
        split(imgYUV, channels);
        vImgs_Y[i] = channels[0];
    }

    // 中值滤波
    for (int y = 0; y < vImages[0].rows; ++y) {
        Vec3b* imgRow = medianPano.ptr<Vec3b>(y);

        for(int x = 0; x < vImages[0].cols; ++x) {
            vector<pair<uchar, uchar>> vLumarAndIndex;
            for (size_t i = 0; i < N; ++i)
                vLumarAndIndex.emplace_back(vImgs_Y[i].at<uchar>(y, x), i);

            sort(vLumarAndIndex.begin(), vLumarAndIndex.end()); // 根据亮度中值决定此像素的值由哪张图像提供
            uchar idx = vLumarAndIndex[N/2].second;
            imgRow[x] = vImages[idx].at<Vec3b>(y, x);
        }
    }
    imwrite("/home/vance/output/ms/fixBackground(medianBlur).jpg", medianPano);
    detector->setFixedBackground(true);
    Mat tmpMask;
    detector->apply(medianPano, tmpMask);   // 喂第一帧

    map<string, string> options;
    options["type"] = "scoremap";


    for (size_t i = 0; i < N; ++i) {
        // 调用SDK获取分割结果
        Mat& image = vImages[i];
        Json::Value result = client.body_seg_cv(image, options);

        // 解析Json结果
        string scoremap = result["scoremap"].asString();    // 灰度图像
        string decode_result = aip::base64_decode(scoremap);

        vector<char> base64_img(decode_result.begin(), decode_result.end());
        Mat mask = imdecode(base64_img, IMREAD_COLOR);
//        imshow("mask", mask);
//        waitKey(0);

    }



    destroyAllWindows();
    return 0;
}
