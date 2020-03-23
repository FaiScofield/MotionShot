/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "FramesDifference.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

#include <Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h>
#include <jsoncpp/json/json.h>
#include <curl/curl.h>

#define WATERSHED 0
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
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;
    cout << " - detector = " << str_detector << endl;

//    vector<Mat> vImages, vGTs;
//    InputType inputType;
//    if (str_type == "video" || str_type == "VIDEO") {
//        inputType = VIDEO;
//        scale = 0.3;
//        ReadImageSequence_video(str_folder, vImages, beginIdx, endIdx);
//    } else if (str_type == "lasiesta" || str_type == "LASSESTA") {
//        inputType = LASIESTA;
//        scale = 1;
//        ReadImageSequence_lasiesta(str_folder, vImages, vGTs, beginIdx, endIdx);
//    } else if (str_type == "huawei" || str_type == "HUAWEI") {
//        inputType = HUAWEI;
//        scale = 0.1;
//        ReadImageSequence_huawei(str_folder, vImages, beginIdx, endIdx);
//    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
//        inputType = SEQUENCE;
//        scale = 0.5;
//        String suffix = parser.get<String>("suffix");
//        ReadImageSequence(str_folder, suffix, vImages, beginIdx, endIdx);
//    } else {
//        ERROR("[Error] Unknown input type for " << str_type);
//        return -1;
//    }
//    cout << " - Image size input = " << vImages[0].size() << endl;
//    resizeFlipRotateImages(vImages, scale);

    vector<string> vImageFiles;
    ReadImageNamesFromFolder(str_folder, vImageFiles);

    map<string, string> options;
    options["type"] = "scoremap";
    const size_t N = vImageFiles.size();
    for (size_t i = 0; i < N; ++i) {
        INFO("Dealing with image " << vImageFiles[i]);

        //! TODO Mat to binary string

        // 调用接口函数
        string imageFile;
        aip::get_file_content(vImageFiles[i].c_str(), &imageFile);
        Json::Value result = client.body_seg(imageFile, options/*aip::null*/);

        // 解析Json结果
        string scoremap = result["scoremap"].asString();    // 灰度图像
        string decode_result = aip::base64_decode(scoremap);
//        string labelmap = result["labelmap"].asString();    // 二值图像
//        string foreground = result["foreground"].asString();// 前景

        vector<char> base64_img(decode_result.begin(), decode_result.end());
        Mat image = imdecode(base64_img, IMREAD_COLOR);
        imshow("mask", image);
        waitKey(0);

//        Json::Value result2 = client.body_seg_cv(image, options);
    }



    destroyAllWindows();
    return 0;
}
