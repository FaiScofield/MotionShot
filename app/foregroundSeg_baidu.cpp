#include "MotionShoter/utility.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

#define USE_BAIDU_API   1
#define USE_FACEPP_API  1

#if USE_BAIDU_API
#include "Thirdparty/aip-cpp-sdk-0.8.5/body_analysis.h"

const std::string app_id = "18679385";
const std::string api_key = "12yptwaZPOxoGBfPR0PGYT43";
const std::string secret_key = "sx8w8l1dzlD2Gt0QAKgZxItRB3uE8DZz";

aip::Bodyanalysis client(app_id, api_key, secret_key);
#elif   USE_FACEPP_API
#include "Thirdparty/facepp_cpp_sdk/FaceppApiLib/FaceppApi.hpp"

const std::string api_key = "KCKnVu0tgNMSE7yAnxmIqxu2kF0o2x7d";
const std::string secret_key = "r7F_WonCxxcgdOtqFuYNtGSub6M78N6N";

FaceppApi client(api_key, secret_key);
#endif

using namespace std;
using namespace cv;
using namespace ms;

int main(int argc, char *argv[])
{
    /// parse input arguments
    CommandLineParser parser(argc, argv,
            "{type      t|SEQUENCE|value input type: VIDEO, HUAWEI, SEQUENCE}"
            "{folder    f| |data folder or video file}"
            "{output    o| |output folder}"
            "{suffix    x|jpg|valid type: \"png\", \"jpg\", \"jpeg\", \"bmp\"}"
            "{scale     c|1|scale to resize image in output}"
            "{start     a|0|start index for image sequence}"
            "{end       e|-1|end index for image sequence}"
            "{mark      m|false|mark the foreground area}"
            "{help      h|false|show help message}");

    if (argc < 2 || parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_folder = parser.get<String>("folder");
    const String str_output = parser.get<String>("output");
    double scale = parser.get<double>("scale");
    int start = parser.get<int>("start");
    int end = parser.get<int>("end");
    bool mark = parser.get<bool>("mark");
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;
    cout << " - output = " << str_output << endl;
    cout << " - scale = " << scale << endl;
    cout << " - startIndex = " << start << endl;
    cout << " - endIndex = " << end << endl;
    cout << " - mark = " << mark << endl;

    const string commond = "mkdir -p " + str_output;
    system(commond.c_str());

    vector<string> vImageNames;
    ReadImageNamesFromFolder(str_folder, vImageNames);

#if USE_BAIDU_API
    map<string, string> options;
    options["type"] = "scoremap";   // labelmap, scoremap, foreground

    for (size_t i = 0, iend = vImageNames.size(); i < iend; ++i) {
        string image;
        aip::get_file_content(vImageNames[i].c_str(), &image);
        Json::Value result = client.body_seg(image, options);

        cout << result["scoremap"] << endl;
    }

#elif USE_FACEPP_API
    for (size_t i = 0, iend = vImageNames.size(); i < iend; ++i) {
        client.humanbodySegment(vImageNames[i].c_str());
    }
#endif


    INFO("done.");

    return 0;
}
