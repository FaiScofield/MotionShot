#include "MotionShoter/utility.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace ms;

struct GCAPP
{
    enum { NOT_SET = 0, IN_PROCESS = 1, SET = 2 };

    void setImage(const cv::Mat& img) { img.copyTo(image_); }
    void setOutConfigName(const string& name) { outConfigName_ = &name; }
    void setWindowName(const string& name) { winName_ = &name; }
    void setOutImageName(const string& name) { outImageName_ = name; }

    void showImage();
    void mouseClick(int event, int x, int y, int flags, void* param);
    void reset();

    const string* winName_;
    const string* outConfigName_;
    string outImageName_;
    Mat image_;
    Rect rect_;
    uchar rectState_;

    ofstream ofs_;
};

void GCAPP::showImage()
{
    if (image_.empty() || winName_->empty())
        return;

    Mat res;
    image_.copyTo(res);
    if (rectState_ == IN_PROCESS || rectState_ == SET)
        rectangle(res, rect_, Scalar(0,0,255), 2);

    imshow(*winName_, res);
}

void GCAPP::mouseClick(int event, int x, int y, int flags, void *param)
{
    switch (event) {
    case EVENT_LBUTTONDOWN: {
        if (rectState_ == NOT_SET) {
            rectState_ = IN_PROCESS;
            rect_ = Rect(x, y, 1, 1);
        }
       break;}
    case EVENT_LBUTTONUP: {
        if (rectState_ == IN_PROCESS) {
            rectState_ = SET;
            rect_ = Rect(Point(rect_.x, rect_.y), Point(x, y));
            showImage();
        }
       break;}
    case EVENT_MOUSEMOVE: {
        if (rectState_ == IN_PROCESS) {
            rect_ = Rect(Point(rect_.x, rect_.y), Point(x, y));
            showImage();
        }
        break;}
    case EVENT_RBUTTONDBLCLK: {
        if (image_.empty() || outImageName_.empty()) {
            ERROR("Please set the image and output image name first!");
            return;
        }

//        vector<int> wParams{IMWRITE_JPEG_QUALITY};
        imwrite(outImageName_, image_(rect_)/*, wParams*/);
        INFO("Save image to " << outImageName_);

        ofs_.open((*outConfigName_), ios_base::app);
        if (!ofs_.is_open())
            ERROR("Failed to open file " << *outConfigName_);
        ofs_ << outImageName_ << "\tRect: " << rect_.x << ", " << rect_.y << ", "
             << rect_.br().x << ", " << rect_.br().y << endl;
        ofs_.close();
        reset();

        break;}
    }
}

void GCAPP::reset()
{
//    outImageName_.clear();
//    image_.release();
    rect_ = Rect();
    rectState_ = NOT_SET;
}

///////////////////////////////////////////////////////////

GCAPP gcapp;

static void on_mouse(int event, int x, int y, int flags, void* param)
{
    gcapp.mouseClick(event, x, y, flags, param);
}

int main(int argc, char* argv[])
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

    vector<Mat> vImages;
    InputType inputType;
    if (str_type == "video" || str_type == "VIDEO") {
        inputType = VIDEO;
        ReadImageSequence_video(str_folder, vImages, start, end - start + 1);
    } else if (str_type == "huawei" || str_type == "HUAWEI") {
        inputType = HUAWEI;
        ReadImageSequence_huawei(str_folder, vImages, start, end - start + 1);
    } else if (str_type == "sequence" || str_type == "SEQUENCE") {
        inputType = SEQUENCE;
        const String str_suffix = parser.get<String>("suffix");
        cout << " - suffix = " << str_suffix << endl;
        ReadImageSequence(str_folder, str_suffix, vImages, start, end - start + 1);
    } else {
        cerr << "[Error] Unknown input type for " << str_type << endl;
        return -1;
    }

    resizeFlipRotateImages(vImages, scale);

    const size_t N = vImages.size();

    vector<int> params{IMWRITE_JPEG_QUALITY};
    if (mark) {
        const string winName = "image show";
        namedWindow(winName, WINDOW_AUTOSIZE);

        string outConfigFile = str_output + "/" + "config.txt";
        gcapp.setOutConfigName(outConfigFile);
        gcapp.setWindowName(winName);

        setMouseCallback(winName, on_mouse, 0);

        for (size_t i = 0; i < N; ++i) {
            const string outFile = str_output + "/" + to_string(i + 1) + "_RECT.png";

            gcapp.reset();
            gcapp.setImage(vImages[i]);
            gcapp.setOutImageName(outFile);
            gcapp.showImage();

            if ((char)waitKey(0) == '\x1b')
                exit(0);
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            const string outFile = str_output + "/" + to_string(i + 1) + ".jpg";
            imwrite(outFile, vImages[i], params);
        }
    }

    destroyAllWindows();
    INFO("done.");

    return 0;
}
