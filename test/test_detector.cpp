/*=================================================
 * Version:
 * v1.0: 原版程序由IplImage转换为Mat
===================================================
*/

#include "BGDifference.h"
#include "BS_MOG2_CV.h"
#include "FramesDifference.h"
#include "ViBePlus.h"
#include "OpticalFlower.h"
#include "Vibe.h"
#include "utility.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>

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
               "{detector  d|fd|value detector: bgd, fd, mog2, vibe, vibe+, flow}"
               "{size      s|5|board size for fd detector}"
               "{showGT    g|false|if show ground for type DATASET}"
               "{scale     c|1|scale to resize image, 0.15 for type HUAWEI}"
               "{flip      p|0|flip image for type VIDEO, 0(x), +(y), -(xy)}"
               "{rotate    r|-1|rotate image for type VIDEO, r = cv::RotateFlags(0, 1, 2)}"
               "{write     w|false|write result sequence to a dideo}"
               "{help      h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    const String str_type = parser.get<String>("type");
    const String str_detector = parser.get<String>("detector");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    double scale = parser.get<double>("scale");
    int flip = parser.get<int>("flip");
    int rotate = parser.get<int>("rotate");
    bool showGT = parser.get<bool>("showGT");
    cout << " - type = " << str_type << endl;
    cout << " - folder = " << str_folder << endl;
    cout << " - detector = " << str_detector << endl;

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

    BaseMotionDetector* detector;
    if (str_detector == "bgd") {
        // TODO
        // detector = dynamic_cast<BaseMotionDetector*>(new BGDiff());
    } else if (str_detector == "fd") {
        int size = parser.get<int>("size");
        cout << " - kernel size = " << size << endl;
        assert(size > 2);
        detector = dynamic_cast<BaseMotionDetector*>(new FramesDifference(2, size));
    } else if (str_detector == "mog2") {
        Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 100.0, true);
        detector = dynamic_cast<BaseMotionDetector*>(new BS_MOG2_CV(bs.get()));
    } else if (str_detector == "flow") {
        Ptr<FarnebackOpticalFlow> fof = FarnebackOpticalFlow::create();
        detector = dynamic_cast<BaseMotionDetector*>(new OpticalFlower(fof.get()));
    } /*else if (str_detector == "vibe") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBe(20, 2, 20, 16));
    } else if (str_detector == "vibe+") {
        // TODO
        detector = dynamic_cast<BaseMotionDetector*>(new ViBePlus(20, 2, 20, 16));
    }*/
    else {
        cerr << "[Error] Unknown input detector for " << str_detector << endl;
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
        for (int i = 0; i < N; ++i) {
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
        for (int i = 0; i < N; ++i) {
            Mat imgi;
            cv::flip(vImages[i], imgi, flip);
            vImgFlipped[i] = imgi;
        }
        vImages.swap(vImgFlipped);
    } else if (rotate >= 0) {
        cout << " - rotate = " << rotate << endl;
        vector<Mat> vImgRotated(N);
        for (int i = 0; i < N; ++i) {
            Mat imgi;
            cv::rotate(vImages[i], imgi, rotate);
            vImgRotated[i] = imgi;
        }
        vImages.swap(vImgRotated);
    }

    /// detect moving frontground
    const bool write = parser.get<bool>("write");
    VideoWriter writer("/home/vance/output/result.avi", VideoWriter::fourcc('M','J','P','G'), 25,
                       Size(vImages[0].cols, vImages[0].rows * 2));
    for (size_t i = 0, iend = vImages.size(); i < iend; ++i) {
        const Mat& frame = vImages[i];
        Mat diff;
        detector->apply(frame, diff);
        if (diff.empty())
            continue;

        // find contours
        Mat frame_contours = frame.clone();
        vector<vector<Point>> contours;
        findContours(diff, contours, RETR_EXTERNAL,
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
        cout << " - max blob area: " << maxArea << endl;

        Mat diff_blobs, tmp, output;
        cvtColor(diff, diff_blobs, COLOR_GRAY2BGR);
        for (int i = 0, iend = blobs.size(); i < iend; ++i) {
            rectangle(diff_blobs, blobs[i], CV_RGB(0, 255, 0), 1);
            string txt = to_string(i) + "-" + to_string(blobs[i].area());
            putText(diff_blobs, txt, blobs[i].tl(), 1, 1., Scalar(0,0,255));

//            const Point tl = blobs[i].tl();
//            const Point br = blobs[i].br();
//            mask.rowRange(tl.y, br.y).colRange(tl.x, br.x).setTo(255);
        }

        vconcat(frame_contours, diff_blobs, tmp);
        if (showGT && !vGTs.empty())
            vconcat(tmp, vGTs[i], output);
        else
            output = tmp;

        imshow("result", output);
        if (write)
            writer.write(output);
        if (waitKey(50) == 27)
            break;
    }

    writer.release();
    destroyAllWindows();
    return 0;
}
