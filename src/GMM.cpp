#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

using namespace std;
using namespace cv;

#define USE_MOG2    1
#define USE_KNN     0

const string g_video = "/home/vance/opt/opencv-3.4.1/samples/data/vtest.avi";

int main(int argc, char** argv)
{
    VideoCapture vc(g_video);

#if USE_MOG2
    bool detectShadows = true;
    Ptr<BackgroundSubtractorMOG2> bs = createBackgroundSubtractorMOG2(500, 16.0, detectShadows);
#elif USE_KNN
    Ptr<BackgroundSubtractorKNN> bs createBackgroundSubtractorKNN(20, 16.0, detectShadows);
#else
    Ptr<BackgroundSubtractorMOG> bs = BackgroundSubtractorMOG::create(20, 16, 1.5);
#endif
    
    Mat frame, mask, th_img, output;
    while (true) {
        vc >> frame;
        bs->apply(frame, mask, 0.001);
        if (!frame.empty() && !mask.empty()) {
            Mat mask_color;
            cvtColor(mask, mask_color, COLOR_GRAY2BGR);
            hconcat(frame, mask_color, output);
            imshow("MOG Method", output);
            waitKey(20);
        }
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
