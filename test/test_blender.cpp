#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/photo.hpp>

#define RESIZE_SRC  0

using namespace std;
using namespace cv;

enum BlendingType {
    ALPHA_BLENDING,
    MULTIBAND_BLENDING,
    POISSON_BLENDING
};

int main(int argc, char* argv[])
{
/*
    BlendingType BT = POISSON_BLENDING;
    detail::Blender* blender = nullptr;

    if (argc < 2) {
        cerr << "Parameters: blending_type(0 = ALPHA_BLENDING, 1 = MULTIBAND_BLENDING, default = POISSON_BLENDING)" << endl << endl;
        BT = POISSON_BLENDING;
        blender = new detail::Blender();
    } else {
        switch (atoi(argv[1])) {
        case 0:
            BT = ALPHA_BLENDING;
            blender = new detail::FeatherBlender();
            break;
        case 1:
            BT = MULTIBAND_BLENDING;
            blender = new detail::MultiBandBlender(false, 5); // CV_32F
            break;
        default:
            BT = POISSON_BLENDING;
            blender = new detail::Blender();
            break;
        }
    }
*/

    Mat src = imread("/home/vance/rk_ws/MotionShot/resource/blending/source_02.jpg", CV_LOAD_IMAGE_COLOR);
    Mat mask = imread("/home/vance/rk_ws/MotionShot/resource/blending/mask_02.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    Mat targ = imread("/home/vance/rk_ws/MotionShot/resource/blending/target_02.jpg", CV_LOAD_IMAGE_COLOR);
    if (src.empty() || mask.empty() || targ.empty()) {
        cerr << "Empty Images!" << endl;
        exit(-1);
    }
    assert(src.rows == mask.rows);
    assert(src.cols == mask.cols);
    cout << src.type() << mask.type() << targ.type() << endl;

//    Mat src_s, targ_s;
//    src.convertTo(src_s, CV_32F);
//    targ.convertTo(targ_s, CV_32F);

//    Mat mask1(image1s.size(), CV_8U);
//    mask1(Rect(0, 0, mask1.cols/2, mask1.rows)).setTo(255);
//    mask1(Rect(mask1.cols/2, 0, mask1.cols - mask1.cols/2, mask1.rows)).setTo(0);

//    Mat mask2(image2s.size(), CV_8U);
//    mask2(Rect(0, 0, mask2.cols/2, mask2.rows)).setTo(0);
//    mask2(Rect(mask2.cols/2, 0, mask2.cols - mask2.cols/2, mask2.rows)).setTo(255);

//    blender->prepare(Rect(0, 0, src.cols, src.rows));
//    blender->feed(src_s, mask, Point(0, 0));
//    blender->feed(targ_s, mask, Point(0, 0));
//    blender->blend(result_s, result_mask);
//    Mat result, result2;
//    result_s.convertTo(result, CV_8U);
//    result_mask.convertTo(result2, CV_8UC3);

#if RESIZE_SRC
    Size s = src.size();
    resize(src, src, Size(s.width/2, s.height/2));
    resize(mask, mask, Size(s.width/2, s.height/2));
    Point placed(400, 200); // for 05
#else
    Point placed(77, 70); // for 02
#endif
    Mat result;
    // NORMAL_CLONE, MIXED_CLONE, 黑白单色MONOCHROME_TRANSFER
    seamlessClone(src, targ, mask, placed, result, NORMAL_CLONE);

    imshow("source image", src);
    imshow("target image", targ);
    imshow("blending result", result);
    imwrite("/home/vance/output/blending_result.bmp", result);

    waitKey(0);
    return 0;
}
