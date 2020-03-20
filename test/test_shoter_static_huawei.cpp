#include "MotionShoter/utility.h"
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <opencv2/photo.hpp>
#include <set>
#include "ImageBlender/cvBlenders.h"
#include "ImageBlender/cvSeamlessCloning_func.hpp"


#define GET_GROUNDTRUTH_FROM_BAIDU      1
#define GET_GROUNDTRUTH_FROM_FACEPP     0

using namespace std;
using namespace cv;
using namespace ms;

enum BlenderType { NO, FEATHER, MULTI_BAND };

bool IS_LARGE_IMAGE_SIZE = true;

void getGradient(const Mat& src, const Mat& mask, Mat& gradient);
void maskCoarse2Fine(const vector<Mat>& srcs, vector<Mat>& masks);

int main(int argc, char* argv[])
{
    /// parse input arguments
    CommandLineParser
        parser(argc, argv,
               "{folder     f| |huawei SEQUENCE dataset folder}"
               "{blender    b|feather|valid blend type: \"feather\", \"multiband\", \"poission\"}"
               "{scale      c|0.5|scale of inpute image}"
               "{begin      a|1|start index for image sequence}"
               "{end        e|7|end index for image sequence}"
               "{help       h|false|show help message}");

    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    TickMeter timer;
    timer.start();

    String str_blender = parser.get<String>("blender");
    String str_folder = parser.get<String>("folder");
    if ((*str_folder.end()) == '/')
        str_folder = str_folder.substr(0, str_folder.size() - 1);
    cout << " - folder = " << str_folder << endl;
    cout << " - blender = " << str_blender << endl;

    string strMode1;
    BlenderType blenderType = NO;
    ms::cvBlender* blender = nullptr;
    if (str_blender == "feather") {
        blender = new ms::cvFeatherBlender(0.1, true);
        blenderType = FEATHER;
        strMode1 = "羽化";
    } else if (str_blender == "multiband") {
        blender = new ms::cvMultiBandBlender(false, 5, CV_32F);
        blenderType = MULTI_BAND;
        strMode1 = "多频带";
    } /*else if (str_blender == "poission") {
        blender = new ms::PoissionBlender(); // TODO
    }*/ else {
        cerr << "[Error] Unknown blender type for " << str_blender << endl;
        exit(-1);
    }

    double scale = parser.get<double>("scale");
    int beginIdx = parser.get<int>("begin");
    int endIdx = parser.get<int>("end");
    cout << " - begin index = " << beginIdx << endl;
    cout << " - end index = " << endIdx << endl;
    cout << " - scale = " << scale << endl;

    // 1.1读取原图
    vector<Mat> vImages, vGTsColor, vGTsGray, vForegroundMasks;
    ReadImageSequence(str_folder, "jpg", vImages, beginIdx, endIdx - beginIdx + 1);  // 1 ~ 12
//    resizeFlipRotateImages(vImages, 0.5);  //! 注意部分rect掩模是在原图缩小0.5倍后获得的
    const int N = vImages.size();

    // 1.2读取前景掩模
    {
#if GET_GROUNDTRUTH_FROM_BAIDU
        //! 注意通过NN得到的掩模可能非唯一/边界不完整/存在小孔洞, 需要过滤/膨胀处理
        vector<Mat> vGTsMaskRect;
        vector<Rect> vGTsRect;
        const string gtFolder = str_folder + "/../gt_rect_baidu/";
        ReadGroundtruthRectFromFolder(gtFolder, "png", vGTsMaskRect, vGTsRect, beginIdx, endIdx - beginIdx + 1);
        assert(vGTsMaskRect.size() == vGTsRect.size());

        vGTsGray.reserve(vGTsMaskRect.size());
        for (size_t i = 0, iend = vGTsMaskRect.size(); i < iend; ++i) {
            Mat gtGray = Mat::zeros(vImages[0].size(), CV_8UC1);
            vGTsMaskRect[i].copyTo(gtGray(vGTsRect[i]));

//            vector<vector<Point>> contours;
//            findContours(gtGray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

#define SHOW_MASK_INPUT 0
#if     SHOW_MASK_INPUT
            Mat toShow;
            cvtColor(gtGray, toShow, COLOR_GRAY2BGR);
            drawContours(toShow, contours, -1, Scalar(0,255,0));
#endif
//            if (contours.size() > 1) {
//                int maxIdx = 0, maxArea = 0;
//                for (int j = 0, jend = contours.size(); j < jend; ++j) {
//                    Rect r = boundingRect(contours[j]);
//                    if (r.area() > maxArea) {
//                        maxIdx = j;
//                        maxArea = r.area();
//                    }
//                }
//                gtGray = Mat::zeros(gtGray.size(), CV_8UC1);
//                drawContours(gtGray, contours, maxIdx, Scalar(255), -1);

#if     SHOW_MASK_INPUT
                rectangle(toShow, boundingRect(contours[maxIdx]), Scalar(0,0,255), 2);
                namedLargeWindow("前景掩模&轮廓", IS_LARGE_IMAGE_SIZE);
                imshow("前景掩模&轮廓", toShow);
                waitKey(500);
#endif
//            }
            vGTsGray.push_back(gtGray);

            //! test
//            Mat fore, fore_YUV, backgroundMask;
//            fore = vImages[i](vGTsRect[i]);
//            bitwise_not(vGTsMaskRect[i], backgroundMask);
//            fore.setTo(0, backgroundMask);
//            imshow("前景", fore);

//            vector<Mat> vForeChannels;
//            cvtColor(fore, fore_YUV, COLOR_BGR2YUV);
//            split(fore_YUV, vForeChannels);
//            imshow("前景Y", vForeChannels[0]);
//            imshow("前景U", vForeChannels[1]);
//            imshow("前景V", vForeChannels[2]);

//            waitKey(0);

        }
//        exit(0);
#elif GET_GROUNDTRUTH_FROM_FACEPP
        vector<Mat> vGTsMaskRect;
        vector<Rect> vGTsRect;
        const string gtFolder = str_folder + "/../gt_rect_small_facepp/";
        ReadGroundtruthRectFromFolder(gtFolder, "jpg", vGTsMaskRect, vGTsRect, beginIdx, endIdx - beginIdx + 1);
        assert(vGTsMaskRect.size() == vGTsRect.size());

        vGTsGray.reserve(vGTsMaskRect.size());
        for (size_t i = 0, iend = vGTsMaskRect.size(); i < iend; ++i) {
            Mat gtGray = Mat::zeros(vImages[0].size(), CV_8UC1);
            vGTsMaskRect[i].copyTo(gtGray(vGTsRect[i]));

            threshold(gtGray, gtGray, 50, 0, THRESH_TOZERO);
            normalize(gtGray, gtGray, 0, 255, NORM_MINMAX);

            vector<vector<Point>> contours;
            findContours(gtGray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//            Mat toShow;
//            cvtColor(gtGray, toShow, COLOR_GRAY2BGR);
//            drawContours(toShow, contours, -1, Scalar(0,255,0), 1);
//            imshow("mask input & contous", toShow);
//            waitKey(300);

            if (contours.size() > 1) {
                int maxIdx = 0, maxArea = 0;
                for (int j = 0, jend = contours.size(); j < jend; ++j) {
                    Rect r = boundingRect(contours[j]);
                    if (r.area() > maxArea) {
                        maxIdx = j;
                        maxArea = r.area();
                    }
                }
                gtGray = Mat::zeros(gtGray.size(), CV_8UC1);
                drawContours(gtGray, contours, maxIdx, Scalar(255), -1);

//                rectangle(toShow, boundingRect(contours[maxIdx]), Scalar(0,0,255), 2);
//                imshow("mask & max rect", toShow);
//                imshow("mask input final", gtGray);
//                waitKey(300);
            }
            vGTsGray.push_back(gtGray);

        }
#else
        ReadImageSequence(str_folder + "/../gt_full", "jpg", vGTsColor, beginIdx, endIdx - beginIdx + 1);
        colorMask2Gray(vGTsColor, vGTsGray);
#endif
        resizeFlipRotateImages(vImages, scale);
        resizeFlipRotateImages(vGTsGray, scale);

        if (vImages.empty() || vGTsGray.empty())
            exit(-1);

        assert(N > 2);
        assert(vImages.size() == vGTsGray.size());
        assert(vImages[0].size() == vGTsGray[0].size());
        if (vImages[0].cols > 1600 || vImages[0].rows > 1200)
            IS_LARGE_IMAGE_SIZE = true;
    }
    destroyAllWindows();

    // 1.3掩膜边缘平滑
    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    const Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
    const Mat kernel3 = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
    vForegroundMasks.reserve(N);
    for (int i = 0; i < N; ++i) {
        Mat mask, maskScaled, tmpMask;
        mask = vGTsGray[i].clone();
        tmpMask = mask.clone();   // 形态学操作前

        compare(mask, 0, mask, CMP_GT);
        morphologyEx(mask, mask, MORPH_OPEN, kernel1, Point(-1,-1), 1);  // 开操作(先腐蚀再膨胀),平滑边界, 去噪点
        morphologyEx(mask, mask, MORPH_CLOSE, kernel2, Point(-1,-1), 1); // 去孔洞
        morphologyEx(mask, mask, MORPH_OPEN, kernel3, Point(-1,-1), 1);  // 平滑边界
#if GET_GROUNDTRUTH_FROM_BAIDU
        smoothMaskWeightEdge(mask, mask, 5, 0);
#elif GET_GROUNDTRUTH_FROM_FACEPP
//        morphologyEx(mask, mask, MORPH_DILATE, kernel1, Point(-1,-1), 1);
//        mask.setTo(255, mask);
        smoothMaskWeightEdge(mask, mask, 5, 0);
#endif
//        resize(mask, maskScaled, Size(), 2, 2);
//        vForegroundMasks.push_back(maskScaled);
        vForegroundMasks.push_back(mask);

#define SHOW_MASK_PREPROCESS_RESULT 0
#if     SHOW_MASK_PREPROCESS_RESULT
        Rect maskRect = boundingRect(mask);
        imshow("前景掩模原始输入", tmpMask);
        imshow("前景掩模平滑后", mask);
        string txt1 = "/home/vance/output/ms/" + to_string(i+1) + "前景掩模原始.jpg";
        string txt2 = "/home/vance/output/ms/" + to_string(i+1) + "前景掩模平滑.jpg";
        imwrite(txt1, tmpMask);
        imwrite(txt2, mask);
        waitKey(1000);
#endif
    }

    timer.stop();
    TIMER("1.系统初始化耗时(s): " << timer.getTimeSec());
    double t1, t2, t3;

    destroyAllWindows();
//    exit(0);
    timer.start();

    maskCoarse2Fine(vImages, vForegroundMasks);
    exit(0);

    // 2.前景拼接融合
    const Mat pano = vImages.front().clone();
    const Rect disRoi(0, 0, pano.cols, pano.rows);

    blender->prepare(disRoi);
    for (int i = 0; i < N; ++i) {
        const Mat& frame = vImages[i];
        const Mat& foreMask = vForegroundMasks[i];

        // blender
        Mat frame_S;
        frame.convertTo(frame_S, CV_16SC3);
        blender->feed(frame_S, foreMask, Point(0, 0));
    }

    Mat allForeground, allForeground_S, allForegroundMask;
    blender->blend(allForeground_S, allForegroundMask);
    allForeground_S.convertTo(allForeground, CV_8UC3);

    const Rect foregroundRect = boundingRect(allForegroundMask);
//    imshow("allForegroundMask", allForegroundMask);
//    imshow("allForeground", allForeground);

    // 画前景轮廓和重叠区域轮廓
    Mat allForegroundShow = allForeground.clone();
    vector<vector<Point>> contours;
    findContours(allForegroundMask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    drawContours(allForegroundShow, contours, -1, Scalar(255,0,0), 1);  // 画前景轮廓
    const Mat overlappedEdges = blender->getOverlappedEdges();
    allForegroundShow.setTo(Scalar(0,255,0), overlappedEdges);  // 画重叠区域轮廓
//    namedLargeWindow("所有前景轮廓(蓝)&重叠区域(绿)", IS_LARGE_IMAGE_SIZE);
//    imshow("所有前景轮廓(蓝)&重叠区域(绿)", allForegroundShow);
//    waitKey(0);

    timer.stop();
    t1 = timer.getTimeSec();
    TIMER("2.前景拼接融合耗时(s): " << t1);

//    string txt1 = "/home/vance/output/ms/前景拼接结果-初始.jpg";
//    imwrite(txt1, allForeground);
    string txt2 = "/home/vance/output/ms/前景重叠区域轮廓.jpg";
    imwrite(txt2, allForegroundShow);
    timer.start();

    // 3.前景拼接结果改善(边缘滤波)
    Mat overlappedEdgesMask = blender->getOverlappedEdgesMask(3);
    bitwise_and(overlappedEdgesMask, allForegroundMask, overlappedEdgesMask);// 扣掉前景轮廓外那部分的待平滑区域
    Mat foregroundFiltered, foregroundFiltered_S;
    overlappedEdgesSmoothing(allForeground, overlappedEdgesMask, foregroundFiltered, 0.25);
    foregroundFiltered.convertTo(foregroundFiltered_S, CV_16SC3);

    timer.stop();
    t2 = timer.getTimeSec();
    TIMER("3.前景拼接结果改善(s): " << t2);

    string txt3 = "/home/vance/output/ms/前景拼接结果-改善.jpg";
    imwrite(txt3, foregroundFiltered);
    timer.start();

    // 4.前背景融合. 把pano和前景对应的mask区域的稍微缩收/扩张, 设置平滑的权重过渡, 然后再融合.
    //! 暂时不能用多频段融合.
    ms::cvFeatherBlender* blender2 = new ms::cvFeatherBlender(0.1, false);
    ms::cvFeatherBlender* blender3 = new ms::cvFeatherBlender(0.1, false);
    const string strMode2 = "羽化";

    blender2->prepare(disRoi);
    blender3->prepare(disRoi);

    Mat foregroundMaskFinal, backgroundMaskFinal, maskFinal;
    smoothMaskWeightEdge(allForegroundMask, foregroundMaskFinal, 0);  // 过渡边缘
    bitwise_not(foregroundMaskFinal, backgroundMaskFinal);

    Mat pano_S, result1, resultMask1, result2, resultMask2;
    pano.convertTo(pano_S, CV_16SC3);
    blender2->feed(pano_S, backgroundMaskFinal, Point(0, 0));
    blender2->feed(foregroundFiltered_S, foregroundMaskFinal, Point(0, 0));
    blender2->blend(result1, resultMask1);
    result1.convertTo(result1, CV_8UC3);

    timer.stop();
    t3 = timer.getTimeSec();
    TIMER("4.前背景融合耗时(s): " << t3);
    TIMER(N << "个前景, 算法整体耗时(s): " << t1+t2+t3);

    blender3->feed(pano_S, backgroundMaskFinal, Point(0, 0));
    blender3->feed(allForeground_S, foregroundMaskFinal, Point(0, 0));
    blender3->blend(result2, resultMask2);
    result2.convertTo(result2, CV_8UC3);

    string txt5 = "/home/vance/output/ms/最终结果.jpg";
    imwrite(txt5, result1);
    string txt6 = "/home/vance/output/ms/初步结果.jpg";
    imwrite(txt6, result2);

    waitKey(10);
    return 0;
}


void maskCoarse2Fine(const vector<Mat>& srcs, vector<Mat>& masks)
{
    const size_t N = srcs.size();
//    for (size_t i = 0; i < N; ++i) {
//        const Mat& img = srcs[i];
//        const Mat& mask = masks[i];
//        const Rect rec = boundingRect(masks[i]);

        const Mat& img = srcs.back();
        const Mat& mask = masks.back();
        const Rect rec = boundingRect(mask);

        // 计算梯度
        Mat edge, ignoreArea, gradientMask;
        const Mat kernel = getStructuringElement(MORPH_RECT, Size(21, 21));
        morphologyEx(mask, edge, MORPH_GRADIENT, Mat());
        erode(mask, ignoreArea, kernel);
//        bitwise_and(mask, 0, gradientMask, ignoreArea);
        gradientMask = mask - ignoreArea;
        imwrite("/home/vance/output/1待计算梯度的区域.png", gradientMask);

        Mat gradient;
        getGradient(img, gradientMask, gradient);

        Mat keepArea1, keepArea2, keepArea3;
        dilate(ignoreArea, keepArea1, Mat(), Point(-1, -1), 3);
        dilate(keepArea1, keepArea2, Mat(), Point(-1, -1), 2);
        dilate(keepArea2, keepArea3, Mat(), Point(-1, -1), 2);

        Mat keepGradient1, keepGradient2, keepGradient3, keepGradientFore;
        Mat toShow0, toShow1, toShow2, toShow3;
        img.copyTo(toShow0, mask);
        img.copyTo(toShow1, mask);
        img.copyTo(toShow2, mask);
//        img.copyTo(toShow3, mask);
        bitwise_and(gradient, keepArea1, keepGradient1);
        bitwise_and(gradient, keepArea2, keepGradient2);
        bitwise_and(gradient, keepArea3, keepGradient3);
        bitwise_and(gradient, mask, keepGradientFore);
        imwrite("/home/vance/output/2前景梯度.png", keepGradientFore );
        imwrite("/home/vance/output/2可能的白边梯度区域1.png", keepGradient1);
        imwrite("/home/vance/output/2可能的白边梯度区域2.png", keepGradient2);
        imwrite("/home/vance/output/2可能的白边梯度区域3.png", keepGradient3);

//        normalize(keepGradient1, keepGradient1, 0, 255, NORM_MINMAX);
//        normalize(keepGradient2, keepGradient2, 0, 255, NORM_MINMAX);
//        normalize(keepGradient3, keepGradient3, 0, 255, NORM_MINMAX);
        threshold(keepGradient1, keepGradient1, 150, 255, THRESH_BINARY);
        threshold(keepGradient2, keepGradient2, 150, 255, THRESH_BINARY);
        threshold(keepGradient3, keepGradient3, 150, 255, THRESH_BINARY);
        toShow0.setTo(Scalar(0,255,255), gradient);
//        toShow1.setTo(Scalar(255,0,0), keepGradient1);
//        toShow2.setTo(Scalar(0,255,0), keepGradient2);
//        toShow3.setTo(Scalar(0,0,255), keepGradient3);
        imwrite("/home/vance/output/人物梯度(原始).png", toShow0);


        Mat gradientFilter1, gradientFilter2;
        const Mat smoothKernel = getStructuringElement(MORPH_RECT, Size(7, 7));
        morphologyEx(gradient, gradientFilter1, MORPH_CLOSE, Mat(), Point(-1, -1), 1);
        morphologyEx(gradient, gradientFilter2, MORPH_OPEN, Mat(), Point(-1, -1), 1);
        morphologyEx(gradientFilter2, gradientFilter2, MORPH_CLOSE, Mat(), Point(-1, -1), 1);
        toShow1.setTo(Scalar(0,255,255), gradientFilter1);
        toShow2.setTo(Scalar(0,255,255), gradientFilter2);

        Mat fore;
        bitwise_and(img, 255, fore,  mask);
        imwrite("/home/vance/output/人物前景.png", fore);

        vector<vector<Point>> contours;
        gradientFilter2.convertTo(gradientFilter2, CV_32SC1);
        findContours(gradientFilter2, contours, RETR_FLOODFILL, CHAIN_APPROX_SIMPLE);
        drawContours(toShow2, contours, -1, Scalar(0,255,0));

        imwrite("/home/vance/output/人物梯度(关操作).png", toShow1);
        imwrite("/home/vance/output/人物梯度(开操作).png", toShow2);

        Mat maskCut(mask.size(), mask.type());
        maskCut.setTo(0);
        drawContours(maskCut, contours, -1, Scalar::all(255), -1);
        bitwise_and(mask, maskCut, maskCut);

        bitwise_and(img, 255, toShow3, maskCut);
        imwrite("/home/vance/output/人物前景(去白边).png", toShow3);

        // 计算重叠区域
//        Mat overlappedArea, ignoreArea, edge;
//        bitwise_and(mask1, 255, overlappedArea, mask2);         // 得到重叠区域
//        if (countNonZero(overlappedArea) > 100)
//        morphologyEx(overlappedArea, edge, MORPH_GRADIENT, Mat());  // 得到重叠区域边缘

//    }
}

void getGradient(const Mat &src, const Mat &mask, Mat &gradient)
{
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    static vector<Mat> vKernels;
    if (vKernels.empty()) {
        vKernels.resize(4);
        vKernels[0] = (Mat_<char>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
        vKernels[1] = vKernels[0].t();
        vKernels[2] = (Mat_<char>(3, 3) << -10, -3, 0, -3, 0, 3, 0, 3, 10);
        vKernels[3] = (Mat_<char>(3, 3) << 0, 3, 10, -3, 0, 3, -10, -3, 0);
    }

    // 1.下采样并在Y域上求解4个方向的梯度
//    resize(src, src_L1, Size(), scale, scale);
//    resize(mask, mask_L1, Size(), scale, scale);
//    resize(srcGray, srcGray_L1, Size(), scale, scale);
    vector<Mat> vGradient(4);   // CV_32FC1
    for (int i = 0; i < 4; ++i)
        filter2D(srcGray, vGradient[i], CV_32FC1, vKernels[i], Point(-1, -1), 0, BORDER_REPLICATE);

    // 2.对掩模对应的区域的梯度, 选出4个方向中的最大者并标记最大梯度方向
    Mat maxGradientEdge = Mat::zeros(srcGray.size(), CV_32FC1);
    Mat dirLabel = Mat::zeros(srcGray.size(), CV_8UC1);   // 掩模区内边缘的方向
    for (int y = 0; y < srcGray.rows; ++y) {
        const char* mask_row = mask.ptr<char>(y);
        float* grad_row = maxGradientEdge.ptr<float>(y);
        char* label_row = dirLabel.ptr<char>(y);

        for (int x = 0; x < srcGray.cols; ++x) {
            if (mask_row[x] != 0) {
                int label = 0;  // 默认无梯度则黑色
                float maxGradient = 0.f;
                for (int i = 0; i < 4; ++i) {
                    const float g = abs(vGradient[i].at<float>(y, x)); // 绝对值符号使正负方向为同一方向
                    if (g > maxGradient) {
                        maxGradient = g;
                        label = i + 1;
                    }
                }
                grad_row[x] = maxGradient;
                label_row[x] = label;
            }
        }
    }
    normalize(maxGradientEdge, maxGradientEdge, 0.f, 1.f, NORM_MINMAX);  // 归一化
    Mat maxGradient_L1;
    maxGradientEdge.convertTo(maxGradient_L1, CV_8UC1, 255);
    threshold(maxGradient_L1, gradient, 10, 0, THRESH_TOZERO);  // 太小的梯度归0

    // 大梯度的白边区域
    Mat largeGradienMask_L1;
    threshold(maxGradient_L1, largeGradienMask_L1, 150, 255, THRESH_OTSU);

    imwrite("/home/vance/output/人物梯度.png", gradient);
}
