#include "utility.h"
#include <boost/algorithm/string_regex.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#define ENABLE_DEBUG 1

namespace ms
{

using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

void namedLargeWindow(const std::string& title, bool flag)
{
    if (flag) {
        namedWindow(title, WINDOW_KEEPRATIO | WINDOW_NORMAL);
        resizeWindow(title, Size(1080, 810));
    } else {
        namedWindow(title, WINDOW_AUTOSIZE);
    }
}

void ReadImageNamesFromFolder(const string& folder, vector<string>& names)
{
    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist! " << path << endl;
        return;
    }

    vector<string> vImageNames;
    vImageNames.reserve(100);
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status()))
            vImageNames.push_back(iter->path().string());
    }

    if (vImageNames.empty()) {
        cerr << "[Error] No image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read " << vImageNames.size() << " images in " << folder << endl;
    }

    vImageNames.shrink_to_fit();
    names.swap(vImageNames);
}

void ReadImagesFromFolder_lasiesta(const string& folder, vector<Mat>& imgs)
{
    imgs.clear();

    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist! " << path << endl;
        return;
    }

    std::map<int, string> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: I_IL_01-2.bmp
            const string fileName = iter->path().string();
            const size_t i = fileName.find_last_of('-');
            const size_t j = fileName.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto idx = atoi(fileName.substr(i + 1, j - i - 1).c_str());
            allImages.emplace(idx, fileName);
        }
    }

    if (allImages.empty()) {
        cerr << "[ERROR] No image data in the folder!" << endl;
        return;
    } else {
        cout << "[INFO ] Read " << allImages.size() << " images in " << folder << endl;
    }

    imgs.reserve(allImages.size());
    for (auto it = allImages.begin(), iend = allImages.end(); it != iend; it++)
        imgs.push_back(imread(it->second, IMREAD_COLOR));
}

void ReadGroundtruthFromFolder_lasiesta(const string& folder, vector<Mat>& imgs)
{
    imgs.clear();

    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist!" << path << endl;
        return;
    }

    map<int, string> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: I_IL_01-GT_29.png
            const string fileName = iter->path().string();
            const size_t i = fileName.find_last_of('_');
            const size_t j = fileName.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto idx = atoi(fileName.substr(i + 1, j - i - 1).c_str());
            allImages.emplace(idx, fileName);
        }
    }

    if (allImages.empty()) {
        cerr << "[Error] No gt image data in " << folder << endl;
        return;
    } else {
        cout << "[Info ] Read " << allImages.size() << " gt images in " << folder << endl;
    }

    imgs.reserve(allImages.size());
    for (auto it = allImages.begin(), iend = allImages.end(); it != iend; it++)
        imgs.push_back(imread(it->second, IMREAD_COLOR));
}

void ReadImageSequence_lasiesta(const string& folder, vector<Mat>& imgs, vector<Mat>& gts, int startIndex, int num)
{
    imgs.clear();

    vector<Mat> allImages, allGts;
    string gtFolder;
    if (folder.back() == '/')
        gtFolder = folder.substr(0, folder.size() - 1) + "-GT/";
    else
        gtFolder = folder + "-GT/";

    ReadImagesFromFolder_lasiesta(folder, allImages);
    ReadGroundtruthFromFolder_lasiesta(gtFolder, allGts);
    assert(startIndex < allImages.size());

    const int S = max(0, startIndex);
    const int N = num <= 0 ? allImages.size() - S : min(num, static_cast<int>(allImages.size()) - S);
    imgs = vector<Mat>(allImages.begin() + S, allImages.begin() + S + N);
    gts = vector<Mat>(allGts.begin() + S, allGts.begin() + S + N);

    assert(gts.size() == imgs.size());
    cout << "[Info ] Get " << imgs.size() << " images from tatal." << endl;
}

void ReadImageSequence_huawei(const string& folder, vector<Mat>& imgs, int startIndex, int num)
{
    imgs.clear();

    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist! " << path << endl;
        return;
    }

    std::map<int, string> allImages;
    bf::directory_iterator end_iter;
    for (bf::directory_iterator iter(path); iter != end_iter; ++iter) {
        if (bf::is_directory(iter->status()))
            continue;
        if (bf::is_regular_file(iter->status())) {
            // format: IMG_20200205_105231_BURST003.jpg
            const string fileName = iter->path().string();
            const size_t i = fileName.find_last_of('T');
            const size_t j = fileName.find_last_of('.');
            if (i == string::npos || j == string::npos)
                continue;
            auto idx = atoi(fileName.substr(i + 1, j - i - 1).c_str());
            allImages.emplace(idx, fileName);
        }
    }

    if (allImages.empty()) {
        cerr << "[Error] No image data in " << folder << endl;
        return;
    } else {
        cout << "[Info ] Read tatal " << allImages.size() << " images in " << folder << endl;
    }
    assert(startIndex < allImages.size());

    const int S = max(0, startIndex);
    const int N = num <= 0 ? allImages.size() - S : min(num, static_cast<int>(allImages.size()) - S);
    int idx = 0;
    imgs.reserve(N);
    for (auto it = allImages.begin(), iend = allImages.end(); it != iend; it++) {
        if (idx++ < S)
            continue;
        imgs.push_back(imread(it->second, IMREAD_COLOR));
        if (idx - S >= N)
            break;
    }
    cout << "[Info ] Get " << imgs.size() << " images from tatal." << endl;
    assert(imgs.size() == N);
}


void ReadImageSequence(const string& prefix, const string& suffix, vector<Mat>& imgs, int startIndex, int num)
{
    imgs.clear();
    imgs.reserve(num);

    string pre(prefix), suf(suffix);
    if (pre.back() != '/')
        pre += "/";
    if (suf.front() != '.')
        suf = "." + suf;

    for (int i = startIndex, iend = startIndex + num; i < iend; ++i) {
        const string imgName = pre + to_string(i) + suf;
        Mat img = imread(imgName, IMREAD_COLOR);
        if (img.empty())
            cerr << "[Error] No image name " << imgName << endl;
        else
            imgs.push_back(img);
    }

    cout << "[Info ] Read " << imgs.size() << " images in the sequence of the foler " << prefix << endl;
}

void ReadImagesFromVideo(const string& video, vector<Mat>& imgs)
{
    imgs.clear();
    imgs.reserve(100);

    VideoCapture capture(video);
    if (!capture.isOpened()) {
        cerr << "[Error] Unable to open video file: " << video << endl;
        return;
    }

    Mat img;
    while (capture.read(img))
        imgs.push_back(img.clone());  //! 注意浅拷贝问题

    imgs.shrink_to_fit();
    capture.release();

    cout << "[Info ] Read " << imgs.size() << " images from the video." << endl;
}

void ReadImageSequence_video(const string& video, vector<Mat>& imgs, int startIndex, int num)
{
    imgs.clear();
    imgs.reserve(num);

    vector<Mat> allImages;
    ReadImagesFromVideo(video, allImages);
    assert(startIndex < allImages.size());

    const int S = max(0, startIndex);
    const int N = num <= 0 ? allImages.size() - S : min(num, static_cast<int>(allImages.size()) - S);
    imgs = vector<Mat>(allImages.begin() + S, allImages.begin() + S + N);

    cout << "[Info ] Get " << imgs.size() << " images from tatal (from video)." << endl;
}

void ReadGroundtruthRectFromFolder(const string& folder, const string& suffix, vector<Mat>& masks,
                                   vector<Rect>& rects, int startIndex, int num)
{
    const int endIdx = startIndex + num - 1;

    string pre, suf;
    if (folder.back() != '/')
        pre = folder + "/";
    else
        pre = folder;

    if (suffix.front() != '.')
        suf = "." + suffix;
    else
        suf = suffix;

    ifstream ifs;
    string rect_param_file = pre + "rect_param.txt";
    ifs.open(rect_param_file);
    if (!ifs.is_open()) {
        rect_param_file = pre + "../image_rect/rect_param.txt";
        ifs.open(rect_param_file);
        if (!ifs.is_open()) {
            cerr << "Cannot open the file " << rect_param_file << endl;
            return;
        }
    }

    vector<Rect> vRects;
    vRects.reserve(num);
    string lineData;
    while (getline(ifs, lineData) && !lineData.empty()) {
        istringstream line(lineData);
        string oriImgLoc;
        int x1, y1, x2, y2;
        line >> oriImgLoc >> x1 >> y1 >> x2 >> y2;
        vRects.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
    }
    rects.swap(vRects);

    vector<Mat> vMasks;
    vMasks.reserve(num);
    for (int i = startIndex; i <= endIdx; ++i) {
        const string gtFile = pre + to_string(i) + suf;
        vMasks.push_back(imread(gtFile, IMREAD_GRAYSCALE));
    }

    cout << "[Info ] Read " << vMasks.size() << " gt files in " << folder << endl;

    masks.swap(vMasks);
}

void colorMask2Gray(const vector<Mat>& colors, vector<Mat>& grays)
{
    assert(!colors.empty());

    vector<Mat> vGrays;
    vGrays.reserve(colors.size());
    for (size_t i = 0, iend = colors.size(); i < iend; ++i) {
        Mat gray;
        cvtColor(colors[i], gray, COLOR_BGR2GRAY);  // 有颜色, 转灰度后不一定是255
        normalize(gray, gray, 0, 255, NORM_MINMAX);
        vGrays.push_back(gray);
    }
    grays.swap(vGrays);
}

void resizeFlipRotateImages(vector<Mat>& imgs, double scale, int flip, int rotate)
{
    assert(!imgs.empty());

    vector<Mat>& vImages = imgs;

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
}

void extractImagesToStitch(const vector<Mat>& vImages, vector<Mat>& vImagesToProcess, vector<int>& vIdxToProcess,
                           vector<vector<int>>& vvIdxPerIter, int minFores, int maxFores)
{
    assert(minFores > 2 && minFores <= maxFores);

    vImagesToProcess.clear();
    vIdxToProcess.clear();
    vvIdxPerIter.clear();

    const int N = vImages.size();
    if (maxFores > N) {
        cout << "输入图片数(" << N << ")少于最大前景数(" << maxFores << "), 全部处理." << endl;
        maxFores = N;
    }

    vImagesToProcess.reserve(maxFores - minFores);
    vIdxToProcess.reserve(maxFores - minFores);
    vvIdxPerIter.reserve(maxFores - minFores);

    std::set<int> sIdxToProcess;
    for (int k = minFores; k <= maxFores; ++k) {
        int d = cvCeil(N / k);
        int idx = 0;
        cout << "[前景数k = " << k << ", 间隔数d = " << d << "], 筛选的帧序号为: ";

        vector<int> vIdxThisIter, vIdxThisIterSubset;
        vIdxThisIter.reserve(k);
        while (idx < N) {
            sIdxToProcess.insert(idx);
            vIdxThisIter.push_back(idx);
            cout << idx << ", ";
            idx += d;
        }
        cout << " 实际个数 = " << vIdxThisIter.size();

        if (vIdxThisIter.size() > k) {
            vIdxThisIterSubset = vector<int>(vIdxThisIter.begin(), vIdxThisIter.begin() + k);
            cout << ", 实际个数过多, 去除最后几帧." << endl;
        } else {
            vIdxThisIterSubset = vIdxThisIter;
            cout << endl;
        }

        vvIdxPerIter.push_back(vIdxThisIterSubset);
    }

    vIdxToProcess = vector<int>(sIdxToProcess.begin(), sIdxToProcess.end());
    for (int i = 0, iend = vIdxToProcess.size(); i < iend; ++i)
        vImagesToProcess.push_back(vImages[vIdxToProcess[i]]);
}

void makeColorWheel(vector<Scalar>& colorwheel)
{
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    colorwheel.reserve(RY + YG + GC + CB + BM + MR);
    for (int i = 0; i < RY; i++)
        colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
    for (int i = 0; i < YG; i++)
        colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
    for (int i = 0; i < GC; i++)
        colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
    for (int i = 0; i < CB; i++)
        colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
    for (int i = 0; i < BM; i++)
        colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
    for (int i = 0; i < MR; i++)
        colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void flowToColor(const Mat& flow, Mat& color)
{
    if (color.empty())
        color.create(flow.rows, flow.cols, CV_8UC3);

    static vector<Scalar> colorwheel;  // Scalar r,g,b
    if (colorwheel.empty())
        makeColorWheel(colorwheel);

    // determine motion range:
    float maxrad = -1;

    // Find max flow to normalize fx and fy
    for (int i = 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);
            float fx = flow_at_point[0];
            float fy = flow_at_point[1];
            if ((fabs(fx) > 1e9) || (fabs(fy) > 1e9))
                continue;
            float rad = sqrt(fx * fx + fy * fy);
            maxrad = maxrad > rad ? maxrad : rad;
        }
    }

    for (int i = 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {
            uchar* data = color.data + color.step[0] * i + color.step[1] * j;
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);

            float fx = flow_at_point[0] / maxrad;
            float fy = flow_at_point[1] / maxrad;
            if ((fabs(fx) > 1e9) || (fabs(fy) > 1e9)) {
                data[0] = data[1] = data[2] = 0;
                continue;
            }
            float rad = sqrt(fx * fx + fy * fy);

            float angle = atan2(-fy, -fx) / CV_PI;
            float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
            int k0 = cvFloor(fk);
            int k1 = (k0 + 1) % colorwheel.size();
            float f = fk - k0;
            // f = 0; // uncomment to see original color wheel

            for (int b = 0; b < 3; b++) {
                float col0 = colorwheel[k0][b] / 255.0;
                float col1 = colorwheel[k1][b] / 255.0;
                float col = (1 - f) * col0 + f * col1;
                if (rad <= 1)
                    col = 1 - rad * (1 - col);  // increase saturation with radius
                else
                    col *= .75;  // out of range
                data[2 - b] = (int)(255.0 * col);
            }
        }
    }
}

void showFlow(const Mat& flow, Mat& color)
{
    Mat flow_uv[2], mag, ang, hsv, hsv_split[3], bgr;
    split(flow, flow_uv);
    multiply(flow_uv[1], -1, flow_uv[1]);
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true);  // 笛卡尔转极坐标系
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, bgr, COLOR_HSV2BGR);  // bgr1 type = CV_32FC3
    bgr.convertTo(color, CV_8UC3, 255, 0);
}


void drawhistogram(const Mat& src, Mat& histGray, const Mat& mask, int binSize)
{
    assert(src.channels() == 1);  // 8UC1
    assert(binSize < 128 && binSize >= 1);

    int channels = 0;
    int histSize = 256 / binSize;  // 51 / 256
    float range_[] = {0, 256};
    const float* ranges = {range_};
    Mat hist;
    calcHist(&src, 1, &channels, mask, hist, 1, &histSize, &ranges);
    //    cout << "hist.type = " << hist.type() << endl;
    //    cout << "hist = " << hist.t() << endl;

    // 直方图归一化
    normalize(hist, hist, 1.0, 0.0, NORM_MINMAX);  // 32FC1
                                                   //    cout << "hist norm = " << hist.t() << endl;

    // 在直方图中画出直方图
    const int boardSize = 5;
    const int imgSize = 256 + 2 * boardSize;
    histGray = Mat(imgSize, imgSize, CV_8UC1, Scalar(255));
    int hpt = static_cast<int>(0.9 * 256);
    for (int h = 0; h < histSize; h++) {
        const float binVal = hist.at<float>(h);
        if (binVal > 0) {
            const int intensity = static_cast<int>(binVal * hpt);
            line(histGray, Point(boardSize + h * binSize, boardSize + 256),
                 Point(boardSize + h * binSize, boardSize + 256 - intensity), Scalar(0), binSize);
        }
    }

    double maxValue, minValue;
    int minLoc, maxLoc;
    minMaxLoc(SparseMat(hist), &minValue, &maxValue, &minLoc, &maxLoc);
    cout << "min/max value/loc: " << minValue << "/" << maxValue << ", " << minLoc << "/" << maxLoc << endl;

    int th = maxLoc;
    while (--th > minLoc) {
        const float binVal = hist.at<float>(th);
        if (binVal < 0.1) {
            cout << "break value = " << binVal << ", loc = " << th << endl;
            break;
        }
    }
    Mat dst;
    threshold(src, dst, th * binSize, 255, THRESH_BINARY);
    imshow("src", src);
    imshow("dst threshold", dst);
    waitKey(0);
}

void drawFlowAndHist(const Mat& flow, Mat& flowGray, Mat& hist, Mat& histGraph, int chanel, int binSize)
{
    assert(flow.type() == CV_32FC2);  // CV_32FC2
    assert(binSize < 128 && binSize >= 1);
    assert(chanel == 1 || chanel == 2);

    // flow to weight mask
    flowGray.release();
    flowGray = Mat::zeros(flow.size(), CV_32FC1);
    for (int y = 0; y < flow.rows; ++y) {
        const Point2f* f_row = flow.ptr<Point2f>(y);
        float* fg_row = flowGray.ptr<float>(y);
        for (int x = 0; x < flow.cols; ++x)
            fg_row[x] = norm(Mat_<Point2f>(f_row[x]), NORM_L2);
    }
    normalize(flowGray, flowGray, 1.0, 0.0, NORM_MINMAX);
    flowGray.convertTo(flowGray, CV_8UC1, 255);
    imshow("flowGray", flowGray);


    // hist (32FC1)
    int histSize = 256 / binSize;  // 51 / 256
    float range_[] = {0, 256};
    const float* ranges = {range_};
    if (chanel == 1) {
        int channels = 0;
        calcHist(&flowGray, 1, &channels, Mat(), hist, 1, &histSize, &ranges);
    } else {
        int channels[2] = {0, 1};
        calcHist(&flowGray, 1, channels, Mat(), hist, 1, &histSize, &ranges);
    }
    normalize(hist, hist, 1.0, 0.0, NORM_MINMAX);  // 直方图归一化. 32FC1

    // 画出直方图
    const int boardSize = 5;
    const int imgSize = 256 + 2 * boardSize;
    histGraph = Mat(imgSize, imgSize, CV_8UC1, Scalar(255));
    int hpt = static_cast<int>(0.9 * 256);
    for (int h = 0; h < histSize; h++) {
        const float binVal = hist.at<float>(h);
        if (binVal > 0) {
            const int intensity = static_cast<int>(binVal * hpt);
            line(histGraph, Point(boardSize + h * binSize, boardSize + 256),
                 Point(boardSize + h * binSize, boardSize + 256 - intensity), Scalar(0), binSize);
        }
    }
    imshow("hist gray", histGraph);

    waitKey(30);
}

/**
 * @brief resultRoi 根据多个输入的图像/ROI区域, 生成一个覆盖了所有图像/ROI子区域的大区域
 * @param corners   多个图像/ROI区域的左上角坐标
 * @param images    多个图像对应的尺寸
 * @param sizes     多个ROI区域对应的尺寸
 * @return  返回覆盖了所有图像/ROI的大区域
 */
Rect resultRoi(const std::vector<Point>& corners, const std::vector<Size>& sizes)
{
    CV_Assert(sizes.size() == corners.size());
    Point tl(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    Point br(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
    for (size_t i = 0; i < corners.size(); ++i) {
        tl.x = std::min(tl.x, corners[i].x);
        tl.y = std::min(tl.y, corners[i].y);
        br.x = std::max(br.x, corners[i].x + sizes[i].width);
        br.y = std::max(br.y, corners[i].y + sizes[i].height);
    }
    return Rect(tl, br);
}

Rect resultRoi(const std::vector<Point>& corners, const std::vector<UMat>& images)
{
    std::vector<Size> sizes(images.size());
    for (size_t i = 0; i < images.size(); ++i)
        sizes[i] = images[i].size();
    return resultRoi(corners, sizes);
}


/**
 * @brief 为输入的二值掩模创建一个过渡区域
 * @param src   输入二值化掩模, 要么0, 要么255
 * @param dst   输出掩模, 边缘区域从0到255过渡
 * @param b1    缩小方向的过渡区域像素大小
 * @param b2    扩大方向的过渡区域像素大小
 */
void smoothMaskWeightEdge(const cv::Mat& src, cv::Mat& dst, int b1, int b2)
{
#define ENABLE_DEBUG_SMOOTH_MASK 0

    assert(src.type() == CV_8UC1);

    if (b1 < 2 && b2 < 2) {
        dst = src.clone();
        return;
    }

    Mat noWeightMask, weightArea, weightAreaValue, weightDistance;
    if (b1 >= 2 && b2 < 2) {  // 缩小
        const Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * b1, 2 * b1));
        erode(src, noWeightMask, kernel1);  // 先腐蚀得到不要过渡的区域
        weightArea = src - noWeightMask;    // 得到需要过渡的区域
        distanceTransform(src, weightDistance, DIST_C, 3);
        weightDistance.convertTo(weightDistance, CV_8UC1);  //! NOTE 32FC1 to 8UC1, 注意不能乘255
        bitwise_and(weightDistance, weightArea, weightAreaValue);  // 得到过渡区域的权重
        normalize(weightAreaValue, weightAreaValue, 0, 255, NORM_MINMAX);  // 归一化
        add(noWeightMask, weightAreaValue, dst);  // 不过渡区域(保留权值为1) + 过渡权区域 = 目标掩模
    } else if (b1 < 2 && b2 >= 2) {  // 扩大
        const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(2 * b2, 2 * b2));
        dilate(src, noWeightMask, kernel2);
        weightArea = noWeightMask - src;
        distanceTransform(noWeightMask, weightDistance, DIST_C, 3);
        weightDistance.convertTo(weightDistance, CV_8UC1);
        bitwise_and(weightDistance, weightArea, weightAreaValue);
        normalize(weightAreaValue, weightAreaValue, 0, 255, NORM_MINMAX);
        add(src, weightAreaValue, dst);
    } else {
        assert(b1 >= 2 && b2 >= 2);

        const Mat kernel1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * b1, 2 * b1));
        const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(2 * b2, 2 * b2));

        Mat tmp1, tmp2;
        erode(src, tmp1, kernel1);
        dilate(src, tmp2, kernel2);
        weightArea = tmp2 - tmp1;

#if ENABLE_DEBUG && ENABLE_DEBUG_SMOOTH_MASK
        imshow("weightArea", weightArea);
#endif
        distanceTransform(tmp2, weightDistance, DIST_C, 3);
        weightDistance.convertTo(weightDistance, CV_8UC1);
        bitwise_and(weightDistance, weightArea, weightAreaValue);
        normalize(weightAreaValue, weightAreaValue, 0, 255, NORM_MINMAX);

#if ENABLE_DEBUG && ENABLE_DEBUG_SMOOTH_MASK
        imshow("weightAreaValue", weightAreaValue);
        waitKey(0);
#endif
        add(tmp1, weightAreaValue, dst);
    }
}


/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param [in] img the input image
 * @param [in] x the position of x in the image
 * @param [in] y the position of y in the image
 * @return a float data for grayscale value in (x,y)
 */
float getPixelValue(const cv::Mat& img, float x, float y)
{
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float((1 - xx) * (1 - yy) * data[0] + xx * (1 - yy) * data[1] +
                 (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]);
}


//导向滤波器
Mat guidedFilter(const Mat& src, int radius, double eps)
{
    Mat srcMat, guidedMat, dstImage;
    vector<Mat> vInputs, vResults;

    if (src.channels() == 3) {
        split(src, vInputs);
        vResults.resize(3);
        for (int i = 0; i < 3; ++i)
            vResults[i] = guidedFilter(vInputs[i], radius, eps);
        merge(vResults, dstImage);
        return dstImage;
    }

    //------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
    src.convertTo(srcMat, CV_64FC1);
    src.convertTo(guidedMat, CV_64FC1, 1. / 255.);
    //--------------【1】各种均值计算----------------------------------
    Mat mean_p, mean_I, mean_Ip, mean_II;
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));  //生成待滤波图像均值mean_p
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));  //生成引导图像均值mean_I
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));  //生成互相关均值mean_Ip
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));  //生成引导图像自相关均值mean_II
    //--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    //---------------【3】计算参数系数a、b-------------------
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    //--------------【4】计算系数a、b的均值-----------------
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    //---------------【5】生成输出矩阵------------------
    dstImage = mean_a.mul(srcMat) + mean_b;
    dstImage.convertTo(dstImage, CV_8UC1);

    return dstImage;
}

void getArrowPoints(const Point& p, Point& s, Point& e, int dir)
{
    switch (dir) {
        case 1:

            break;
        default:
            break;
    }
}

/**
 * @brief 应用边缘滤波
 * @param img   输入输出图像, 单通道
 * @param x     边缘点横坐标
 * @param y     边缘点纵坐标
 * @param dir   边缘点梯度方向, (0, 4]分别表示: - | \ /
 */
void applyEdgeFilter(cv::Mat& img, int x, int y, int dir)
{
    assert(img.type() == CV_8UC1);  // 8UC1
    assert(dir > 0 && dir < 5);

    static vector<Mat> vKernels;
    if (vKernels.empty()) {
        vKernels.resize(4);
        vKernels[0] = (Mat_<float>(3, 3) << 3, 0, 3, 10, 0, 10, 3, 0, 3);
        vKernels[0] = vKernels[0] / sum(vKernels[0]);
        vKernels[1] = vKernels[0].t();
        vKernels[2] = (Mat_<float>(3, 3) << 10, 3, 0, 3, 0, 3, 0, 3, 10);
        vKernels[2] = vKernels[2] / sum(vKernels[2]);
        vKernels[3] = (Mat_<float>(3, 3) << 0, 3, 10, 3, 0, 3, 10, 3, 0);
        vKernels[3] = vKernels[3] / sum(vKernels[3]);
    }
//    if (vKernels.empty()) {
//        vKernels.resize(4);
//        vKernels[0] = (Mat_<float>(3, 3) << 1, 0, 1, 2, 0, 2, 1, 0, 1);
//        vKernels[0] = vKernels[0] / sum(vKernels[0]);
//        vKernels[1] = vKernels[0].t();
//        vKernels[2] = (Mat_<float>(3, 3) << 2, 1, 0, 1, 0, 1, 0, 1, 2);
//        vKernels[2] = vKernels[2] / sum(vKernels[2]);
//        vKernels[3] = (Mat_<float>(3, 3) << 0, 1, 2, 1, 0, 1, 2, 1, 0);
//        vKernels[3] = vKernels[3] / sum(vKernels[3]);
//    }

    if (x - 1 < 0 || y - 1 < 0 || x + 1 >= img.cols || y + 1 >= img.rows)
        return;

    const Mat& kernel = vKernels[dir - 1];
    const Mat& neighbor = img.rowRange(y - 1, y + 2).colRange(x - 1, x + 2);

    Mat result;
    multiply(neighbor, kernel, result, 1, CV_32FC1);
    img.at<uchar>(y, x) = static_cast<uchar>(sum(result)[0]);
}

void applyEdgeFilter(const cv::Mat& src, const cv::Mat& gradient, const cv::Mat& dirLabel, cv::Mat& dst)
{
    assert(src.type() == CV_8UC1);
    assert(dirLabel.type() == CV_8UC1);
    assert(src.size() == gradient.size());
    assert(src.size() == dirLabel.size());
    assert(gradient.channels() == 1);

    static vector<Mat> vKernels;
    if (vKernels.empty()) {
        vKernels.resize(5);
        vKernels[0] = (Mat_<float>(3, 3) << 3, 0, 3, 10, 0, 10, 3, 0, 3);
        vKernels[0] = vKernels[0] / sum(vKernels[0]);
        vKernels[1] = vKernels[0].t();
        vKernels[2] = (Mat_<float>(3, 3) << 10, 3, 0, 3, 0, 3, 0, 3, 10);
        vKernels[2] = vKernels[2] / sum(vKernels[2]);
        vKernels[3] = (Mat_<float>(3, 3) << 0, 3, 10, 3, 0, 3, 10, 3, 0);
        vKernels[3] = vKernels[3] / sum(vKernels[3]);
        vKernels[4] = (Mat_<float>(3, 3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
        vKernels[4] = vKernels[4] / sum(vKernels[4]);
    }

    Mat grad;
    gradient.convertTo(grad, CV_32FC1);
    normalize(grad, grad, 0.f, 1.f, NORM_MINMAX);

    dst = src.clone();
    for (int y = 1; y < src.rows - 1; ++y) {
        const float* grad_row = grad.ptr<float>(y);
        const uchar* label_row = dirLabel.ptr<uchar>(y);
        uchar* dst_row = dst.ptr<uchar>(y);

        for (int x = 1; x < src.cols - 1; ++x) {
            if (label_row[x] > 0/* && grad_row[x] > 0.01*/) {
                const Mat& kernel = vKernels[label_row[x] - 1];
                const Mat& neighbor = src.rowRange(y - 1, y + 2).colRange(x - 1, x + 2);
                Mat result;
                multiply(neighbor, kernel, result, 1, CV_32FC1);
                dst_row[x] = static_cast<uchar>(sum(result)[0]);
            }
        }
    }

    // 再做个最小值/均值滤波
    for (int y = 1; y < dst.rows - 1; ++y) {
        const float* grad_row = grad.ptr<float>(y);
        const uchar* label_row = dirLabel.ptr<uchar>(y);
        uchar* dst_row = dst.ptr<uchar>(y);

        for (int x = 1; x < dst.cols - 1; ++x) {
            if (label_row[x] > 0 && grad_row[x] > 0.5) {
                const Mat& kernel = vKernels[4];
                const Mat& neighbor = dst.rowRange(y - 1, y + 2).colRange(x - 1, x + 2);
                Mat result;
                multiply(neighbor, kernel, result, 1, CV_32FC1);
                dst_row[x] = static_cast<uchar>(sum(result)[0]);
            }
        }
    }
}

/**
 * @brief overlappedEdgesSmoothing  重叠区域的边缘平滑, 主要是为了消除前景重叠区域边缘处的锯齿
 * @param src   拼接后的前景图
 * @param mask  前景图中重叠区域的边缘掩模(降低计算量, 限制处理范围)
 * @param dst   结果图
 */
void overlappedEdgesSmoothing(const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, double scale)
{
#define ENABLE_DEBUG_EDGE_FILTER 1
#define SMOOTH_UPLOARD 0  // 梯度上采样后计算(反之下采样)

    assert(scale > 0 && scale <= 1.);
    const double invScale = 1. / scale;

    static vector<Mat> vKernels;
    if (vKernels.empty()) {
        vKernels.resize(4);
        vKernels[0] = (Mat_<char>(3, 3) << -3, 0, 3, -10, 0, 10, -3, 0, 3);
        vKernels[1] = vKernels[0].t();
        vKernels[2] = (Mat_<char>(3, 3) << -10, -3, 0, -3, 0, 3, 0, 3, 10);
        vKernels[3] = (Mat_<char>(3, 3) << 0, 3, 10, -3, 0, 3, -10, -3, 0);
    }

    static vector<Point3_<uchar>> vLableColor;
    if (vLableColor.empty()) {
        vLableColor.resize(5);
        vLableColor[0] = Point3_<uchar>(0, 0, 0);
        vLableColor[1] = Point3_<uchar>(0, 0, 255);
        vLableColor[2] = Point3_<uchar>(0, 255, 0);
        vLableColor[3] = Point3_<uchar>(255, 0, 0);
        vLableColor[4] = Point3_<uchar>(255, 255, 255);
    }

    // 1.下采样并在Y域上求解4个方向的梯度
    Mat srcGray, srcScaled, srcGrayScaled, maskScaled;
//    Mat src_yuv, src_luma, src_chroma;
//    vector<Mat> srcChannels_yuv(3);
//    cvtColor(src, src_yuv, COLOR_BGR2YCrCb);
//    split(src_yuv, srcChannels_yuv);

    resize(src, srcScaled, Size(), scale, scale);
    resize(mask, maskScaled, Size(), scale, scale);
    cvtColor(src, srcGray, COLOR_BGR2GRAY);
    resize(srcGray, srcGrayScaled, Size(), scale, scale);
    vector<Mat> vEdges_F(4);
    for (int i = 0; i < 4; ++i) {
        filter2D(srcGrayScaled, vEdges_F[i], CV_32FC1, vKernels[i], Point(-1, -1), 0, BORDER_REPLICATE);
#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
//        const string txt = "/home/vance/output/ms/初始梯度方向" + to_string(i + 1) + ".png";
//        imwrite(txt, cv::abs(vEdges_F[i]));
#endif
    }
#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
    Mat g12 = (cv::abs(vEdges_F[0]) + cv::abs(vEdges_F[1])) * 0.5;
    Mat g34 = (cv::abs(vEdges_F[2]) + cv::abs(vEdges_F[3])) * 0.5;
    normalize(g12, g12, 0.f, 1.f, NORM_MINMAX);
    g12.convertTo(g12, CV_8UC1, 255);
    normalize(g34, g34, 0.f, 1.f, NORM_MINMAX);
    g34.convertTo(g34, CV_8UC1, 255);
    bitwise_or(0, g12, g12, maskScaled);
    bitwise_or(0, g34, g34, maskScaled);
    imwrite("/home/vance/output/ms/初始梯度方向1+2.png", g12);
    imwrite("/home/vance/output/ms/初始梯度方向3+4.png", g34);
#endif

    // 2.对掩模对应的区域的梯度, 选出4个方向中的最大者并标记最大梯度方向
    Mat maxEdge_Full = Mat::zeros(srcGrayScaled.size(), CV_32FC1);
    Mat maxEdge_F = Mat::zeros(srcGrayScaled.size(), CV_32FC1);
    Mat dstLabel = Mat::zeros(srcGrayScaled.size(), CV_8UC1);   // 掩模区内边缘的方向
    Mat distLabelColor = Mat::zeros(srcGrayScaled.size(), CV_8UC3);
    for (int y = 0; y < srcGrayScaled.rows; ++y) {
        const char* mask_row = maskScaled.ptr<char>(y);
        float* dst_row = maxEdge_F.ptr<float>(y);
        char* label_row = dstLabel.ptr<char>(y);
        Point3_<uchar>* distLabelColor_row = distLabelColor.ptr<Point3_<uchar>>(y);

        float* tmp_row = maxEdge_Full.ptr<float>(y);

        for (int x = 0; x < srcGrayScaled.cols; ++x) {

//            if (mask_row[x] != 0) {
                int label = 0;  // 默认无梯度则黑色
                float maxGradient = 0.f;
                for (int i = 0; i < 4; ++i) {
                    const float g = abs(vEdges_F[i].at<float>(y, x)); // 绝对值符号使正负方向为同一方向
                    if (g > maxGradient) {
                        maxGradient = g;
                        label = i + 1;
                    }
                }
                tmp_row[x] = maxGradient;

                if (mask_row[x] != 0) {
                dst_row[x] = maxGradient;
                label_row[x] = label;
                distLabelColor_row[x] = vLableColor[label];
            }
        }
    }
    normalize(maxEdge_F, maxEdge_F, 0.f, 1.f, NORM_MINMAX);  // 归一化
    Mat finalEdge;
    maxEdge_F.convertTo(finalEdge, CV_8UC1, 255);
    threshold(finalEdge, finalEdge, 10, 0, THRESH_TOZERO);  // 太小的梯度归0

#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
    Mat tmp1;
    imwrite("/home/vance/output/ms/重叠区域最大梯度强度.png", finalEdge);
    imwrite("/home/vance/output/ms/重叠区域最大梯度方向.png", distLabelColor);

    normalize(maxEdge_Full, maxEdge_Full, 0.f, 1.f, NORM_MINMAX);
    maxEdge_Full.convertTo(tmp1, CV_8UC1, 255);
    imwrite("/home/vance/output/ms/初始梯度最大值.png", tmp1);
    Mat toShow = src.clone();
#endif

    // 3.应用边缘滤波
    vector<Mat> srcChannels(3), dstChanels(3);
    split(src, srcChannels);
    dstChanels = srcChannels;

    vector<Point> vPointsToSmooth;
    vector<int> vDirsToSmooth;
    vPointsToSmooth.reserve(10000);
    vDirsToSmooth.reserve(10000);

#if SMOOTH_UPLOARD
    resize(finalEdge, finalEdge, Size(), invScale, invScale);
    {
#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
//    namedLargeWindow("边缘处梯度(原始尺度)", true);
//    imshow("边缘处梯度(原始尺度)", finalEdge);
//    imwrite("/home/vance/output/ms/边缘处梯度(原始尺度).png", finalEdge);
#endif
        const Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(finalEdge, finalEdge, MORPH_OPEN, kernel);
#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
//    namedLargeWindow("边缘处梯度-平滑(原始尺度)", true);
//    imshow("边缘处梯度-平滑(原始尺度)", finalEdge);
//    imwrite("/home/vance/output/ms/边缘处梯度-平滑(原始尺度).png", finalEdge);
#endif
    }
    for (int y = 0; y < src.rows; ++y) {
        const int y_scaled = cvRound(y * scale);
        if (y_scaled >= dstLabel.rows)
            continue;

        const char* edge_row = finalEdge.ptr<char>(y);
        const char* label_row = dstLabel.ptr<char>(y_scaled);
        for (int x = 0; x < src.cols; ++x) {
            const int x_scaled = cvRound(x * scale);
            if (x_scaled >= dstLabel.cols)
                continue;

            if (edge_row[x] > 0) {
                applyEdgeFilter(dstChanels[0], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[1], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[2], x, y, label_row[x_scaled]);
//                vPointsToSmooth.push_back(Point(x, y));
//                vDirsToSmooth.push_back(label_row[x_scaled]);
            }
        }
    }
#else
    // 先在低尺度上滤波一次, 然后上采样回原尺度加权融合
//    vector<Mat> vDstScaledFilterd(3), vDstScaled(3);
//    resize(dstChanels[0], vDstScaled[0], Size(), scale, scale);
//    resize(dstChanels[1], vDstScaled[1], Size(), scale, scale);
//    resize(dstChanels[2], vDstScaled[2], Size(), scale, scale);
//    for (int i = 0; i < 3; ++i)
//        applyEdgeFilter(vDstScaled[i], finalEdge, dstLabel, vDstScaledFilterd[i]);

//    Mat dstScaledFilterd, dstFilterdOnce;
//    merge(vDstScaledFilterd, dstScaledFilterd);
//    resize(dstScaledFilterd, dstFilterdOnce, src.size());

//    imwrite("/home/vance/output/ms/前景(scaled).png", srcScaled);
//    imwrite("/home/vance/output/ms/前景(scaled)低尺度滤波.png", dstScaledFilterd);
//    imwrite("/home/vance/output/ms/前景低尺度滤波.png", dstFilterdOnce);

//    dst = src.clone();
//    dst.setTo(0, mask);

//    bitwise_or(dst, dstFilterdOnce, dst, mask);
//    imwrite("/home/vance/output/ms/前景低尺度滤波应用到原尺度.png", dst);
//    split(dst, dstChanels);

    // 再到原尺度上滤波
    for (int y = 0; y < src.rows; ++y) {
        const int y_scaled = cvRound(y * scale);
        if (y_scaled >= dstLabel.rows)
            continue;

        const uchar* edge_row = finalEdge.ptr<uchar>(y_scaled);
        const uchar* label_row = dstLabel.ptr<uchar>(y_scaled);
        for (int x = 0; x < src.cols; ++x) {
            const int x_scaled = cvRound(x * scale);
            if (x_scaled >= dstLabel.cols)
                continue;

            if (label_row[x_scaled] > 0 /*&& edge_row[x_scaled] > 50*/) {
                applyEdgeFilter(dstChanels[0], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[1], x, y, label_row[x_scaled]);
                applyEdgeFilter(dstChanels[2], x, y, label_row[x_scaled]);
//                vPointsToSmooth.push_back(Point(x, y));
//                vDirsToSmooth.push_back(label_row[x_scaled]);
                toShow.at<Vec3b>(y, x) = Vec3b(0,255,0);
            }
        }
    }
#endif
//    applyEdgeFilter(srcChannels[0], dstChanels[0], vPointsToSmooth, vDirsToSmooth);
//    applyEdgeFilter(srcChannels[1], dstChanels[1], vPointsToSmooth, vDirsToSmooth);
//    applyEdgeFilter(srcChannels[2], dstChanels[2], vPointsToSmooth, vDirsToSmooth);
    merge(dstChanels, dst);

#if ENABLE_DEBUG && ENABLE_DEBUG_EDGE_FILTER
    Mat input = src.clone(), output = dst.clone();
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    drawContours(input, contours, -1, Scalar(0, 255, 0), 1);
    drawContours(output, contours, -1, Scalar(0, 255, 0), 1);
    //    namedLargeWindow("points to smooth", 1);
    namedLargeWindow("src", 1);
    namedLargeWindow("dst", 1);
    //    imshow("src", input);
    //    imshow("dst", dst);
    imwrite("/home/vance/output/ms/输入前景和边缘掩模.png", input);
    imwrite("/home/vance/output/ms/输出图像和边缘掩模.png", output);
    imwrite("/home/vance/output/ms/被处理的区域.png", toShow);
    //    imshow("points to smooth", toShow);
    waitKey(10);
    destroyAllWindows();
#endif
}

}  // namespace ms
