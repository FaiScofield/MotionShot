#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <set>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

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
        cerr << "[Error] No image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read " << allImages.size() << " images in the folder." << endl;
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
        cerr << "[Error] No gt image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read " << allImages.size() << " gt images in the folder." << endl;
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
        cerr << "[Error] No image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read tatal " << allImages.size() << " images in the folder." << endl;
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

    cout << "[Info ] Read " << imgs.size() << " images in the sequence." << endl;
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

void extractImagesToStitch(const vector<Mat>& vImages, vector<Mat>& vImagesToProcess,
                           vector<size_t>& vIdxToProcess, vector<vector<size_t>>& vvIdxPerIter,
                           size_t minFores = 3, size_t maxFores = 8)
{
    assert(minFores > 2 && minFores <= maxFores);

    vImagesToProcess.clear();
    vIdxToProcess.clear();
    vvIdxPerIter.clear();

    const size_t N = vImages.size();
    if (maxFores > N) {
        cout << "输入图片数(" << N << ")少于最大前景数(" << maxFores << "), 全部处理." << endl;
        maxFores = N;
    }

    vImagesToProcess.reserve(maxFores - minFores);
    vIdxToProcess.reserve(maxFores - minFores);
    vvIdxPerIter.reserve(maxFores - minFores);

    std::set<size_t> sIdxToProcess;
    for (size_t k = minFores; k <= maxFores; ++k) {
        size_t d = N / k;
        size_t idx = 0;
        cout << "[前景数k = " << k << ", 间隔数d = " << d << "], 筛选的帧序号为: ";

        vector<size_t> vIdxThisIter;
        vIdxThisIter.reserve(k);
        while (idx < N) {
            sIdxToProcess.insert(idx);
            vIdxThisIter.push_back(idx);
            cout << idx << ", ";
            idx += d;
        }
        cout << endl;

        vvIdxPerIter.push_back(vIdxThisIter);
    }

    vIdxToProcess = vector<size_t>(sIdxToProcess.begin(), sIdxToProcess.end());
    for (size_t i = 0, iend = vIdxToProcess.size(); i < iend; ++i)
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
    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true); // 笛卡尔转极坐标系
    normalize(mag, mag, 0, 1, NORM_MINMAX);
    hsv_split[0] = ang;
    hsv_split[1] = mag;
    hsv_split[2] = Mat::ones(ang.size(), ang.type());
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, bgr, COLOR_HSV2BGR);    // bgr1 type = CV_32FC3
    bgr.convertTo(color, CV_8UC3, 255, 0);
}


void drawhistogram(const Mat& src, Mat& histGray, const Mat& mask, int binSize)
{
    assert(src.channels() == 1);  // 8UC1
    assert(binSize < 128 && binSize >= 1);

    int channels = 0;
    int histSize = 256 / binSize; // 51 / 256
    float range_[] = {0, 256};
    const float* ranges = {range_};
    Mat hist;
    calcHist(&src, 1, &channels, mask, hist, 1, &histSize, &ranges);
//    cout << "hist.type = " << hist.type() << endl;
//    cout << "hist = " << hist.t() << endl;

    // 直方图归一化
    normalize(hist, hist, 1.0, 0.0, NORM_MINMAX); // 32FC1
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

void drawFlowAndHist(const Mat& flow, Mat& flowGray, Mat& hist, Mat& histGraph, int chanel,int binSize)
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
    int histSize = 256 / binSize; // 51 / 256
    float range_[] = {0, 256};
    const float* ranges = {range_};
    if (chanel == 1) {
        int channels = 0;
        calcHist(&flowGray, 1, &channels, Mat(), hist, 1, &histSize, &ranges);
    } else {
        int channels[2] = {0, 1};
        calcHist(&flowGray, 1, channels, Mat(), hist, 1, &histSize, &ranges);
    }
    normalize(hist, hist, 1.0, 0.0, NORM_MINMAX); // 直方图归一化. 32FC1

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

void shrinkRoi(const Mat& src, Mat& dst, int size)
{
    assert(src.type() == CV_8UC1);

//    imshow("origin mask", src);

//    Mat tmp1, tmp2;
//    cvtColor(src, tmp1, COLOR_GRAY2BGR);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    erode(src, dst, kernel, Point(-1,-1), 1, BORDER_CONSTANT);
//    imshow("mask shrink", dst);

//    waitKey(0);
}

void smoothMaskWeightEdge(const cv::Mat& src, cv::Mat& dst, int size)
{
    assert(src.type() == CV_8UC1);

    Mat noWeightMask;
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));
    erode(src, noWeightMask, kernel, Point(-1, -1), 1, BORDER_CONSTANT); // 先腐蚀得到不要过渡的区域

    Mat distance, weightArea, weightAreaValue;
    bitwise_xor(src, noWeightMask, weightArea); // 得到需要过渡的区域
    distanceTransform(src, distance, DIST_C, 3);
//    imshow("distance 32F", distance);
    distance.convertTo(distance, CV_8UC1);  //! NOTE 32FC1 to 8UC1, 注意不能乘255
//    imshow("distance 8U", distance);

    bitwise_and(distance, weightArea, weightAreaValue); // 得到过渡区域的权重
    normalize(weightAreaValue, weightAreaValue, 0, 255, NORM_MINMAX);   // 归一化
//    imshow("weightAreaValue", weightAreaValue);

    add(noWeightMask, weightAreaValue, dst); // 不过渡区域(保留权值为1) + 过渡权区域 = 目标掩模
//    imshow("norm distance 8U", dst);
//    waitKey(0);
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


}  // namespace ms

#endif  // UTILITY_HPP
