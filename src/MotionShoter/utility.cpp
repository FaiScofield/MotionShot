#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

using namespace cv;
using namespace std;
namespace bf = boost::filesystem;

void ReadImagesFromFolder_lasisesta(const string& folder, vector<Mat>& imgs)
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

void ReadGroundtruthFromFolder_lasisesta(const string& folder, vector<Mat>& imgs)
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

void ReadImageSequence_lasisesta(const string& folder, vector<Mat>& imgs, vector<Mat>& gts, int startIndex, int num)
{
    imgs.clear();

    vector<Mat> allImages, allGts;
    string gtFolder;
    if (folder.back() == '/')
        gtFolder = folder.substr(0, folder.size() - 1) + "-GT/";
    else
        gtFolder = folder + "-GT/";

    ReadImagesFromFolder_lasisesta(folder, allImages);
    ReadGroundtruthFromFolder_lasisesta(gtFolder, allGts);
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

void drawhistogram(const Mat& src, Mat& histColor)
{
    assert(src.channels() == 1);

    int channels = 0;
    int histSize = 256;
    float range_[] = {0, 256};
    const float* ranges = {range_};
    Mat hist;
    calcHist(&src, 1, &channels, Mat(), hist, 1, &histSize, &ranges);

    // 创建直方图画布
    int nHistWidth = 600;
    int nHistHeight = 400;
    int nBinWidth = cvRound((double)nHistWidth / histSize);
    histColor = Mat(nHistHeight, nHistWidth, CV_8UC3, Scalar(255, 255, 255));

    // 直方图归一化
    normalize(hist, hist, 0.0, histColor.rows, NORM_MINMAX, -1, Mat());

    // 在直方图中画出直方图
    for (int i = 1; i < histSize; i++) {
        line(histColor, Point(nBinWidth * (i - 1), nHistHeight - cvRound(hist.at<float>(i - 1))),
             Point(nBinWidth * i, nHistHeight - cvRound(hist.at<float>(i))), Scalar(0, 0, 0), 2, 8, 0);
    }
}

}  // namespace ms

#endif  // UTILITY_HPP
