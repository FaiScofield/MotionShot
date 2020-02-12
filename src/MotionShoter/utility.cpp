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
            // format: I_IL_01-GT_29.bmp
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

void ReadImageSequence_lasisesta(const string& folder, vector<Mat>& imgs,
                                 vector<Mat>& gts, int startIndex, int num)
{
    imgs.clear();

    vector<Mat> allImages, allGts;
    string gtFolder;
    if (folder.back() == '/')
        gtFolder = folder.substr(0, folder.size() - 1) + "-GT/";
    else
        gtFolder = folder + "-GT/";

    ReadImagesFromFolder_lasisesta(folder, allImages);
    ReadImagesFromFolder_lasisesta(gtFolder, allGts);
    assert(startIndex < allImages.size());

    const int S = max(0, startIndex);
    const int N = num <= 0 ? allImages.size() - S : min(num, static_cast<int>(allImages.size()) - S);
    imgs = vector<Mat>(allImages.begin() + S, allImages.begin() + S + N);
    gts = vector<Mat>(allGts.begin() + S, allGts.begin() + S + N);

    assert(gts.size() == imgs.size());
    cout << "[Info ] Read " << imgs.size() << " images in the folder." << endl;
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

void ReadImageSequence(const string& prefix, const string& suffix, vector<Mat>& imgs,
                       int startIndex, int num)
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
        imgs.push_back(img.clone()); //! 注意浅拷贝问题

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

}  // namespace ms

#endif  // UTILITY_HPP
