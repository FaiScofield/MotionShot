#ifndef UTILITY_HPP
#define UTILITY_HPP

#include <boost/filesystem.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ms
{

using namespace std;
namespace bf = boost::filesystem;

void ReadImageFiles(const string& folder, vector<string>& files)
{
    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist!" << endl;
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
        cerr << "[Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read " << allImages.size() << " image files in the folder." << endl;
    }

    files.clear();
    files.reserve(allImages.size());
    for (auto it = allImages.begin(), iend = allImages.end(); it != iend; it++)
        files.push_back(it->second);
}

void ReadImageGTFiles(const string& folder, vector<string>& gtFiles)
{
    bf::path path(folder);
    if (!bf::exists(path)) {
        cerr << "[Error] Data folder doesn't exist!" << endl;
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
        cerr << "[Error] Not image data in the folder!" << endl;
        return;
    } else {
        cout << "[Info ] Read " << allImages.size() << " image files in the folder." << endl;
    }

    gtFiles.clear();
    gtFiles.reserve(allImages.size());
    for (auto it = allImages.begin(), iend = allImages.end(); it != iend; it++)
        gtFiles.push_back(it->second);

}

}  // namespace ms

#endif  // UTILITY_HPP
