#ifndef PRECOMPILED_H
#define PRECOMPILED_H

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


#ifdef DEBUG
#include <opencv2/highgui.hpp>
#endif

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;

using cv::Mat;
using cv::Mat_;
using cv::Point;
using cv::Point2i;
using cv::Point2f;
using cv::Rect;
using cv::Size;
using cv::UMat;
using cv::Ptr;
using cv::String;
using cv::KeyPoint;
using cv::DMatch;
using cv::InputArray;
using cv::InputArrayOfArrays;
using cv::OutputArray;
using cv::OutputArrayOfArrays;
using cv::InputOutputArray;
using cv::InputOutputArrayOfArrays;


#define INFO(msg) (std::cout << "[INFO ] " << msg << std::endl)
#define WARNING(msg) (std::cerr << "\033[33m[WARNI] " << msg << "\033[0m" << std::endl)
#define ERROR(msg) (std::cerr << "\033[31m[ERROR] " << msg << "\033[0m (in file \"" \
                    << __FILE__ << "\", at line " << __LINE__ << ")" << std::endl)
#define TIMER(msg) (std::cout << "\033[32m[TIMER] " << msg << "\033[0m" << std::endl)
#define ATTENTION(msg) (std::cout << "\033[35m[ATTEN] " << msg << "\033[0m" << std::endl)


#ifndef MS_ABANDON
#define MS_ABANDON
#define MS_DEBUG_TO_DELETE MS_ABANDON
#endif

MS_DEBUG_TO_DELETE
#include <opencv2/highgui.hpp>

#endif // PRECOMPILED_H
