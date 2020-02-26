#include "OpticalFlower.h"
#include "utility.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <vector>

namespace ms
{

using namespace std;
using namespace cv;

OpticalFlower::OpticalFlower()
{
#ifdef USE_OPENCV4
    _denseFlow = dynamic_cast<DenseOpticalFlow*>(
        cv::DISOpticalFlow::create(cv::DISOpticalFlow::PRESET_MEDIUM).get());
#else
    //    _denseFlow = FarnebackOpticalFlow::create();
    _denseFlow = DualTVL1OpticalFlow::create();

//    _denseFlow = dynamic_cast<DenseOpticalFlow*>(FarnebackOpticalFlow::create().get());
//    _denseFlow = dynamic_cast<DenseOpticalFlow*>(DualTVL1OpticalFlow::create().get());
#endif
}

void OpticalFlower::setCompareToPano(const Mat& pano)
{
    pano.copyTo(_pano);
    cvtColor(pano, _panoGray, COLOR_BGR2GRAY);
    _compareToPano = true;
}

void OpticalFlower::apply(const Mat& img, Mat& mask)
{
    if (img.channels() == 3)
        cvtColor(img, _imgCurr, COLOR_BGR2GRAY);
    else if (img.channels() == 1)
        _imgCurr = img.clone();

    if (_compareToPano) {
        _denseFlow->calc(_panoGray, _imgCurr, _flow);
    } else {
        if (isFirstFrame()) {
            _imgLast = _imgCurr.clone();
            setFirstFrameFlag(false);
            mask.release();
            return;
        }

        _denseFlow->calc(_imgLast, _imgCurr, _flow);
    }

    Mat flowColor;
    showFlow(_flow, flowColor);
    //    imshow("flowColor", flowColor);
    //    waitKey(30);

    flowToWeightMask(_flow, _weightMask);
    mask = _weightMask.clone();
    //    flowToMask(flowColor, mask);

    // TODO
    if (!_compareToPano)
        _imgLast = _imgCurr.clone();
}

void OpticalFlower::apply(const std::vector<cv::Mat>& vImgs, std::vector<cv::Mat>& vMasks)
{
    assert(!vImgs.empty());

    const size_t N = vImgs.size();

    vector<Mat> imgs;
    imgs.reserve(N);
    if (vImgs[0].channels() == 3)
        for_each(vImgs.begin(), vImgs.end(), [&](const Mat& img) {
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            imgs.push_back(gray);
        });
    else if (vImgs[0].channels() == 1)
        imgs = vImgs;

    vMasks.clear();
    vMasks.resize(N);
    Mat lastFrame = imgs[0], currFrame;
    for (size_t i = 1; i < N; ++i) {
        Mat flow, weight;
        currFrame = imgs[i];
        _denseFlow->calc(lastFrame /*imgs[0]*/, currFrame, flow);
        flowToWeightMask(flow, weight);
        vMasks[i] = weight.clone();

        //        Mat flowColor;
        //        showFlow(flow, flowColor);
        //        imshow("flowColor", flowColor);
        //        imshow("flow weight", weight);
        //        waitKey(30);
    }

    // 反向计算首帧的光流
    Mat firstFlow, firstWeight;
    _denseFlow->calc(imgs[1], imgs[0], firstFlow);
    flowToWeightMask(firstFlow, firstWeight);
    vMasks[0] = firstWeight.clone();
}

// void OpticalFlower::flowToMask(const Mat &flow, Mat &mask)
//{
//    assert(flow.channels() == 2);

//    Mat flow_uv[2], mag, ang, hsv, hsv_split[3], bgr, color;
//    split(flow, flow_uv);
//    multiply(flow_uv[1], -1, flow_uv[1]);
//    cartToPolar(flow_uv[0], flow_uv[1], mag, ang, true); // 笛卡尔转极坐标系
//    normalize(mag, mag, 0, 1, NORM_MINMAX);
//    hsv_split[0] = ang;
//    hsv_split[1] = mag;
//    hsv_split[2] = Mat::ones(ang.size(), ang.type());
//    merge(hsv_split, 3, hsv);
//    cvtColor(hsv, bgr, COLOR_HSV2BGR);    // bgr1 type = CV_32FC3
//    bgr.convertTo(color, CV_8UC3, 255, 0);

//    const Mat kernel = getStructuringElement(MORPH_RECT, Size(structureSize(), structureSize()));

//    Mat flowU;
//    cvtColor(color, flowU, COLOR_BGR2GRAY);
//    bitwise_not(flowU, flowU);
//    threshold(flowU, flowU, 30, 255, THRESH_BINARY); // THRESH_OTSU, THRESH_BINARY
//    erode(flowU, flowU, kernel);
//    dilate(flowU, flowU, kernel);
//    dilate(flowU, flowU, kernel);

//    mask = flowU.clone();
//}

void OpticalFlower::flowToWeightMask(const Mat& flow, Mat& weight)
{
    assert(flow.channels() == 2);

    Mat flow_uv[2], mag, ang, hsv, hsv_split[3], bgr, color;
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

    cvtColor(color, weight, COLOR_BGR2GRAY);
    bitwise_not(weight, weight);
}


void OpticalFlower::calcOpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2,
                                               const vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2,
                                               vector<bool>& success, bool inverse)
{
    // parameters
    int half_patch_size = 50;
    int iterations = 10;
    bool have_initial = !kp2.empty();

    for (size_t i = 0; i < kp1.size(); i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0;  // dx,dy need to be estimated
        if (have_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;  // indicate if this point succeeded

        // Gauss-Newton iterations
        for (int iter = 0; iter < iterations; iter++) {
            Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
            Eigen::Vector2d b = Eigen::Vector2d::Zero();
            cost = 0;

            if (kp.pt.x + dx <= half_patch_size || kp.pt.x + dx >= img1.cols - half_patch_size ||
                kp.pt.y + dy <= half_patch_size || kp.pt.y + dy >= img1.rows - half_patch_size) {  // go outside
                succ = false;
                break;
            }

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) {
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    // TODO START YOUR CODE HERE (~8 lines)
                    double error = 0;
                    error = -getPixelValue(img1, kp.pt.x + x, kp.pt.y + y) +
                            getPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    Eigen::Vector2d J = Eigen::Vector2d::Zero();  // Jacobian
                    if (inverse == false) {
                        // Forward Jacobian
                        J[0] = (getPixelValue(img2, kp.pt.x + x + dx + 1, kp.pt.y + y + dy) -
                                getPixelValue(img2, kp.pt.x + x + dx - 1, kp.pt.y + y + dy)) /
                               2;
                        J[1] = (getPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy + 1) -
                                getPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy - 1)) /
                               2;
                    } else {
                        // Inverse Jacobian
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J[0] = (getPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                getPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)) /
                               2;
                        J[1] = (getPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                getPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1)) /
                               2;
                    }

                    // compute H, b and set cost;
                    H += J * J.transpose();
                    b += -error * J;
                    cost += error * error;
                    // TODO END YOUR CODE HERE
                }
            }

            // compute update
            // TODO START YOUR CODE HERE (~1 lines)
            Eigen::Vector2d update = Eigen::Vector2d::Zero();
            update = H.ldlt().solve(b);
            // TODO END YOUR CODE HERE

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) {
                //                cout << "point " << i << " tatal iteration: " << iter << endl;
                //                cout << "point " << i << " cost increased: " << cost << " --> " << lastCost << endl;
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
        }


        success.push_back(succ);

        // set kp2
        if (have_initial) {
            kp2[i].pt = kp.pt + Point2f(dx, dy);
        } else {
            KeyPoint tracked = kp;
            tracked.pt += cv::Point2f(dx, dy);
            kp2.push_back(tracked);
        }
    }
}

void OpticalFlower::calcOpticalFlowMultiLevel(const cv::Mat& img1, const cv::Mat& img2,
                                              const vector<cv::KeyPoint>& kp1, vector<cv::KeyPoint>& kp2,
                                              vector<bool>& success, int nLevel, bool inverse)
{

    // parameters
    int pyramids = nLevel;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2;  // image pyramids
    // TODO START YOUR CODE HERE (~8 lines)
    vector<vector<cv::KeyPoint>> kp1_pyr;
    vector<cv::KeyPoint> kp1_tmp(kp1.size());
    for (int i = 0; i < pyramids; i++) {
        cv::Mat img1_tmp, img2_tmp;
        resize(img1, img1_tmp, cv::Size(img1.cols * scales[i], img1.rows * scales[i]));
        resize(img2, img2_tmp, cv::Size(img1.cols * scales[i], img1.rows * scales[i]));
        pyr1.push_back(img1_tmp);
        pyr2.push_back(img2_tmp);

        kp1_tmp.clear();
        for (size_t j = 0; j < kp1.size(); j++) {
            cv::KeyPoint kp_tmp;
            kp_tmp.pt.x = kp1[j].pt.x * scales[i];
            kp_tmp.pt.y = kp1[j].pt.y * scales[i];
            kp1_tmp.push_back(kp_tmp);
        }
        kp1_pyr.push_back(kp1_tmp);
    }
    // TODO END YOUR CODE HERE

    // coarse-to-fine LK tracking in pyramids
    // TODO START YOUR CODE HERE
    for (int i = pyramids - 1; i >= 0; i--) {
        success.clear();  // clear data

        calcOpticalFlowSingleLevel(pyr1[i], pyr2[i], kp1_pyr[i], kp2, success, inverse);

        // 上层计算出的kp2作为下层的输入,坐标值要先对应到下层
        for (size_t j = 0; j < kp2.size(); j++) {
            if (i != 0) {
                kp2[j].pt.x /= pyramid_scale;
                kp2[j].pt.y /= pyramid_scale;
            }
        }
    }
    // TODO END YOUR CODE HERE
    // don't forget to set the results into kp2
}

Mat derivative_x(Mat img1, Mat img2)
{
    float arr[2][2] = {{-1.0, 1.0}, {-1.0, 1.0}};
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_x, img2_x;
    filter2D(img1, img1_x, CV_32FC1, filter);
    filter2D(img2, img2_x, CV_32FC1, filter);
    return img1_x + img2_x;
}

Mat derivative_y(Mat img1, Mat img2)
{
    float arr[2][2] = {{-1.0, -1.0}, {1.0, 1.0}};
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_y, img2_y;
    filter2D(img1, img1_y, CV_32FC1, filter);
    filter2D(img2, img2_y, CV_32FC1, filter);
    return img1_y + img2_y;
}

Mat color_coding(Mat x, Mat y)
{
    Mat magnitude, angle;
    cartToPolar(x, y, magnitude, angle, true);

    double mag_max;
    minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    merge(_hsv, 3, hsv);

    // convert to BGR and show
    Mat bgr;  // CV_32FC3 matrix
    cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    return bgr;
}


Mat derivative_t(Mat img1, Mat img2)
{
    float arr[2][2] = {{-1.0, -1.0}, {-1.0, -1.0}};
    Mat filter = Mat(2, 2, CV_32FC1, arr), img1_t, img2_t;
    filter2D(img1, img1_t, CV_32FC1, filter);
    filter *= -1;
    filter2D(img2, img2_t, CV_32FC1, filter);
    return img1_t + img2_t;
}

Mat optical_flow_horn_schunk(Mat img1, Mat img2, float lambda, int iter)
{
    Mat img_x = derivative_x(img1, img2), img_y = derivative_y(img1, img2), img_t = derivative_t(img1, img2);
    Mat u_current = Mat::zeros(img1.rows, img1.cols, CV_32F);
    Mat v_current = Mat::zeros(img1.rows, img1.cols, CV_32F);
    Mat u_last, v_last;
    for (int k = 0; k < iter; k++) {
        u_current.copyTo(u_last);
        v_current.copyTo(v_last);
        for (int i = 0; i < img1.rows; i++)
            for (int j = 0; j < img1.cols; j++) {
                float u_avg = 0.0, v_avg = 0.0, n = 0;
                if (i - 1 >= 0) {
                    u_avg += u_last.at<float>(i - 1, j);
                    v_avg += v_last.at<float>(i - 1, j);
                    n++;
                }
                if (i + 1 < img1.cols) {
                    u_avg += u_last.at<float>(i + 1, j);
                    v_avg += v_last.at<float>(i + 1, j);
                    n++;
                }
                if (j - 1 >= 0) {
                    u_avg += u_last.at<float>(i, j - 1);
                    v_avg += v_last.at<float>(i, j - 1);
                    n++;
                }
                if (j + 1 < img1.rows) {
                    u_avg += u_last.at<float>(i, j + 1);
                    v_avg += v_last.at<float>(i, j + 1);
                    n++;
                }
                u_avg /= n;
                v_avg /= n;

                float p = img_x.at<float>(i, j) * u_avg + img_y.at<float>(i, j) * v_avg +
                          img_t.at<float>(i, j);
                float d = lambda + img_x.at<float>(i, j) * img_x.at<float>(i, j) +
                          img_y.at<float>(i, j) * img_y.at<float>(i, j);

                u_current.at<float>(i, j) = u_avg - img_x.at<float>(i, j) * p / d;
                v_current.at<float>(i, j) = v_avg - img_y.at<float>(i, j) * p / d;
            }
    }
    return color_coding(u_current, v_current);
}

template <typename T>
void hornSchunck(const Mat& src1, const Mat& src2, Mat& dstX, Mat& dstY, const double alpha, const uint iterations)
{
    T laplaceu, laplacev;
    const size_t width = src1.width();
    const size_t height = src1.height();

    Mat dx(width, height);
    Mat dy(width, height);
    Mat dt(width, height);

    Mat tmpX(width, height);
    Mat tmpY(width, height);

    // gradient calculation
    for (uint y = 0; y < height; ++y) {
        for (uint x = 0; x < width; ++x) {
            dx(x, y) = .5 * (src1.getMirrored(x + 1, y) - src1.getMirrored(x - 1, y) +
                             src2.getMirrored(x + 1, y) - src2.getMirrored(x - 1, y));
            dy(x, y) = .5 * (src1.getMirrored(x, y + 1) - src1.getMirrored(x, y - 1) +
                             src2.getMirrored(x, y + 1) - src2.getMirrored(x, y - 1));
            dt(x, y) = src2(x, y) - src1(x, y);
        }
    }

    // flow calculation
    for (uint i = 0; i < iterations; ++i) {
        for (uint y = 0; y < height; ++y) {
            for (uint x = 0; x < width; ++x) {
                laplaceu = (dstX.getMirrored(x + 1, y) - 2 * dstX(x, y) + dstX.getMirrored(x - 1, y)) +
                           (dstX.getMirrored(x, y + 1) - 2 * dstX(x, y) + dstX.getMirrored(x, y - 1));
                laplacev = (dstY.getMirrored(x + 1, y) - 2 * dstY(x, y) + dstY.getMirrored(x - 1, y)) +
                           (dstY.getMirrored(x, y + 1) - 2 * dstY(x, y) + dstY.getMirrored(x, y - 1));

                tmpX(x, y) = (-dx(x, y) * dt(x, y) - (dx(x, y) * dy(x, y) * dstY(x, y) - alpha * laplaceu)) /
                             (dx(x, y) * dx(x, y) + alpha * 8.);
                tmpY(x, y) = (-dy(x, y) * dt(x, y) - (dx(x, y) * dy(x, y) * dstX(x, y) - alpha * laplacev)) /
                             (dy(x, y) * dy(x, y) + alpha * 8.);
            }
        }
        dstX.swap(tmpX);
        dstY.swap(tmpY);
    }
}

}  // namespace ms
