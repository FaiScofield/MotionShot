#include "MotionShoter.h"
#include "utility.h"

#include <iostream>
#include <map>
#include <opencv2/video/video.hpp>

#include <jsoncpp/json/json.h>
#include <curl/curl.h>

namespace ms
{

using namespace std;
using namespace cv;

MotionShot::MotionShot()
{
    stitcher_ = ImageStitcher::create(ImageStitcher::ORB, ImageStitcher::BF);
    stitcher_->setScales(0.25, 0.1, 1.);
    blender_ = cvBlender::createDefault(cvBlender::FEATHER);
    detector_ = cv::createBackgroundSubtractorMOG2();
//    detector_ = cv::bgsegm::createBackgroundSubtractorGMG();

    // for NN-based body segmentation
    const string app_id = "18679385";
    const string aip_key = "12yptwaZPOxoGBfPR0PGYT43";
    const string secret_key = "sx8w8l1dzlD2Gt0QAKgZxItRB3uE8DZz";
    bodyAnalyst_ = std::unique_ptr<aip::Bodyanalysis>(new aip::Bodyanalysis(app_id, aip_key, secret_key));
//    bodyAnalyst_ = std::make_unique<aip::Bodyanalysis>(app_id, aip_key, secret_key);    // c++ 14
    bodySegOptions_["type"] = "scoremap";
}

MotionShot::Status MotionShot::setInputs(InputArrayOfArrays _imgs, InputArrayOfArrays _masks)
{
    _imgs.getUMatVector(inputImages_);
    _masks.getUMatVector(inputMasks_);

    numImgs_ = inputImages_.size();
    if (numImgs_ < 2)
        return ERR_NEED_MORE_IMGS;

    const bool emptyMask = inputMasks_.empty();
    if ((!emptyMask) && (numImgs_ != inputMasks_.size()))
        return ERR_BAD_ARGUMENTS;

    inputSizes_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        inputSizes_[i] = inputImages_[i].size();
        if (!emptyMask)
            assert(inputImages_[i].size() == inputMasks_[i].size());
    }

    bfi_ = (numImgs_ - 1) / 2;
    INFO("Base frame index: " << bfi_);

    return OK;
}

MotionShot::Status MotionShot::run()
{
    // 1.计算图像变换(在低分辨率上), 得到统一了尺寸的 warp image 和 warp mask
    Status status = getWarpedImagsFromStitcher(0.25);
    if (status != OK) {
        WARNING("Return status not OK in MotionShot::getWarpedImagsFromStitcher()");
        return status;
    }

    // 2. 统一 warp 后图像的尺寸, 使其和 pano 一致
    status = makeBorderForWarpedImages();
    if (status != OK) {
        WARNING("Return status not OK in MotionShot::makeBorderForWarpedImages()");
        return status;
    }

    // 3.检测前景(在低分辨率上)
    status = detectorForeground();
    if (status != OK) {
        WARNING("Return status not OK in MotionShot::detectorForeground()");
        return status;
    }

    // 4.图像合成
    return compose();
}

MotionShot::Status MotionShot::getWarpedImagsFromStitcher(double scale)
{
    INFO(endl << "\t Estimate transform...");
    int status = stitcher_->estimateTransform(inputImages_, inputMasks_);
    if (status != 0) {
        WARNING("Return status not OK in ImageStitcher::estimateTransform()");
        return (Status)status;
    }

    warpedImages_.resize(numImgs_);
    warpedMasks_.resize(numImgs_);
    status = stitcher_->getWarpedImages(warpedImages_, warpedMasks_, corners_, scale);
    if (status != 0) {
        WARNING("Return status not OK in ImageStitcher::getWarpedImages()");
        return (Status)status;
    }

    return OK;
}

MotionShot::Status MotionShot::makeBorderForWarpedImages()
{
#define ENABLE_DEBUG_MAKING_BORDER  1
    INFO(endl << "\t Making border...");
    assert(!warpedImages_.empty() && !warpedMasks_.empty() && !corners_.empty());
    assert(warpedImages_.size() == numImgs_);
    assert(warpedMasks_.size() == numImgs_);

    vector<Size> warpedSizes(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        warpedSizes[i] = warpedImages_[i].size();
        assert(warpedImages_[i].size() == warpedMasks_[i].size());
    }

    prepare(corners_, warpedSizes);
    INFO("(Low resolution)Pano size: " << dstROI_.size());
    assert(!dstImg_.empty() && !dstMask_.empty());

    const Point dst_tl = dstROI_.tl();
    const Point dst_br = dstROI_.br();

    vector<UMat> warpedImgsWithDstSize(numImgs_);
    vector<UMat> warpedMasksWithDstSize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        const Point img_tl = corners_[i];
        const Point img_br = img_tl + Point(warpedSizes[i].width, warpedSizes[i].height);
        copyMakeBorder(warpedImages_[i], warpedImgsWithDstSize[i], img_tl.y - dst_tl.y,
                       dst_br.y - img_br.y, img_tl.x - dst_tl.x, dst_br.x - img_br.x, BORDER_ISOLATED);
        copyMakeBorder(warpedMasks_[i], warpedMasksWithDstSize[i], img_tl.y - dst_tl.y,
                       dst_br.y - img_br.y, img_tl.x - dst_tl.x, dst_br.x - img_br.x, BORDER_ISOLATED);
        assert(warpedImgsWithDstSize[i].size() == warpedMasksWithDstSize[i].size());

#if DEBUG && ENABLE_DEBUG_MAKING_BORDER
        NamedLargeWindow("warpedImgsWithDstSize");
        imshow("warpedImgsWithDstSize", warpedImgsWithDstSize[i]);
        NamedLargeWindow("warpedMasksWithDstSize");
        imshow("warpedMasksWithDstSize", warpedMasksWithDstSize[i]);
        waitKey(100);
        imwrite("/home/vance/output/img"+to_string(i)+".jpg", warpedImgsWithDstSize[i]);
        imwrite("/home/vance/output/mask"+to_string(i)+".jpg", warpedMasksWithDstSize[i]);
#endif
    }
    destroyAllWindows();

    swap(warpedImgsWithDstSize, warpedImages_);
    swap(warpedMasksWithDstSize, warpedMasks_);

    return OK;
}

MotionShot::Status MotionShot::detectorForeground()
{
#define ENBALE_DEBUG_DETECTION 1
    INFO(endl << "\t Detecing foreground...");

    Status status = OK;

    // 把所有输入图像做一个中值滤波, 获得一个没有前景的背景
    Mat medianPano = Mat::zeros(warpedImages_[0].size(), CV_8UC3);
    vector<Mat> vImgs_Y(numImgs_); // 每副图像的Y域分量
    for (size_t i = 0; i < numImgs_; ++i) {
        Mat imgYUV;
        cvtColor(warpedImages_[i], imgYUV, COLOR_BGR2YUV);
        vector<Mat> channels;
        split(imgYUV, channels);
        vImgs_Y[i] = channels[0];
    }
    // 中值滤波
    for (int y = 0; y < warpedImages_[0].rows; ++y) {
        Vec3b* imgRow = medianPano.ptr<Vec3b>(y);

        for(int x = 0; x < warpedImages_[0].cols; ++x) {
            vector<pair<uchar, uchar>> vLumarAndIndex;
            for (size_t imgIdx = 0; imgIdx < numImgs_; ++imgIdx)
                vLumarAndIndex.emplace_back(vImgs_Y[imgIdx].at<uchar>(y, x), imgIdx);

            sort(vLumarAndIndex.begin(), vLumarAndIndex.end()); // 根据亮度中值决定此像素的值由哪张图像提供
            uchar idx1, idx2;
            if (numImgs_ % 2 == 0) {
                idx1 = vLumarAndIndex[numImgs_/2-1].second;
                idx2 = vLumarAndIndex[numImgs_/2].second;
            } else {
                idx1 = idx2 = vLumarAndIndex[(numImgs_-1)/2].second;
            }

            if (idx1 == idx2)
                imgRow[x] = warpedImages_[idx1].getMat(ACCESS_READ).at<Vec3b>(y, x);
            else
                imgRow[x] = 0.5*(warpedImages_[idx1].getMat(ACCESS_READ).at<Vec3b>(y, x) +
                                 warpedImages_[idx2].getMat(ACCESS_READ).at<Vec3b>(y, x));
        }
    }
    imwrite("/home/vance/output/medianPano.jpg", medianPano);

    // 计算所有图像的共同重叠区域, 得到最大内接矩形
    Mat overlapedMask(warpedMasks_[0].size(), CV_8UC1);
    overlapedMask.setTo(255);
    for (size_t i = 0; i < numImgs_; ++i) {
        const Mat mask = warpedMasks_[i].getMat(ACCESS_READ);
        overlapedMask &= mask;
    }
    imwrite("/home/vance/output/overlapedMask.jpg", overlapedMask);
    const Rect maxRect = getLargestInscribedRectangle(overlapedMask);
    Mat tmp = overlapedMask.clone();
    cvtColor(tmp, tmp, COLOR_GRAY2BGR);
    rectangle(tmp, maxRect, Scalar(0,0,255), 2);
    imwrite("/home/vance/output/overlapedMask_withRect.jpg", tmp);

    // 前景检测
    Mat frame, rawForeMask, foreMask, foreMaskNN;
    bitwise_and(medianPano, 255, frame, overlapedMask);
    detector_->apply(medianPano(maxRect), rawForeMask, 0);

    foregroundMasks_.resize(numImgs_);
    foregroundRectsRough_.resize(numImgs_);
    foregroundRectsRefine_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        bitwise_and(warpedImages_[i], 255, frame, overlapedMask);

        // 得到粗糙的前景掩模
        detector_->apply(frame(maxRect), rawForeMask, 0);
        if (rawForeMask.empty()) {
            WARNING("Empty raw fore maks!");
            continue;
        }

        // 对前景掩模做形态学滤波, 尽量筛选出正确的前景区域
        Rect foreRect, foreRectRough;
        foreMaskFilter(rawForeMask, foreMask, foreRect);
        foreRect.x += maxRect.x;
        foreRect.y += maxRect.y;
        foreRectRough = ResizeRectangle(foreRect, frame.size(), -30, 30);
        foregroundRectsRough_[i] = foreRectRough;

#if DEBUG && ENBALE_DEBUG_DETECTION
        imwrite("/home/vance/output/#"+to_string(i)+"-rawDiff.jpg", rawForeMask);

        imshow("input frame to detector", frame);
        imshow("raw mask from detector", rawForeMask);
        imshow("filtered mask", foreMask);
        imwrite("/home/vance/output/#"+to_string(i)+"-filteredDiff.jpg", foreMask);

        Mat ff = frame.clone();
        rectangle(ff, foreRect, Scalar(0,255,0), 2);
        rectangle(ff, foreRectRough, Scalar(0,0,255), 2);
        imshow("foreground rect filtered", ff);
        imwrite("/home/vance/output/#"+to_string(i)+"-FilteredRectforeMask.jpg", ff);
        waitKey(0);
#endif
        // 用NN检测人物前景, 得到最终的前景掩模
        status = detectorForegroundWithNN(frame(foreRectRough)/*warpedImages_[i]*/, foreMaskNN);
        if (status == OK)
            copyMakeBorder(foreMaskNN, foregroundMasks_[i], foreRectRough.y,
                           frame.rows - foreRectRough.br().y, foreRectRough.x,
                           frame.cols - foreRectRough.br().x, BORDER_ISOLATED);
        else
            break;

//#if DEBUG && ENBALE_DEBUG_DETECTION
//        NamedLargeWindow("foreground mask NN");
//        imshow("foreground mask NN", foreMaskNN);
//        imwrite("/home/vance/output/#"+to_string(i)+"-maskNN.jpg", foreMaskNN);
//        waitKey(0);
//#endif
    }
    destroyAllWindows();

    return status;
}

MotionShot::Status MotionShot::compose()
{
    return OK; //! TODO

#define ENABLE_EBUG_COMPOSE 1
    INFO(endl << "\t Composing...");

    foregroundRectsRough_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        UMat foreRectArea;
        dilate(foregroundMasks_[i], foreRectArea, UMat(), Point(-1, -1), 5);
        foregroundRectsRough_[i] = boundingRect(foreRectArea);
#if DEBUG && ENABLE_EBUG_COMPOSE
        UMat rect = warpedImages_[i].clone();
        rectangle(rect, foregroundRectsRough_[i], Scalar(0,0,255), 3);
        NamedLargeWindow("rect");
        imshow("rect", rect);
        waitKey(0);
        destroyWindow("rect");
#endif
    }

    int numOverlapped = checkOverlappedArea(foregroundRectsRough_);
    if (numOverlapped > 0)
        return composeWithOverlapped();
    else
        return composeWithoutOverlapped();
}


MotionShot::Status MotionShot::detectorForegroundWithNN(InputArray _src, OutputArray _dst)
{
    assert(_src.channels() == 3);

    // 调用SDK获得分割结果
    Json::Value result = bodyAnalyst_->body_seg_cv(_src, bodySegOptions_);

    // 解析Json结果
    string scoremap = result["scoremap"].asString();  // 灰度图像
    string decode_result = aip::base64_decode(scoremap);
    vector<char> base64_img(decode_result.begin(), decode_result.end());
    Mat mask = imdecode(base64_img, IMREAD_GRAYSCALE);

    _dst.create(mask.size(), mask.type());
    mask.copyTo(_dst);

    if (countNonZero(mask))
        return OK;
    else
        return ERR_FORE_DETECT_FAIL;
}


/// private functions
///
void MotionShot::prepare(const vector<Point>& corners, const vector<Size>& sizes)
{
    prepare(ResultRoi(corners, sizes));
}

void MotionShot::prepare(Rect dst_roi)
{
    dstImg_.create(dst_roi.size(), CV_16SC3);
    dstImg_.setTo(Scalar::all(0));
    dstMask_.create(dst_roi.size(), CV_8U);
    dstMask_.setTo(Scalar::all(0));
    dstROI_ = dst_roi;
}




//! TODO
MotionShot::Status MotionShot::composeWithoutOverlapped()
{


    return OK;
}

MotionShot::Status MotionShot::composeWithOverlapped()
{
    return composeWithoutOverlapped(); //! TODO
}

void MotionShot::foreMaskFilter(InputArray src, OutputArray dst, Rect& foreRect)
{
    const Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    morphologyEx(src, dst, MORPH_OPEN, kernel);
    morphologyEx(dst, dst, MORPH_CLOSE, kernel);

    const Size imgSize = src.size();
    const int th_w = imgSize.width * 0.05;
    const int th_h = imgSize.height * 0.05;
    const int th_area = imgSize.area() * 0.01;

    Mat recShow = Mat::zeros(imgSize, CV_8UC3);

    vector<Rect> potentialForeRects;
    vector<vector<Point>> contours;
    findContours(dst, contours, RETR_LIST, CHAIN_APPROX_SIMPLE); //? 凸包convexHull()

    for (size_t i = 0, iend = contours.size(); i < iend; ++i) {
        Rect rec = boundingRect(contours[i]);

        // rec太靠近边界(5%)的大概率不是前景, 或者太高/宽也不是前景
        if (rec.x > imgSize.width - th_w || rec.y > imgSize.height - th_h)
            continue;
        if (rec.br().x < th_w || rec.br().y < th_h)
            continue;
        if ((rec.br().x - rec.x > imgSize.width*0.8) || (rec.br().y - rec.y > imgSize.height*0.8))
            continue;

        // rec 面积小于1%的大概率不是前景
        if (rec.area() < th_area)
            continue;

        potentialForeRects.push_back(rec);
        drawContours(recShow, contours, i, Scalar(255,0,0), 2);
    }

    // 得到了多个可能前景区域
    for (size_t j = 0, jend = potentialForeRects.size(); j < jend; ++j) {
        rectangle(recShow, potentialForeRects[j], Scalar(0,255,0), 2);
    }
    imshow("potentialForeRects", recShow);
    waitKey(1000);

    if (potentialForeRects.empty())
        foreRect = Rect(Point(), imgSize);
    else
        foreRect = potentialForeRects[0];
}

//! TODO
int MotionShot::checkOverlappedArea(const vector<Rect>& rects)
{
    return 0;
}

Rect MotionShot::getLargestInscribedRectangle(InputArray _mask)
{
    Mat mask = _mask.getMat();
    assert(mask.type() == CV_8UC1);

    Rect minOutRect = boundingRect(mask);

    Mat show = mask.clone();
    cvtColor(show, show, COLOR_GRAY2BGR);
    rectangle(show, minOutRect, Scalar(0,255,0), 2);

    Rect maxInRect = minOutRect;
    do { //! TODO  待改进
        maxInRect.x += 2; maxInRect.y += 2;
        maxInRect.width -= 4; maxInRect.height -= 4;

        Mat a = show.clone();
        rectangle(a, maxInRect, Scalar(0,0,255), 1);
        imshow("tmp rect", a);
        waitKey(300);
    } while (countNonZero(mask(maxInRect)) != maxInRect.area());

    return maxInRect;
}


}  // namespace ms
