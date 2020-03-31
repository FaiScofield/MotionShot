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
    stitcher_ = ImageStitcher::create(ImageStitcher::AKAZE, ImageStitcher::BF);
    blender_ = cvBlender::createDefault(cvBlender::FEATHER);
    detector_ = cv::createBackgroundSubtractorMOG2();

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
    _imgs.getMatVector(inputImages_);
    _masks.getMatVector(inputMasks_);

    numImgs_ = inputImages_.size();
    if (numImgs_ < 2)
        return ERR_NEED_MORE_IMGS;

    const bool emptyMask = inputMasks_.empty();
    if ((!emptyMask) && (numImgs_ != inputMasks_.size()))
        return ERR_BAD_ARGUMENTS;

    for (size_t i = 0; i < numImgs_; ++i) {
        inputSizes_[i] = inputImages_[i].size();
        if (!emptyMask)
            assert(inputImages_[i].size() == inputMasks_[i].size());
    }

    bfi_ = (numImgs_ - 1) / 2;
    //    baseFrame_ = inputFrames_[bfi_];
    //    baseFrameMask_ = inputMasks_[bfi_];

    return OK;
}

MotionShot::Status MotionShot::run()
{
    // 1.计算图像变换
    Status status = estimateTransform();
    if (status != OK)
        return status;

    // 2.得到统一了尺寸的 warp image 和 warp mask
    status = warpImages();
    if (status != OK)
        return status;

    // 3.检测前景
    status = detectorForeground();
    if (status != OK)
        return status;

    // 4.图像合成
    return compose();
}

MotionShot::Status MotionShot::estimateTransform()
{
    stitcher_->estimateTransform(inputImages_, inputMasks_);
    stitcher_->getWarpedImages(warpedImages_, warpedMasks_, corners_);  //! TODO

    return OK;
}

MotionShot::Status MotionShot::warpImages()
{
    warpedSize_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        warpedSize_[i] = warpedImages_[i].size();
        assert(warpedImages_[i].size() == warpedMasks_[i].size());
    }

    prepare(corners_, warpedSize_);

    makeBorderForWarpedImages();

    return OK;
}

MotionShot::Status MotionShot::detectorForeground()
{
#define ENBALE_DEBUG_DETECTION 1

    const Mat& baseFrameWarped = warpedImages_[bfi_];
    const Mat& baseMaskWarped = warpedMasks_[bfi_];

    // 将基准帧设为背景, 检测其他帧的前景
    Mat foreMask;
    detector_->apply(baseFrameWarped, foreMask, 0);

    foregroundMasks_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
    // if (i != bfi_) {
        Mat frameWarped, rawMask;
        bitwise_and(warpedImages_, 255, frameWarped, baseMaskWarped);

        // 得到粗糙的前景掩模
        detector_->apply(frameWarped, rawMask, 0);

        // 对前景掩模做形态学滤波
        foreMaskFilter(rawMask, foreMask);

#if DEBUG && ENBALE_DEBUG_DETECTION
        NamedLargeWindow("foreground mask raw");
        NamedLargeWindow("foreground mask filtered");
        imshow("foreground mask raw", rawMask);
        imshow("foreground mask filtered", foreMask);
#endif
        // 用NN检测人物前景, 得到最终的前景掩模
        detectorForegroundWithNN(frameWarped, foreMask, foreMask);
        foregroundMasks_[i] = foreMask.clone();

#if DEBUG && ENBALE_DEBUG_DETECTION
        NamedLargeWindow("foreground mask NN");
        imshow("foreground mask NN", foreMask);
        waitKey(0);
#endif
    // }
    }
    foreMask.release();

    return OK;
}

MotionShot::Status MotionShot::compose()
{
    foregroundRects_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        Mat foreRectArea;
        dilate(foregroundMasks_[i], foreRectArea, Mat(), Point(-1, -1), 10);
        foregroundRects_[i] = boundingRect(foreRectArea);
    }

    int numOverlapped = checkOverlappedArea(foregroundRects_);
    if (numOverlapped > 0)
        return composeWithOverlapped();
    else
        return composeWithoutOverlapped();
}


void MotionShot::detectorForegroundWithNN(InputArray _src, InputArray _mask, OutputArray _dst)
{
    Mat src, mask, image;
    src = _src.getMat();
    mask = _mask.getMat();
    bitwise_and(src, 255, image, mask);

    // 调用SDK获得分割结果
    Json::Value result = bodyAnalyst_->body_seg_cv(image, bodySegOptions_);

    // 解析Json结果
    string scoremap = result["scoremap"].asString();  // 灰度图像
    string decode_result = aip::base64_decode(scoremap);
    vector<char> base64_img(decode_result.begin(), decode_result.end());
    Mat dst = imdecode(base64_img, IMREAD_COLOR);
    _dst.assign(dst);
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

void MotionShot::makeBorderForWarpedImages()
{
    const Point dst_tl = dstROI_.tl();
    const Point dst_br = dstROI_.br();

    vector<Mat> warpImgsWithDstSize(numImgs_);
    vector<Mat> warpMasksWithDstSize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        const Point img_tl = corners_[i];
        const Point img_br = img_tl + Point(warpedSize_[i].width, warpedSize_[i].height);
        copyMakeBorder(warpedImages_[i], warpImgsWithDstSize[i], img_tl.y - dst_tl.y,
                       dst_br.y - img_br.y, img_tl.x - dst_tl.x, dst_br.x - img_br.x, BORDER_ISOLATED);
        copyMakeBorder(warpedMasks_[i], warpMasksWithDstSize[i], img_tl.y - dst_tl.y,
                       dst_br.y - img_br.y, img_tl.x - dst_tl.x, dst_br.x - img_br.x, BORDER_ISOLATED);
        assert(warpImgsWithDstSize[i].size() == warpMasksWithDstSize[i].size());
    }

    swap(warpImgsWithDstSize, warpedImages_);
    swap(warpMasksWithDstSize, warpedMasks_);
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

//! TODO
void MotionShot::foreMaskFilter(InputArray src, OutputArray dst)
{

}

//! TODO
int MotionShot::checkOverlappedArea(const vector<Rect>& rects)
{
    return 0;
}


}  // namespace ms
