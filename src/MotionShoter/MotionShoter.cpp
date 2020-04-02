#include "MotionShoter.h"
#include "utility.h"

#include <iostream>
#include <map>
#include <opencv2/video/video.hpp>

#include <curl/curl.h>
#include <jsoncpp/json/json.h>

namespace ms
{

using namespace std;
using namespace cv;

MotionShot::MotionShot()
{
    detectScale_ = 1.0;

    stitcher_ = ImageStitcher::create(ImageStitcher::ORB, ImageStitcher::BF);
    stitcher_->setScales(0.25, 0.1, 1.);
    blender_ = cvBlender::createDefault(cvBlender::FEATHER);
    detector_ = cv::createBackgroundSubtractorMOG2(20, 12, false);
    //    detector_ = cv::bgsegm::createBackgroundSubtractorGMG();

    // for NN-based body segmentation
    useBaiduAIP_ = false;
    const string app_id = "18679385";
    const string aip_key = "12yptwaZPOxoGBfPR0PGYT43";
    const string secret_key = "sx8w8l1dzlD2Gt0QAKgZxItRB3uE8DZz";
    bodyAnalyst_ = std::unique_ptr<aip::Bodyanalysis>(new aip::Bodyanalysis(app_id, aip_key, secret_key));
    // bodyAnalyst_ = std::make_unique<aip::Bodyanalysis>(app_id, aip_key, secret_key);  // c++ 14
    bodySegOptions_["type"] = "scoremap";
}

MotionShot::~MotionShot()
{
    bodyAnalyst_.release();
    stitcher_.release();
    blender_.release();
    detector_.release();
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
    Status status = getWarpedImagsFromStitcher(detectScale_);
    if (status != OK) {
        WARNING("Return status not OK in MotionShot::getWarpedImagsFromStitcher()");
        return status;
    }

    // 2.检测前景(在低分辨率上)
    status = detectorForeground();
    if (status != OK) {
        WARNING("Return status not OK in MotionShot::detectorForeground()");
        return status;
    }

    // 3.图像合成
    return compose(pano_);
}

MotionShot::Status MotionShot::getWarpedImagsFromStitcher(double scale)
{
    MS_DEBUG_TO_DELETE assert(detectScale_ == scale);

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

    makeBorderForWarpedImages();

    return OK;
}

MotionShot::Status MotionShot::detectorForeground()
{
#define ENBALE_DEBUG_DETECTION 0
    INFO(endl << "\t Detecing foreground...");

    Status status = OK;

    // 把所有输入图像做一个中值滤波, 获得一个没有前景的背景
    Mat medianPano(dstROI_.size(), CV_8UC3);
    ImagesMedianFilterToOne(warpedImages_, medianPano);
    imwrite("/home/vance/output/medianPano.jpg", medianPano);

    // 计算所有图像的共同重叠区域, 得到最大内接矩形
    Mat overlapedMask(dstROI_.size(), CV_8UC1);
    overlapedMask.setTo(255);
    for (size_t i = 0; i < numImgs_; ++i) {
        const Mat mask = warpedMasks_[i].getMat(ACCESS_READ);
        overlapedMask &= mask;
    }
    const Rect maxRect = getLargestInscribedRectangle(overlapedMask);
#if DEBUG && ENBALE_DEBUG_DETECTION
    Mat tmp = overlapedMask.clone();
    cvtColor(tmp, tmp, COLOR_GRAY2BGR);
    rectangle(tmp, maxRect, Scalar(0, 0, 255), 2);
    imwrite("/home/vance/output/overlapedMask_withRect.jpg", tmp);
#endif

    // 前景检测
    Mat rawForeMask, foreMask;
    detector_->apply(medianPano(maxRect), rawForeMask, 0);

    foregroundMasksRough_.resize(numImgs_);
    foregroundMasksRefine_.resize(numImgs_);
    foregroundRectsRough_.resize(numImgs_);
    foregroundRectsRefine_.resize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        // 得到粗糙的前景掩模
        detector_->apply(warpedImages_[i](maxRect), rawForeMask, 0);
        if (rawForeMask.empty()) {
            WARNING("Empty raw fore maks!");
            continue;
        }

        // 对前景掩模做形态学滤波, 尽量筛选出正确的前景区域
        Rect foreRect, foreRectRough;
        foreMaskFilter(rawForeMask, foreMask, foreRect);
        copyMakeBorder(foreMask, foregroundMasksRough_[i], maxRect.y, dstROI_.height - maxRect.br().y,
                       maxRect.x, dstROI_.width - maxRect.br().x, BORDER_ISOLATED); // 相对pano的mask
        foreRect.x += maxRect.x;  // 相对pano的rect
        foreRect.y += maxRect.y;
        foregroundRectsRefine_[i] = foreRect;
        foreRectRough = ResizeRectangle(foreRect, dstROI_.size(), -20, 20);
        foregroundRectsRough_[i] = foreRectRough;

        // 掩模中会残留一些背景, 去掉粗糙前景矩形之外的背景
        Mat mask = Mat::zeros(foregroundMasksRough_[i].size(), CV_8UC1);
        mask(foregroundRectsRough_[i]).setTo(255);
        bitwise_and(foregroundMasksRough_[i], mask, foregroundMasksRough_[i]);

#if DEBUG && ENBALE_DEBUG_DETECTION
        imwrite("/home/vance/output/#" + to_string(i) + "-rawDiff.jpg", rawForeMask);

        imshow("input frame to detector", warpedImages_[i]);
        imshow("raw mask from detector", rawForeMask);
        imshow("filtered mask", foreMask);
        imwrite("/home/vance/output/#" + to_string(i) + "-filteredDiff.jpg", foreMask);

        Mat ff = warpedImages_[i].clone();
        rectangle(ff, foreRect, Scalar(0, 255, 0), 2);
        rectangle(ff, foreRectRough, Scalar(0, 0, 255), 2);
        imshow("foreground rect filtered", ff);
        imwrite("/home/vance/output/#" + to_string(i) + "-FilteredRectforeMask.jpg", ff);
        waitKey(0);
        destroyAllWindows();
#endif
    }
    rawForeMask.release();

    if (useBaiduAIP_) {
        Mat foreMaskNN;
        for (size_t i = 0; i < numImgs_; ++i) {
            // 用NN检测人物前景, 得到最终的前景掩模
            const Rect& rec = foregroundRectsRough_[i];
            status = detectorForegroundWithNN(warpedImages_[i](rec), foreMaskNN);
            if (status == OK) {
                copyMakeBorder(foreMaskNN, foregroundMasksRefine_[i], rec.y, dstROI_.height - rec.br().y,
                               rec.x, dstROI_.width - rec.br().x, BORDER_ISOLATED);
                foregroundRectsRefine_[i] = boundingRect(foregroundMasksRefine_[i]);
            } else {
                WARNING("Cann't detect the foreground body using Baidu AIP for #" << i);
                continue;
            }
#if DEBUG && ENBALE_DEBUG_DETECTION
            imshow("foreground mask NN", foregroundMasksRefine_[i]);
            imwrite("/home/vance/output/#" + to_string(i) + "-maskNN.jpg", foregroundMasksRefine_[i]);
            waitKey(0);
#endif
        }
    } else {
        foregroundMasksRefine_ = foregroundMasksRough_;
    }

    MS_DEBUG_TO_DELETE destroyAllWindows();
    return status;
}

MotionShot::Status MotionShot::compose(OutputArray pano)
{
    INFO(endl << "\t Composing...");

    int numOverlapped = countOverlappedRects(foregroundRectsRough_, overlappedForegroundRects_);
    INFO("Number of overlapped foregrounds: " << numOverlapped);

    if (numOverlapped > 0)
        return composeWithOverlapped(pano);
    else
        return composeWithoutOverlapped(pano);
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

    result.clear();

    if (countNonZero(mask) > 0)
        return OK;
    else
        return ERR_FORE_DETECT_FAIL;
}


/// private functions
///
void MotionShot::makeBorderForWarpedImages()
{
#define ENABLE_DEBUG_MAKING_BORDER 0
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

    imgBorders_.resize(numImgs_);
    vector<UMat> warpedImgsWithDstSize(numImgs_);
    vector<UMat> warpedMasksWithDstSize(numImgs_);
    for (size_t i = 0; i < numImgs_; ++i) {
        const Point img_tl = corners_[i];
        const Point img_br = img_tl + Point(warpedSizes[i].width, warpedSizes[i].height);
        vector<int> borders{img_tl.y - dst_tl.y, dst_br.y - img_br.y, img_tl.x - dst_tl.x,
                            dst_br.x - img_br.x};
        copyMakeBorder(warpedImages_[i], warpedImgsWithDstSize[i], borders[0], borders[1],
                       borders[2], borders[3], BORDER_ISOLATED);
        copyMakeBorder(warpedMasks_[i], warpedMasksWithDstSize[i], borders[0], borders[1],
                       borders[2], borders[3], BORDER_ISOLATED);
        imgBorders_[i] = borders;
        assert(warpedImgsWithDstSize[i].size() == warpedMasksWithDstSize[i].size());

#if DEBUG && ENABLE_DEBUG_MAKING_BORDER
        imshow("warpedImgsWithDstSize", warpedImgsWithDstSize[i]);
        imshow("warpedMasksWithDstSize", warpedMasksWithDstSize[i]);
        waitKey(100);
        imwrite("/home/vance/output/img" + to_string(i) + ".jpg", warpedImgsWithDstSize[i]);
        imwrite("/home/vance/output/mask" + to_string(i) + ".jpg", warpedMasksWithDstSize[i]);
#endif
    }

    swap(warpedImgsWithDstSize, warpedImages_);
    swap(warpedMasksWithDstSize, warpedMasks_);
}

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

MotionShot::Status MotionShot::composeWithoutOverlapped(OutputArray pano)
{
#define ENABLE_DEBUG_COMPOSE    1
    INFO("Composing with independent rects...");

    assert(imgBorders_.size() == numImgs_);
    assert(foregroundRectsRough_.size() == numImgs_);

    // masks和rects去掉之前添加的border, 返回给stitcher
    vector<Mat> masksRough(numImgs_);
    vector<Rect> rectsRough = foregroundRectsRough_;

    const double invScale = 1. / detectScale_;
    const Size size = dstROI_.size();

    for (size_t i = 0; i < numImgs_; ++i) {
        const vector<int>& borders = imgBorders_[i];
        assert(borders.size() == 4);
        MS_DEBUG_TO_DELETE INFO("#" << i << " borders: " << borders[0] << ", " << borders[1] << ", " << borders[2] << ", " << borders[3]);

        masksRough[i] = foregroundMasksRough_[i].rowRange(borders[0], size.height - borders[1])
                            .colRange(borders[2], size.width - borders[3]).clone();
        rectsRough[i].x -= borders[2];
        rectsRough[i].y -= borders[0];

        ATTENTION("#" << i << " size from sitcher is: " << masksRough[i].size());

        MS_DEBUG_TO_DELETE UMat img = warpedImages_[i].rowRange(borders[0], size.height - borders[1])
                .colRange(borders[2], size.width - borders[3]).clone();
        if (std::abs(detectScale_ - 1.) > 1e-2) {
            resize(masksRough[i], masksRough[i], Size(), invScale, invScale, INTER_NEAREST);
            MS_DEBUG_TO_DELETE resize(img, img, Size(), invScale, invScale, INTER_LINEAR_EXACT);

            ATTENTION("#" << i << " invScaled size to stitcher is: " << img.size());

            rectsRough[i].x *= invScale;
            rectsRough[i].y *= invScale;
            rectsRough[i].width *= invScale;
            rectsRough[i].height *= invScale;
        }

#if DEBUG && ENABLE_DEBUG_COMPOSE
        Rect maskRect = boundingRect(masksRough[i]);
        rectangle(img, maskRect, Scalar(255,0,0), 2);
        rectangle(img, rectsRough[i], Scalar(0,255,0), 2);
        NamedLargeWindow("resize to orginal scale");
        imshow("resize to orginal scale", img);
        imwrite("/home/vance/output/#" + to_string(i) + "恢复至原分辨率的前景检测结果.jpg", img);
        waitKey(200);
#endif
    }

    // 传回原尺度上的前景掩模和矩形区域
    stitcher_->setForegrounds(masksRough, rectsRough);

    return (Status)stitcher_->composePanoWithoutOverlapped(pano);
}

MotionShot::Status MotionShot::composeWithOverlapped(OutputArray pano)
{
#define ENABLE_DEBUG_COMPOSE_OVERLAPPED 1
    INFO("Composing with overlapped rects...");

    assert(imgBorders_.size() == numImgs_);
    assert(foregroundRectsRough_.size() == numImgs_);
    assert(overlappedForegroundRects_.size() == numImgs_);

    // masks和rects去掉之前添加的border, 返回给stitcher
    vector<Mat> masksRough(numImgs_);
    vector<Rect> rectsRough = foregroundRectsRough_;    //! TODO 先用粗糙的做
    vector<Rect> rectsOverlapped = overlappedForegroundRects_;

    const double invScale = 1. / detectScale_;
    const Size size = dstROI_.size();

    for (size_t i = 0; i < numImgs_; ++i) {
        const vector<int>& borders = imgBorders_[i];
        assert(borders.size() == 4);
        MS_DEBUG_TO_DELETE INFO("#" << i << " borders: " << borders[0] << ", " << borders[1] << ", " << borders[2] << ", " << borders[3]);

        masksRough[i] = foregroundMasksRough_[i].rowRange(borders[0], size.height - borders[1])
                .colRange(borders[2], size.width - borders[3]).clone();

        rectsRough[i].x -= borders[2];
        rectsRough[i].y -= borders[0];
        rectsOverlapped[i].x -= borders[2];
        rectsOverlapped[i].y -= borders[0];

        ATTENTION("#" << i << " size from sitcher is: " << masksRough[i].size());
        MS_DEBUG_TO_DELETE UMat img = warpedImages_[i].rowRange(borders[0], size.height - borders[1])
                .colRange(borders[2], size.width - borders[3]).clone();
        if (std::abs(detectScale_ - 1.) > 1e-2) {
            resize(masksRough[i], masksRough[i], Size(), invScale, invScale, INTER_NEAREST);
            MS_DEBUG_TO_DELETE resize(img, img, Size(), invScale, invScale, INTER_LINEAR_EXACT);
            ATTENTION("#" << i << " invScaled size to stitcher is: " << img.size());

            rectsRough[i].x *= invScale;
            rectsRough[i].y *= invScale;
            rectsRough[i].width *= invScale;
            rectsRough[i].height *= invScale;

            rectsOverlapped[i].x *= invScale;
            rectsOverlapped[i].y *= invScale;
            rectsOverlapped[i].width *= invScale;
            rectsOverlapped[i].height *= invScale;
        }

#if DEBUG && ENABLE_DEBUG_COMPOSE_OVERLAPPED
        Rect maskRect = boundingRect(masksRough[i]);
        rectangle(img, maskRect, Scalar(255,0,0), 2);
        rectangle(img, rectsRough[i], Scalar(0,255,0), 2);
        rectangle(img, rectsOverlapped[i], Scalar(0,0,255), 2);
        NamedLargeWindow("resize to orginal scale");
        imshow("resize to orginal scale", img);
        imwrite("/home/vance/output/#" + to_string(i) + "恢复至原分辨率的前景检测结果.jpg", img);
        waitKey(200);
#endif
    }

    // 传回原尺度上的前景掩模和矩形区域
    stitcher_->setForegrounds(masksRough, rectsRough);
    stitcher_->setOverlappedForegroundRects(rectsOverlapped);

    return (Status)stitcher_->composePanoWithOverlapped(pano);
}

void MotionShot::foreMaskFilter(InputArray src, OutputArray dst, Rect& foreRect)
{
#define ENABLE_DEBUG_FILTER 0
    const Mat kernel1 = getStructuringElement(MORPH_RECT, Size(7, 7));
    const Mat kernel2 = getStructuringElement(MORPH_RECT, Size(11, 11));

#if DEBUG && ENABLE_DEBUG_FILTER
    Mat tmp1, tmp2;
    morphologyEx(src, tmp1, MORPH_OPEN, kernel1);
    imshow("1 raw mask", src);
    imshow("2 open mask", tmp1);
    morphologyEx(tmp1, tmp2, MORPH_CLOSE, kernel2);
    imshow("3 close mask", tmp2);
    morphologyEx(tmp2, dst, MORPH_OPEN, kernel2);
    imshow("4 open mask", dst);
    waitKey(0);

    MS_DEBUG_TO_DELETE Mat recShow = src.getMat().clone();
    MS_DEBUG_TO_DELETE cvtColor(recShow, recShow, COLOR_GRAY2BGR);
#else
    morphologyEx(src, dst, MORPH_OPEN, kernel1);
    morphologyEx(dst, dst, MORPH_CLOSE, kernel2);
    morphologyEx(dst, dst, MORPH_OPEN, kernel2);
#endif

    const Size imgSize = src.size();
    const int th_w = imgSize.width * 0.05;
    const int th_h = imgSize.height * 0.05;
    const int th_area = imgSize.area() * 0.01;

    vector<Rect> potentialForeRects;
    vector<vector<Point>> contours;
    findContours(dst, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);  //? 凸包convexHull()

    for (size_t i = 0, iend = contours.size(); i < iend; ++i) {
        Rect rec = boundingRect(contours[i]);

        // rec太靠近边界(5%)的大概率不是前景, 或者太高/宽也不是前景
        if (rec.x > imgSize.width - th_w || rec.y > imgSize.height - th_h)
            continue;
        if (rec.br().x < th_w || rec.br().y < th_h)
            continue;
        if ((rec.br().x - rec.x > imgSize.width * 0.8) ||
            (rec.br().y - rec.y > imgSize.height * 0.8))
            continue;

        //! TODO rec 面积小于1%的大概率不是前景
        if (rec.area() < th_area)
            continue;

        potentialForeRects.push_back(rec);
#if DEBUG && ENABLE_DEBUG_FILTER
        drawContours(recShow, contours, i, Scalar(255, 0, 0), 2);
#endif
    }

#if DEBUG && ENABLE_DEBUG_FILTER
    // 得到了多个可能前景区域
    for (size_t j = 0, jend = potentialForeRects.size(); j < jend; ++j)
        rectangle(recShow, potentialForeRects[j], Scalar(0, 255, 0), 2);
    imshow("potentialForeRects", recShow);
    waitKey(0);
#endif

    if (potentialForeRects.empty())
        foreRect = Rect(Point(), imgSize);
    else  //! TODO
        foreRect = potentialForeRects[0];
}

int MotionShot::countOverlappedRects(const vector<Rect>& rects, vector<Rect>& overlapRects)
{
    assert(rects.size() == numImgs_);

    int numOverlap = 0;
    vector<Rect> overlaps(numImgs_, Rect());

    MS_DEBUG_TO_DELETE UMat toShow = warpedImages_[bfi_].clone();
    MS_DEBUG_TO_DELETE for (size_t i = 0; i < numImgs_; ++i)
        MS_DEBUG_TO_DELETE rectangle(toShow, rects[i], Scalar(0, 255, 0), 3);

    for (size_t i = 0; i < numImgs_ - 1; ++i) {
        Rect overlap;
        if (isOverlapped(rects[i], rects[i + 1], overlap)) {
            MS_DEBUG_TO_DELETE rectangle(toShow, overlap, Scalar(0, 0, 255), 2);
            overlaps[i] = overlap;
            numOverlap++;
        }
    }

    MS_DEBUG_TO_DELETE imwrite("/home/vance/output/fore_rects.jpg", toShow);
    MS_DEBUG_TO_DELETE INFO("Write fore rects on pano to file /home/vance/output/fore_rects.jpg");

    overlapRects = overlaps;

    return numOverlap;
}

bool MotionShot::isOverlapped(Rect rec1, Rect rec2, Rect& overlap)
{
    if (rec1.x >= rec2.x + rec2.width || rec1.y >= rec2.y + rec2.height)
        return false;
    if (rec1.x + rec1.width <= rec2.x || rec1.y + rec1.height <= rec2.y)
        return false;

    const int ox1 = max(rec1.x, rec2.x);
    const int oy1 = max(rec1.y, rec2.y);
    const int ox2 = min(rec1.x + rec1.width, rec2.x + rec2.width);
    const int oy2 = min(rec1.y + rec1.height, rec2.y + rec2.height);

    overlap = Rect(Point(ox1, oy1), Point(ox2, oy2));

    return true;
}

Rect MotionShot::getLargestInscribedRectangle(InputArray _mask)
{
#define ENABLE_DEBUG_RECTANGLE  0
    Mat mask = _mask.getMat();
    assert(mask.type() == CV_8UC1);

    Rect minOutRect = boundingRect(mask);

    Mat show = mask.clone();
    cvtColor(show, show, COLOR_GRAY2BGR);
    rectangle(show, minOutRect, Scalar(0, 255, 0), 2);

    Rect maxInRect = minOutRect;
    do {  //! TODO  待改进
        maxInRect.x += 2;
        maxInRect.y += 2;
        maxInRect.width -= 4;
        maxInRect.height -= 4;
#if DEBUG && ENABLE_DEBUG_RECTANGLE
        Mat a = show.clone();
        rectangle(a, maxInRect, Scalar(0, 0, 255), 1);
        imshow("tmp rect", a);
        waitKey(300);
#endif
    } while (countNonZero(mask(maxInRect)) != maxInRect.area());

    return maxInRect;
}


}  // namespace ms
