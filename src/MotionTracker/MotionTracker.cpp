#include "MotionTracker.h"
#include "BS_MOG2_CV.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

using namespace std;
using namespace cv;

MotionTracker::MotionTracker() : _curId(0), _defaultConstruct(true)
{
    _minBlobSize = Size(3, 3);
    auto cvBS = createBackgroundSubtractorMOG2(500, 200, true);
    BS_MOG2_CV be(cvBS.get());
    _substractor = dynamic_cast<BaseMotionDetector*>(&be);
}

MotionTracker::MotionTracker(BaseMotionDetector* bs)
    : _curId(0), _defaultConstruct(false), _substractor(bs)
{
    _minBlobSize = Size(3, 3);
}

MotionTracker::~MotionTracker()
{
    if (_defaultConstruct)
        delete _substractor;
}

void MotionTracker::setPano(const Mat &pano)
{
    _pano = pano.clone();
    cvtColor(pano, _panoGray, COLOR_BGR2GRAY);
}

void MotionTracker::substractBackground(const Mat& input, Mat& mask)
{
    _currentFrame = input.clone();
    if (input.type() == CV_8UC3)
        cvtColor(_currentFrame, _currentFrameGray, COLOR_BGR2GRAY);
    else
        _currentFrameGray = _currentFrame;  // 浅拷贝即可

    _substractor->apply(input, mask);
    assert(!mask.empty());

    _currentForeground = mask.clone();
    _curId++;


    trackOpticalFlow();
    _lastFrame = _currentFrame.clone();
    _lastFrameGray = _currentFrameGray.clone();
    _lastForeground = _currentForeground.clone();
}

void MotionTracker::detectBlocks()
{
    vector<vector<Point>> contours;
    findContours(_currentForeground, contours, RETR_EXTERNAL,
                 CHAIN_APPROX_TC89_KCOS);  // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE

    _currentContoursImage = _currentFrame.clone();
    drawContours(_currentContoursImage, contours, -1, Scalar(0, 255, 0), -1);

    blobFilter(contours);

    displayBlobs("blobsFilteredImage");
}

void MotionTracker::blobFilter(const vector<vector<Point>>& contours)
{
    /// All finding blocks
    int maxArea = 0;
    for (int i = 0, lim = contours.size(); i < lim; ++i) {
        _blobs.push_back(boundingRect(contours[i]));
        if (_blobs.back().area() > maxArea)
            maxArea = _blobs.back().area();
    }

    /// Erase bad blocks
    for (auto it = _blobs.begin(), iend = _blobs.end(); it != iend; it++) {
        if (it->width < _minBlobSize.width || it->height < _minBlobSize.height) {
            /*it = */ _blobs.erase(it);
            continue;
        }
        if (it->area() < maxArea * 0.1) {
            /*it = */ _blobs.erase(it);
            continue;
        }
        //        it++;
    }
}

void MotionTracker::updateObjects(float minDist)
{
    for (auto objI = _objects.begin(), oend = _objects.end(); objI != oend; ++objI) {
        MovingObject& object = *objI;
        object._updated = false;

        float current_min_dist = minDist;

        vector<Rect>::iterator nearestBlob = _blobs.end();
        vector<Rect>::iterator blobI = _blobs.begin();
        vector<Rect>::iterator blobI_end = _blobs.end();
        for (; blobI != blobI_end; ++blobI) {
            float dist = norm(Point2f(object._rect.x, object._rect.y) - Point2f(blobI->x, blobI->y));
            if (dist < current_min_dist) {
                current_min_dist = dist;
                nearestBlob = blobI;
            }
        }

        if (nearestBlob != _blobs.end()) {
            object._speed = Point(nearestBlob->x - object._rect.x, nearestBlob->y - object._rect.y);

            object._rect = *nearestBlob;
            object._scaleToFrame = object._rect.area() * 1. / _currentFrame.size().area();
            object._history.push_back(Point(object._rect.x + object._rect.width / 2.0,
                                            object._rect.y + object._rect.height / 2.0));
            object._updated = true;

            _blobs.erase(nearestBlob);
        }
    }
}

void MotionTracker::createObjects()
{
    if (_blobs.empty())
        return;

    for (auto it = _blobs.begin(), iend = _blobs.end(); it != iend; ++it) {
        MovingObject newObject;
        newObject._rect = *it;
        newObject._scaleToFrame = newObject._rect.area() * 1. / _currentFrame.size().area();
        // newObject._id = Object::_globalID++;
        newObject._color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
        newObject._updated = true;

        _objects.push_back(newObject);
    }
}

void MotionTracker::eraseLostObjects()
{
    if (_objects.empty())
        return;

    list<MovingObject>::iterator it = _objects.begin();
    list<MovingObject>::iterator it_end = _objects.end();
    for (; it != it_end;) {
        if (!it->_updated)
            it = _objects.erase(it);
        else
            ++it;
    }
}

void MotionTracker::matchObject(float minDist)
{
    updateObjects(minDist);
    createObjects();
    eraseLostObjects();
}


void MotionTracker::trackOpticalFlow()
{
    if (_curId < 2)
        return;
    //    buildOpticalFlowPyramid(_currentFrameGray, _currentPyrFrames, Size(5, 5), 5);
    //    Ptr<FarnebackOpticalFlow> fof = FarnebackOpticalFlow::create();
    //    Ptr<DualTVL1OpticalFlow> dof = DualTVL1OpticalFlow::create();
    //    Mat flow(_currentFrameGray.size(), CV_32FC2);
    //    fof->calc(_currentFrameGray, _lastFrameGray, flow);

    // _optFlow - CV_32FC2
//    calcOpticalFlowBM();

    // 基于梯度的光流法
    calcOpticalFlowFarneback(_lastFrameGray, _currentFrameGray, _optFlow, 0.5, 3, 15, 3, 5, 1.1, 0);
    displayOpticalFlow("optical flow");
}


void MotionTracker::displayBlobs(const string& name_of_window)
{
    _currentBlobImage = _currentFrame.clone();

    for (int i = 0, lim = _blobs.size(); i < lim; ++i)
        rectangle(_currentBlobImage, _blobs[i], CV_RGB(0, 255, 0), 1);

    //    imshow(name_of_window, _blockImage);
}

void MotionTracker::displayObjects(const string& name_of_window)
{
    Mat objectImg = _currentFrame.clone();

    for (auto objI = _objects.begin(), oend = _objects.end(); objI != oend; ++objI) {
        list<Point>::const_iterator it = objI->_history.begin();
        list<Point>::const_iterator it_end = objI->_history.end();

        for (; it != it_end; ++it)
            circle(objectImg, *it, 1, objI->_color, 1);

        rectangle(objectImg, objI->_rect, objI->_color, 1);

        Point center =
            Point(objI->_rect.x + objI->_rect.width / 2.0, objI->_rect.y + objI->_rect.height / 2.0);

        line(objectImg, center, center + objI->_speed * 3, CV_RGB(0, 255, 0), 2);
        putText(objectImg, format("%d", objI->_id), Point(objI->_rect.x, objI->_rect.y - 15),
                FONT_HERSHEY_SIMPLEX, 1.0, objI->_color, 2);
    }

    imshow(name_of_window, objectImg);
}

void MotionTracker::displayDetail(const std::string& name_of_window)
{
    assert(_currentBlobImage.type() == _currentFrame.type());

    Mat fore, out, tmp1, tmp2;
    if (_currentFrame.type() == CV_8UC3)
        cvtColor(_currentForeground, fore, COLOR_GRAY2BGR);
    else
        fore = _currentForeground;

    hconcat(_currentFrame, fore, tmp1);
    hconcat(_currentBlobImage, _currentContoursImage, tmp2);
    vconcat(tmp1, tmp2, out);

    putText(out, to_string(_curId), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    imshow(name_of_window, out);
}

void MotionTracker::displayOpticalFlow(const string &name_of_window)
{
    Mat flow = _currentFrame.clone();
    Mat normFlowf = Mat::zeros(_optFlow.size(), CV_32FC1);

    // TODO 考虑边界
    const int cellSize = 10;
    const int cellArea = pow(cellSize, 2);
    const int X = _currentFrame.cols / cellSize;
    const int Y = _currentFrame.rows / cellSize;
    const int N = X * Y;
    vector<vector<Point2f>> flowInCell(N);
    for (int i = 0; i < N; ++i)
        flowInCell[i].reserve(cellArea);

    Scalar color(0, 255, 0);
    for (int r = 0; r < _optFlow.rows; ++r) {
        for (int c = 0; c < _optFlow.cols; ++c) {
            const Point2f& fxy = _optFlow.at<Point2f>(r, c);
            float d = norm(fxy);
            if (d < 1.)
                continue;

            normFlowf.at<float>(r, c) = d;

            int idx = c / cellSize + X * r / cellSize;
            if (idx < N)
                flowInCell[idx].push_back(fxy);

            if (r % cellSize == 0 && c % cellSize == 0) {
                Point2i start(c, r);
                circle(flow, start, 1, color, -1);
                line(flow, start, start + Point2i(cvRound(fxy.y), cvRound(fxy.x)), color);
            }
        }
    }

//    for (size_t i = 0; i < N; ++i) {
//        const size_t n = flowInCell[i].size();
//        if (n < 3)
//            continue;

//        Point2f dir(0.f, 0.f);
//        for (size_t j = 0; j < n; ++j)
//            dir += flowInCell[i][j];
//        dir *= 1./ n;
//        if (norm(dir) < 1)
//            continue;

//        int idx_x = i % X;
//        int idx_y = i / X;
//        Point2i center_i(idx_x * cellSize + 5, idx_y * cellSize + 5);
//        circle(flow, center_i, 1, color, -1);
//        line(flow, center_i, center_i + Point2i(cvRound(dir.y), cvRound(dir.x)), color);
//    }

    Mat normFlow, show;

    normalize(normFlowf, normFlow, 255, 0, NORM_MINMAX, CV_8UC1);
    cvtColor(normFlow, normFlow, COLOR_GRAY2BGR);
    hconcat(normFlow, flow, show);
    imshow(name_of_window, show);
}


}  // namespace ms
