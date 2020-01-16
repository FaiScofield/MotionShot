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

int Object::_globalID = 0;

int Object::splitToCells() {
    int w = 0, h = 0, n = 0;
    if (_scaleToFrame < 1e-6)
        return 0;

    const int minSize = 5;
    const int minArea = minSize * minSize;
    if (_rect.area() < minArea)
        return 0;

    _cellsXY.width = max(_rect.width * 0.5 / minSize, 2.);
    _cellsXY.height = max(_rect.height * 0.5 / minSize, 2.);
    n = _cellsXY.width * _cellsXY.height;
    return n;
}

MotionTracker::MotionTracker() : _curId(0), _defaultConstruct(true)
{
    _minBlobSize = Size(3, 3);
    auto cvBS = createBackgroundSubtractorMOG2(500, 200, true);
    BS_MOG2_CV be(cvBS.get());
    _substractor = dynamic_cast<BaseBackgroundSubtractor*>(&be);
}

MotionTracker::MotionTracker(BaseBackgroundSubtractor* bs)
    : _curId(0), _defaultConstruct(false), _substractor(bs)
{
     _minBlobSize = Size(3, 3);
}

MotionTracker::~MotionTracker()
{
    if (_defaultConstruct)
        delete _substractor;
}

void MotionTracker::SubstractBackground(const Mat& input, Mat& mask)
{
    _substractor->apply(input, mask);
    assert(!mask.empty());
//    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
//    erode(mask, mask, element);   // 腐蚀
//    dilate(mask, mask, element);  // 膨胀
//    dilate(mask, mask, element);  // 膨胀
//    erode(mask, mask, element);   // 腐蚀
//    dilate(mask, mask, Mat());

    _frame = input.clone();
    _foreground = mask.clone();
    _curId++;
}

void MotionTracker::DetectBlocks()
{
    vector<vector<Point>> contours;
    findContours(_foreground, contours, RETR_EXTERNAL, CHAIN_APPROX_TC89_KCOS); // CHAIN_APPROX_TC89_L1, CHAIN_APPROX_NONE

    _contoursImage = _frame.clone();
    drawContours(_contoursImage, contours, -1, Scalar(0, 255, 0), -1);

    BlobFilter(contours);

    DisplayBlobs("blobsFilteredImage");
}

void MotionTracker::BlobFilter(const vector<vector<Point>>& contours)
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
            /*it = */_blobs.erase(it);
            continue;
        }
        if (it->area() < maxArea * 0.1) {
            /*it = */_blobs.erase(it);
            continue;
        }
//        it++;
    }
}

void MotionTracker::UpdateObjects(float minDist)
{
    for (auto objI = _objects.begin(), oend = _objects.end(); objI != oend; ++objI) {
        Object& object = *objI;
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
            object._scaleToFrame = object._rect.area() * 1. / _frame.size().area();
            object._history.push_back(Point(object._rect.x + object._rect.width / 2.0,
                                            object._rect.y + object._rect.height / 2.0));
            object._updated = true;

            _blobs.erase(nearestBlob);
        }
    }
}

void MotionTracker::CreateObjects()
{
    if (_blobs.empty())
        return;

    for (auto it = _blobs.begin(), iend = _blobs.end(); it != iend; ++it) {
        Object newObject;
        newObject._rect = *it;
        newObject._scaleToFrame = newObject._rect.area() * 1. / _frame.size().area();
        //newObject._id = Object::_globalID++;
        newObject._color = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
        newObject._updated = true;

        _objects.push_back(newObject);
    }
}

void MotionTracker::EraseLostObjects()
{
    if (_objects.empty())
        return;

    list<Object>::iterator it = _objects.begin();
    list<Object>::iterator it_end = _objects.end();
    for (; it != it_end;) {
        if (!it->_updated)
            it = _objects.erase(it);
        else
            ++it;
    }
}

void MotionTracker::MatchObject(float minDist)
{
    UpdateObjects(minDist);
    CreateObjects();
    EraseLostObjects();
}


void MotionTracker::DisplayBlobs(const string& name_of_window)
{
    _blobImage = _frame.clone();

    for (int i = 0, lim = _blobs.size(); i < lim; ++i)
        rectangle(_blobImage, _blobs[i], CV_RGB(0, 255, 0), 1);

    //    imshow(name_of_window, _blockImage);
}

void MotionTracker::DisplayObjects(const string& name_of_window)
{
    Mat objectImg = _frame.clone();

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

void MotionTracker::DisplayDetail(const std::string& name_of_window)
{
    assert(_blobImage.type() == _frame.type());

    Mat fore, out, tmp1, tmp2;
    if (_frame.type() == CV_8UC3)
        cvtColor(_foreground, fore, COLOR_GRAY2BGR);
    else
        fore = _foreground;

    hconcat(_frame, fore, tmp1);
    hconcat(_blobImage, _contoursImage, tmp2);
    vconcat(tmp1, tmp2, out);

    putText(out, to_string(_curId), Point(15, 15), 1, 1, Scalar(0, 0, 255), 2);
    imshow(name_of_window, out);
}


}  // namespace ms
