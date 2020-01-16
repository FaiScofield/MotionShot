#ifndef MOTION_TRACKER_H
#define MOTION_TRACKER_H

#include "BaseBackgroundSubtractor.h"
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>

namespace ms
{

class Object {
public:
    Object(): _id(_globalID++), _updated(false) {}

    int splitToCells();

    static int _globalID;
    int _id;
    bool _updated;

    double _scaleToFrame;
    cv::Rect _rect;
    cv::Scalar _color;
    cv::Point _speed;
    std::list<cv::Point> _history;
    std::vector<std::vector<cv::Point>> _contours;

    cv::Size _cellsXY;
    std::vector<cv::Rect> _cells;
    std::vector<bool> _validCell;
};

class MotionTracker
{
public:
    MotionTracker();
    MotionTracker(BaseBackgroundSubtractor* bs);
    ~MotionTracker();

    void SetBackgroundSubtractor(BaseBackgroundSubtractor* bs){ _substractor = bs; }
    void SetMinBlobSize(const cv::Size& s) { _minBlobSize = s; }
    std::vector<cv::Rect> GetBlobs() const { return _blobs; }
    std::list<Object> GetObjects() const { return _objects; }


    void SubstractBackground(const cv::Mat& input, cv::Mat& mask);
    void DetectBlocks();
    void MatchObject(float minDist = 30);

    void DisplayObjects(const std::string& name_of_window);
    void DisplayBlobs(const std::string& name_of_window);
    void DisplayDetail(const std::string& name_of_window);

    // 光流计算


private:
    void BlobFilter(const std::vector<std::vector<cv::Point>>& contours);
    void UpdateObjects(float minDist);
    void CreateObjects();
    void EraseLostObjects();

    int _curId;
    bool _defaultConstruct;

    BaseBackgroundSubtractor* _substractor;

    std::list<Object> _objects;
    std::vector<cv::Rect> _blobs;

    cv::Mat _frame, _foreground;
    cv::Mat _blobImage, _contoursImage;

    cv::Size _minBlobSize;
};

}  // namespace ms
#endif  // MOTION_TRACKER_H
