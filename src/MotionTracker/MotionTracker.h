#ifndef MOTION_TRACKER_H
#define MOTION_TRACKER_H

#include "MovingObject.h"
#include "BaseMotionDetector.h"
#include <opencv2/video/video.hpp>
#include <list>

namespace ms
{

class MovingObject;

class MotionTracker
{
public:
    MotionTracker();
    MotionTracker(BaseMotionDetector* bs);
    ~MotionTracker();

    void setBackgroundSubtractor(BaseMotionDetector* bs){ _substractor = bs; }
    void setMinBlobSize(const cv::Size& s) { _minBlobSize = s; }
    std::vector<cv::Rect> getBlobs() const { return _blobs; }
    std::list<MovingObject> getObjects() const { return _objects; }

    void substractBackground(const cv::Mat& input, cv::Mat& mask);
    void detectBlocks();
    void matchObject(float minDist = 30);

    void displayObjects(const std::string& name_of_window);
    void displayBlobs(const std::string& name_of_window);
    void displayDetail(const std::string& name_of_window);
    void displayOpticalFlow(const std::string& name_of_window);

    // 光流计算
    void trackOpticalFlow();


private:
    void blobFilter(const std::vector<std::vector<cv::Point>>& contours);
    void updateObjects(float minDist);
    void createObjects();
    void eraseLostObjects();

    int _curId;
    bool _defaultConstruct;

    BaseMotionDetector* _substractor;

    std::list<MovingObject> _objects;
    std::vector<cv::Rect> _blobs;

    cv::Mat _lastFrame, _lastFrameGray, _lastForeground;
    cv::Mat _currentFrame, _currentFrameGray, _currentForeground;
    cv::Mat _currentBlobImage, _currentContoursImage;
    std::vector<cv::Mat> _currentPyrFrames;
    cv::Mat _optFlow;

    cv::Size _minBlobSize;
};

}  // namespace ms
#endif  // MOTION_TRACKER_H
