#ifndef MOVING_OBJECT_H
#define MOVING_OBJECT_H

#include <list>
#include <opencv2/core/core.hpp>

namespace ms
{

class MovingObject
{
public:
    MovingObject() : _id(_globalID++), _updated(false) {}

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

int MovingObject::_globalID = 0;

}  // namespace ms
#endif  // MOVING_OBJECT_H
