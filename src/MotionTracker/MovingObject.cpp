#include "MovingObject.h"


namespace ms
{

int MovingObject::_globalID = 0;

int MovingObject::splitToCells()
{
    int w = 0, h = 0, n = 0;
    if (_scaleToFrame < 1e-6)
        return 0;

    const int minSize = 5;
    const int minArea = minSize * minSize;
    if (_rect.area() < minArea)
        return 0;

    _cellsXY.width = std::max(_rect.width * 0.5 / minSize, 2.);
    _cellsXY.height = std::max(_rect.height * 0.5 / minSize, 2.);
    n = _cellsXY.width * _cellsXY.height;
    return n;
}

}  // namespace ms
