#include "ImageStitcher/ImageStitcher.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        cerr << "Parameters: image1 image2" << endl;
        exit(-1);
    }

    Mat image1 = imread(argv[1]);
    Mat image2 = imread(argv[2]);
    Mat tmp;
    hconcat(image1, image2, tmp);
    imshow("original images", tmp);

    vector<Mat> vImages;
    vImages.push_back(image2);
    vImages.push_back(image1);

    Mat pano;
    Stitcher stitcher = Stitcher::createDefault(false);
    Stitcher::Status status = stitcher.stitch(vImages, pano);
    if (status != Stitcher::OK) {
        cerr << "Can't stitch images, error code = " << int(status) << endl;
        system("pause");
        return -1;
    }

    imshow("Result Pano", pano);
    imwrite("/home/vance/output/result_pano.bmp", pano);

    // test
    ms::ImageStitcher is;
    Mat H21 = is.computeHomography(image1, image2);
    cout << H21 << endl;

    waitKey(0);

    return 0;
}
