#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>
 
using namespace cv;
 
int main(int argc, char **argv) {
 
    CascadeClassifier cascade;
    const float scale_factor(1.1);
    const int min_neighbors(3.0);
 
    if (cascade.load("./cascade_ear_new/cascade.xml")) {
 
        for (int i = 1; i < argc; i++) {
 
            Mat img = imread(argv[i], CV_LOAD_IMAGE_GRAYSCALE);
            // equalizeHist(img, img);
            vector<Rect> objs;
            vector<int> reject_levels;
            vector<int> weights;
            vector<double> level_weights;
            cascade.detectMultiScale(img, objs, reject_levels, level_weights, scale_factor, min_neighbors, 1, cv::Size(img.cols/4,img.rows/4), Size(), true);

            cv::groupRectangles(objs,weights,2,0.5);

            Mat img_color = imread(argv[i], CV_LOAD_IMAGE_COLOR);
            for (int n = 0; n < objs.size(); n++) {
                rectangle(img_color, objs[n], Scalar(255,0,0), 8);
                putText(img_color, boost::lexical_cast<string>(weights[n]),Point(objs[n].x, objs[n].y), 1, 1, Scalar(0,0,255));
            }
            imshow("VJ Face Detector", img_color);
            waitKey(0);
        }
    }
 
    return 0;
}