#include "opencv2/opencv.hpp"
#include <iostream>
 
using namespace cv;
using namespace std;
 
int main(int argc, char** argv)
{
    // Open another image
    Mat image;
    image = cv::imread(argv[1], 1);
 
    if(! image.data ) // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
 
    // define bounding rectangle
    int border = 20;
    int border2 = border + border;
    cv::Rect rectangle(border,border,image.cols-border2,image.rows-border2);
 
    cv::Mat result; // segmentation result (4 possible values)
    cv::Mat bgModel,fgModel; // the models (internally used)
 
    // GrabCut segmentation
    cv::grabCut(image,    // input image
        result,   // segmentation result
        rectangle,// rectangle containing foreground
        bgModel,fgModel, // models
        1,        // number of iterations
        cv::GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground
    cv::compare(result,cv::GC_PR_FGD,result,cv::CMP_EQ);
    // Generate output image
    cv::Mat foreground(image.size(),CV_8UC3,cv::Scalar(255,255,255));
    image.copyTo(foreground,result); // bg pixels not copied
 
    // draw rectangle on original image
    cv::rectangle(image, rectangle, cv::Scalar(255,255,255),1);
    cv::namedWindow("Image");
    cv::imshow("Image",image);
 
    // display result
    cv::namedWindow("Segmented Image");
    cv::imshow("Segmented Image",foreground);
 
 
    waitKey();
    return 0;
 
}