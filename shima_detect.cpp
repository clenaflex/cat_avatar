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

using namespace std;

int shima_detect(cv::Mat img){
  cv::Mat dst_img = img.clone();
  string cascade_file = string("./cascades/cascade_shima.xml");
  cv::CascadeClassifier cascade;
  cascade.load(cascade_file);
  // cols:width
  // rows:height
  int img_width,img_height;
  img_width = dst_img.cols;
  img_height = dst_img.rows;
  if(cascade.empty()) {
    cerr << "cannot load cascade file" << endl;
    exit(-1);
  }
  vector<cv::Rect> objects;
  cascade.detectMultiScale(dst_img, objects, 1.1, 3);
  vector<cv::Rect>::const_iterator iter = objects.begin();
  int shima_count=0;
  while(iter!=objects.end()) {
    // cv::rectangle(dst_img,
    //               cv::Rect(iter->x, iter->y, iter->width, iter->height),
    //               cv::Scalar(0, 0, 255), 2);
    if(iter->y <img_height/2 && 0 < iter->y){
      if(iter->x > (img_width/3) && iter->x<(img_width*2/3)){
        shima_count++;
        cout << "(x, y, width, height) = (" << iter->x << ", " << iter->y << ", "<< iter->width << ", " << iter->height << ")" << endl;
        cv::rectangle(dst_img,cv::Rect(iter->x, iter->y, iter->width, iter->height),cv::Scalar(0, 0, 255), 2);
      }
    }
    ++iter;
  }
  cv::imwrite("shima_result.jpg", dst_img);
  return shima_count;
}

// int main(int argc, char** argv) {
//   cv::Mat img_in = cv::imread(argv[1], 1);
//   cv::Mat dst_img = img_in.clone();
//   cout << shima_detect(dst_img) << endl;
//   return 0;
// }