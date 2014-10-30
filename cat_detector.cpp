#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

int detect_cat_face(const char *path_img,int detected_rects[100][4]) {
  //int *detected_rects;
  int i = 1 ;
  using namespace std;
  cv::Mat src_img = cv::imread(path_img, 1);
  if(src_img.empty()) {
    cerr << "cannot load image" << endl;
    exit(-1);
  }
  cv::Mat dst_img = src_img.clone();
  cv::CascadeClassifier cascade;
  cascade.load("./cascade_face/cascade.xml");
  if(cascade.empty()) {
    cerr << "cannot load cascade file" << endl;
    exit(-1);
  }
  vector<cv::Rect> objects;
  cascade.detectMultiScale(src_img, objects, 1.1, 3);
  vector<cv::Rect>::const_iterator iter = objects.begin();
  cout << "face_count: " << objects.size() << endl;
  
  int a = src_img.size().width;
  detected_rects[0][0] = src_img.size().width;
  detected_rects[0][1] = src_img.size().height;
  detected_rects[0][2] = objects.size();
  while(iter!=objects.end()) {
    // cout << "(x, y, width, height) = (" << iter->x << ", " << iter->y << ", "
    //      << iter->width << ", " << iter->height << ")" << endl;
    cv::rectangle(dst_img,
                  cv::Rect(iter->x, iter->y, iter->width, iter->height),
                  cv::Scalar(0, 0, 255), 2);
    detected_rects[i][0] = iter->x;
    detected_rects[i][1] = iter->y;
    detected_rects[i][2] = iter->width;
    detected_rects[i][3] = iter->height; 
    ++iter;
    i++;
  }
  cv::imwrite("face.jpg", dst_img);
  return 0;
}

int detect_cat_eye(const char *path_img,int detected_rects[100][4]) {
  //int *detected_rects;
  int i = 1 ;
  using namespace std;
  cv::Mat src_img = cv::imread(path_img, 1);
  if(src_img.empty()) {
    cerr << "cannot load image" << endl;
    exit(-1);
  }
  cv::Mat dst_img = src_img.clone();
  cv::CascadeClassifier cascade;
  cascade.load("./cascade_eye/cascade.xml");
  if(cascade.empty()) {
    cerr << "cannot load cascade file" << endl;
    exit(-1);
  }
  vector<cv::Rect> objects;
  cascade.detectMultiScale(src_img, objects, 1.1, 3);
  vector<cv::Rect>::const_iterator iter = objects.begin();
  cout << "eye_count: " << objects.size() << endl;
  
  int a = src_img.size().width;
  detected_rects[0][0] = src_img.size().width;
  detected_rects[0][1] = src_img.size().height;
  detected_rects[0][2] = objects.size();
  while(iter!=objects.end()) {
    // cout << "(x, y, width, height) = (" << iter->x << ", " << iter->y << ", "
    //      << iter->width << ", " << iter->height << ")" << endl;
    cv::rectangle(dst_img,
                  cv::Rect(iter->x, iter->y, iter->width, iter->height),
                  cv::Scalar(0, 0, 255), 2);
    detected_rects[i][0] = iter->x;
    detected_rects[i][1] = iter->y;
    detected_rects[i][2] = iter->width;
    detected_rects[i][3] = iter->height; 
    ++iter;
    i++;
  }
  cv::imwrite("eyes.jpg", dst_img);
  return 0;
}

int detect_cat_ear(const char *path_img,int detected_rects[100][4]) {
  //int *detected_rects;
  int i = 1 ;
  using namespace std;
  cv::Mat src_img = cv::imread(path_img, 1);
  if(src_img.empty()) {
    cerr << "cannot load image" << endl;
    exit(-1);
  }
  cv::Mat dst_img = src_img.clone();
  cv::CascadeClassifier cascade;
  cascade.load("./cascade_ear/cascade.xml");
  if(cascade.empty()) {
    cerr << "cannot load cascade file" << endl;
    exit(-1);
  }
  vector<cv::Rect> objects;
  cascade.detectMultiScale(src_img, objects, 1.1, 3);
  vector<cv::Rect>::const_iterator iter = objects.begin();
  cout << "ear_count: " << objects.size() << endl;
  
  int a = src_img.size().width;
  detected_rects[0][0] = src_img.size().width;
  detected_rects[0][1] = src_img.size().height;
  detected_rects[0][2] = objects.size();
  while(iter!=objects.end()) {
    // cout << "(x, y, width, height) = (" << iter->x << ", " << iter->y << ", "
    //      << iter->width << ", " << iter->height << ")" << endl;
    cv::rectangle(dst_img,
                  cv::Rect(iter->x, iter->y, iter->width, iter->height),
                  cv::Scalar(0, 0, 255), 2);
    detected_rects[i][0] = iter->x;
    detected_rects[i][1] = iter->y;
    detected_rects[i][2] = iter->width;
    detected_rects[i][3] = iter->height; 
    ++iter;
    i++;
  }
  cv::imwrite("ears.jpg", dst_img);
  return 0;
}


int main(int argc, char* argv[])
{
  char *path_img;
  int detected_face_rects[100][4];
  int detected_eyes_rects[100][4];
  int detected_ears_rects[100][4];
  path_img = argv[1];
  detect_cat_face(path_img,detected_face_rects);
  detect_cat_eye(path_img,detected_eyes_rects);
  detect_cat_ear(path_img,detected_ears_rects);
  printf("%d,%d,%d\n",detected_face_rects[0][0],detected_face_rects[0][1],detected_face_rects[0][2]);

  for(int i = 1; i < (detected_face_rects[0][2]+1) ; i++){
    printf("%d,%d,%d,%d\n",detected_face_rects[i][0],detected_face_rects[i][1],detected_face_rects[i][2],detected_face_rects[i][3]);
  }
  return 0;
}