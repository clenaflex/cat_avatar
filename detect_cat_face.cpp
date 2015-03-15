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
using namespace std;

void wider_rect(cv::Mat src_img,double param, int result[4]){
  int img_width,img_height;
  int center[2];
  int ans[4];
  center[0] = result[0] + result[2]/2;
  center[1] = result[1] + result[3]/2;
  cv::Mat dst_img = src_img.clone();
  img_width = dst_img.cols;
  img_height = dst_img.rows;
  ans[0] = center[0] - (result[2]/2)*param;
  ans[1] = center[1] - (result[3]/2)*param;
  ans[2] = result[2] * param;
  ans[3] = result[3] * param;
  if(ans[0] < 0 || ans[1] < 0 || img_width < ans[2] || img_height < ans[3]){
    wider_rect(dst_img,(param-0.01),result);
  }else{
    result[0] = ans[0];
    result[1] = ans[1];
    result[2] = ans[2];
    result[3] = ans[3];
  }
}

void detect_cat_face(cv::Mat src_img,int result[4]){
    cv::Mat dst_img = src_img.clone();
    string cascade_file = string("./cascades/cascade_face.xml");
    cv::CascadeClassifier cascade;
    cascade.load(cascade_file);
    if(cascade.empty()) {
      cerr << "cannot load cascade file" << endl;
      exit(-1);
    }
    vector<cv::Rect> objects;
    vector<int> weights;
    cascade.detectMultiScale(src_img, objects, 1.1, 3,0,cv::Size(src_img.cols/6,src_img.rows/6),cv::Size());
    vector<cv::Rect>::const_iterator iter = objects.begin();
    int wh=0;
    int count=0;
    while(iter!=objects.end()) {
      if(wh < iter->width){
        result[0] = iter->x;
        result[1] = iter->y;
        result[2] = iter->width;
        result[3] = iter->height;
        count++;
      }
    ++iter;
  }
  if(count>0){
    wider_rect(dst_img,1.19,result);  
  }else{
    exit(1);
  }
  
}



void detect_cat_ears(cv::Mat cat_face,int result[2][4]){
  cv::Mat dst_cat_face = cat_face.clone();
  int img_width,img_height;
  img_width = dst_cat_face.cols;
  img_height = dst_cat_face.rows;
  //猫から見てr
  cv::Mat r_face(dst_cat_face, cv::Rect(0,0,img_width/2,img_height));
  cv::Mat l_face(dst_cat_face, cv::Rect(img_width/2,0,img_width/2,img_height));

  CascadeClassifier cascade;
  const float scale_factor(1.1);
  const int min_neighbors(3);
  cascade.load("./cascades/cascade_ear.xml");
  vector<Rect> objs;
  vector<int> reject_levels;
  vector<int> weights;
  vector<double> level_weights;
  cascade.detectMultiScale(r_face, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, cv::Size(r_face.cols/3.1,r_face.rows/3.1), Size(), true);
  cv::groupRectangles(objs,weights,-1,10);
  int min_y = img_height;
  for (int n = 0; n < objs.size(); n++) {
    if (min_y > objs[n].y ){
      result[0][0] = objs[n].x;
      result[0][1] = objs[n].y;
      result[0][2] = objs[n].width;
      result[0][3] = objs[n].height;
      min_y = objs[n].y;
    }
  }

  cascade.detectMultiScale(l_face, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/3.1,l_face.rows/3.1),Size(), true);
  cv::groupRectangles(objs,weights,-1,10);
   min_y = img_height;
  for (int n = 0; n < objs.size(); n++) {
    if (min_y > objs[n].y ){
      result[1][0] = objs[n].x+img_width/2;
      result[1][1] = objs[n].y;
      result[1][2] = objs[n].width;
      result[1][3] = objs[n].height;
      min_y = objs[n].y;
    }
  }
}

void detect_cat_eyes(cv::Mat cat_face,int result[2][4]){
  cv::Mat dst_cat_face = cat_face.clone();
  int img_width,img_height;
  img_width = dst_cat_face.cols;
  img_height = dst_cat_face.rows;
  //猫から見てr
  cv::Mat r_face(dst_cat_face, cv::Rect(0,0,img_width/2,img_height));
  cv::Mat l_face(dst_cat_face, cv::Rect(img_width/2,0,img_width/2,img_height));

  CascadeClassifier cascade;
  const float scale_factor(1.1);
  const int min_neighbors(3);
  cascade.load("./cascades/cascade_eye.xml");
  vector<Rect> objs;
  vector<Rect> objs_list_r;
  vector<Rect> objs_list_l;
  vector<int> reject_levels;
  vector<int> weights;
  vector<double> level_weights;
  cascade.detectMultiScale(r_face, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/6,l_face.rows/6), Size(l_face.cols/2,l_face.rows/2), true);
  int min_y = img_height*2/3;
  int max_y = img_height*1/3;
  int min_x = img_width*2.05/(2*5);
  int max_x = img_width*9/(2*10);
  for (int n = 0; n < objs.size(); n++) {
    if (min_y > objs[n].y && max_y < objs[n].y && min_x < objs[n].x && max_x > objs[n].x){
      result[0][0] = objs[n].x;
      result[0][1] = objs[n].y;
      result[0][2] = objs[n].width;
      result[0][3] = objs[n].height;
      objs_list_r.push_back(objs[n]);
    }
  }
  cascade.detectMultiScale(l_face, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/5.9,l_face.rows/5.9),Size(l_face.cols/2,l_face.rows/2), true);
   min_y = img_height*2/3;
   max_y = img_height*1/3;
   min_x = img_width*7/200;
   max_x = img_width/(2*3);
  for (int n = 0; n < objs.size(); n++) {
    if (min_y > objs[n].y && max_y < objs[n].y && min_x < objs[n].x && max_x > objs[n].x ){
      result[1][0] = objs[n].x+img_width/2;
      result[1][1] = objs[n].y;
      result[1][2] = objs[n].width;
      result[1][3] = objs[n].height;
      objs_list_l.push_back(objs[n]);
    }
  }
  int dif_y_min = img_height;
  for (int i = 0; i <  objs_list_l.size(); i++) {
    for(int j = 0; j < objs_list_r.size();j++ ){
      int dif_y = abs(objs_list_l[i].y+(objs_list_l[i].height/2)- (objs_list_r[j].y+(objs_list_r[j].height/2) ) );
      if(dif_y < dif_y_min){
      result[0][0] = objs_list_r[j].x;
      result[0][1] = objs_list_r[j].y;
      result[0][2] = objs_list_r[j].width;
      result[0][3] = objs_list_r[j].height;

      result[1][0] = objs_list_l[i].x+img_width/2;
      result[1][1] = objs_list_l[i].y;
      result[1][2] = objs_list_l[i].width;
      result[1][3] = objs_list_l[i].height;
      dif_y_min = dif_y;
      }
    }
  }
}


void detect_cat_mouth(cv::Mat cat_face,int result[4]){
  cv::Mat dst_cat_face = cat_face.clone();
  int img_width,img_height;
  img_width = dst_cat_face.cols;
  img_height = dst_cat_face.rows;

  CascadeClassifier cascade;
  const float scale_factor(1.1);
  const int min_neighbors(3);
  cascade.load("./cascades/cascade_mouth.xml");
  vector<Rect> objs;
  vector<int> reject_levels;
  vector<int> weights;
  vector<double> level_weights;
  cascade.detectMultiScale(dst_cat_face, objs, reject_levels, level_weights, scale_factor, min_neighbors, 0, cv::Size(dst_cat_face.cols/5,dst_cat_face.rows/5), Size(dst_cat_face.cols/3,dst_cat_face.rows/3), true);
  cv::groupRectangles(objs,weights,-1,10);
  int min_x = img_width/2;
  for (int n = 0; n < objs.size(); n++) {
    rectangle(dst_cat_face, cv::Rect(objs[n].x,objs[n].y,objs[n].width,objs[n].height), Scalar(255,0,0), 8);
    if (min_x > fabs( img_width/2 - (objs[n].x+objs[n].width/2) ) && (objs[n].y+objs[n].height/2) > (img_height*2/3) ){
      result[0] = objs[n].x;
      result[1] = objs[n].y;
      result[2] = objs[n].width;
      result[3] = objs[n].height;
      min_x = fabs( img_width/2 - (objs[n].x+objs[n].width/2) );
    }
  }
  if(min_x > 20){
    //error
    result[0] = 0;
    result[1] = 0;
    result[2] = 0;
    result[3] = 0;
  }
}

