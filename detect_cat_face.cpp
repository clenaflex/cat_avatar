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

double tan_calc(double x1,double y1,double x2,double y2){
  double la = (y2 - y1)/( (x2 - x1)*10*0.1);
  return la;
}

int calc_score(cv::Point r,cv::Point l,double r_sc,double l_sc){
  double tan_value = tan_calc(l.x,l.y,r.x,r.y);
  double tr_tan = atan(tan_value)*180/3.14;
  int sc = (r_sc+l_sc)*(180 - tr_tan)/180;
  return sc;
}

void compare_score(vector<Rect> objs_r,vector<Rect> objs_l,vector<int> weights_r,vector<int> weights_l,int result[2][4]){
  int min_score=0;
  int sc;
  cv::Point r,l;

  for(int m=0;m < objs_r.size();m++){
    for(int n=0;n < objs_l.size();n++){
      r.x = objs_r[m].x;
      r.y = objs_r[m].y;
      l.x = objs_l[m].x;
      l.y = objs_l[m].y;
      sc = calc_score(r,l,weights_r[m],weights_l[n]);
      if(sc>min_score){
      result[0][0] = objs_r[m].x;
      result[0][1] = objs_r[m].y;
      result[0][2] = objs_r[m].width;
      result[0][3] = objs_r[m].height;

      result[1][0] = objs_l[n].x;
      result[1][1] = objs_l[n].y;
      result[1][2] = objs_l[n].width;
      result[1][3] = objs_l[n].height;
      }
    }
  }
}


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
  int count = 0;
  cascade.load("./cascades/cascade_ear.xml");

  vector<Rect> objs_r,objs_l;
  vector<Rect> objs_list_r;
  vector<Rect> objs_list_l;
  vector<int> reject_levels_r,reject_levels_l;
  vector<int> weights_r,weights_l;
  vector<double> level_weights_r,level_weights_l;

  cascade.detectMultiScale(r_face, objs_r, reject_levels_r, level_weights_r, scale_factor, min_neighbors, 0, cv::Size(r_face.cols/3.1,r_face.rows/3.1), Size(), true);
  cv::groupRectangles(objs_r,weights_r,-1,10);
  int min_y = img_height;
  for (int n = 0; n < objs_r.size(); n++) {
    if (min_y > objs_r[n].y ){
      result[0][0] = objs_r[n].x;
      result[0][1] = objs_r[n].y;
      result[0][2] = objs_r[n].width;
      result[0][3] = objs_r[n].height;
      min_y = objs_r[n].y;
      count++;
    }
  }

  cascade.detectMultiScale(l_face, objs_l, reject_levels_l, level_weights_l, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/3.1,l_face.rows/3.1),Size(), true);
  cv::groupRectangles(objs_l,weights_l,-1,10);
   min_y = img_height;
  for (int n = 0; n < objs_l.size(); n++) {
    if (min_y > objs_l[n].y ){
      result[1][0] = objs_l[n].x+img_width/2;
      result[1][1] = objs_l[n].y;
      result[1][2] = objs_l[n].width;
      result[1][3] = objs_l[n].height;
      min_y = objs_l[n].y;
      count++;
    }
  }
  if (count<1){
    compare_score(objs_r,objs_l,weights_r,weights_l,result);
  }
}

void detect_cat_eyes(cv::Mat cat_face,int result[2][4]){
  cv::Mat dst_cat_face = cat_face.clone();
  int count=0;
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
  vector<Rect> objs_r,objs_l;
  vector<Rect> objs_list_r;
  vector<Rect> objs_list_l;
  vector<int> reject_levels_r,reject_levels_l;
  vector<int> weights_r,weights_l;
  vector<double> level_weights_r,level_weights_l;
  cascade.detectMultiScale(r_face, objs_r, reject_levels_r, level_weights_r, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/6,l_face.rows/6), Size(l_face.cols/2,l_face.rows/2), true);
  int min_y = img_height*2/3;
  int max_y = img_height*1/3;
  int min_x = img_width*2.05/(2*5);
  int max_x = img_width*9/(2*10);
  for (int n = 0; n < objs_r.size(); n++) {
    if (min_y > objs_r[n].y && max_y < objs_r[n].y && min_x < objs_r[n].x && max_x > objs_r[n].x){
      result[0][0] = objs_r[n].x;
      result[0][1] = objs_r[n].y;
      result[0][2] = objs_r[n].width;
      result[0][3] = objs_r[n].height;
      objs_list_r.push_back(objs_r[n]);
    }
  }
  cascade.detectMultiScale(l_face, objs_l, reject_levels_l, level_weights_l, scale_factor, min_neighbors, 0, cv::Size(l_face.cols/5.9,l_face.rows/5.9),Size(l_face.cols/2,l_face.rows/2), true);
   min_y = img_height*2/3;
   max_y = img_height*1/3;
   min_x = img_width*7/200;
   max_x = img_width/(2*3);
  for (int n = 0; n < objs_l.size(); n++) {
    if (min_y > objs_l[n].y && max_y < objs_l[n].y && min_x < objs_l[n].x && max_x > objs_l[n].x ){
      result[1][0] = objs_l[n].x+img_width/2;
      result[1][1] = objs_l[n].y;
      result[1][2] = objs_l[n].width;
      result[1][3] = objs_l[n].height;
      objs_list_l.push_back(objs_l[n]);
    }
  }
  int dif_y_min = img_height;
  for (int i = 0; i <  objs_list_l.size(); i++) {
    for(int j = 0; j < objs_list_r.size();j++ ){
      int dif_y = abs(objs_list_l[i].y+(objs_list_l[i].height/2)- (objs_list_r[j].y+(objs_list_r[j].height/2) ) );
      if(dif_y < dif_y_min){
      count++;
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
  if (count<1){
    compare_score(objs_r,objs_l,weights_r,weights_l,result);
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