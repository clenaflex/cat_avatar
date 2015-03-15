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

using namespace std;

int line_cross_point_y(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4){
  double la = (y2 - y1)/( (x2 - x1) *10*0.1 );
  double lb = y2 - la * x2;
  double lc = (y4 - y3)/( (x4 - x3 *10*0.1) );
  double ld = y4 - lc * x4;
  double x = (ld - lb) / ( (la - lc)*10*0.1 );
  double y = la * x + lb;
  return y;
}

int line_cross_point_x(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4){
  double la = (y2 - y1)/( (x2 - x1)*10*0.1 );
  double lb = y2 - la * x2;
  double lc = (y4 - y3)/( (x4 - x3)*10*0.1 );
  double ld = y4 - lc * x4;
  double x = (ld - lb) / ( (la - lc)*10*0.1 );
  return x;
}

double calc_tan_single(double x1,double y1,double x2,double y2){
  double la = (y2 - y1)/( (x2 - x1)*10*0.1);
  return la;
}

double calc_tan_two_line(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4){
  double la = (y2 - y1)/((x2 - x1)*10*0.1 );
  double lc = (y4 - y3)/( (x4 - x3)*10*0.1 );
  return fabs((lc -la)/( (1+la*lc)*10*0.1 ));
}

int detect_ear_line(cv::Mat& img,int linea[4],int lineb[4]){
  //検出成功時は3以上の値をreturnする
  int sc_flag = 2;
  cv::Mat org_img = img.clone();
  cv::Mat origin_img = img.clone();
  int img_height = org_img.rows;
  int image_width = img_height; 
  cv::cvtColor(img, img, CV_BGR2GRAY);
  cv::Canny(img,img, 50, 100, 3);
  std::vector<cv::Vec4i> lines;
  std::vector<cv::Vec4i> lines2;
  cv::HoughLinesP(img, lines, 1, CV_PI/180, 50, 40, 10);
  cv::HoughLinesP(img, lines2, 1, CV_PI/180, 50, 40, 10);

  std::vector<cv::Vec4i>::iterator ita = lines.begin();
  std::vector<cv::Vec4i>::iterator itb = lines2.begin();
  double tan_calc = 0;
  double tan_ans = 0;
  for(; ita !=lines.end();ita++){
    cv::Vec4i la = *ita;
    std::vector<cv::Vec4i>::iterator itb = lines2.begin();
    for(; itb !=lines2.end();itb++){
      cv::Vec4i lb = *itb;
      tan_calc = calc_tan_two_line(la[0],la[1],la[2],la[3],lb[0],lb[1],lb[2],lb[3]);
      double calc_tan_sa = calc_tan_single(la[0],la[1],la[2],la[3]);
      double calc_tan_sb = calc_tan_single(lb[0],lb[1],lb[2],lb[3]);
      if(calc_tan_sa > 0.5 && calc_tan_sb > 0.5   ){
      if(0.7 < tan_calc && tan_calc <2){
        if(tan_calc > tan_ans){
          if(line_cross_point_y(la[0],la[1],la[2],la[3],lb[0],lb[1],lb[2],lb[3]) < img_height*1/10 && line_cross_point_y(la[0],la[1],la[2],la[3],lb[0],lb[1],lb[2],lb[3]) >0 && 0 < line_cross_point_x(la[0],la[1],la[2],la[3],lb[0],lb[1],lb[2],lb[3]) && line_cross_point_x(la[0],la[1],la[2],la[3],lb[0],lb[1],lb[2],lb[3]) < image_width){
          tan_ans = tan_calc;
          linea[0] = la[0];
          linea[1] = la[1];
          linea[2] = la[2];
          linea[3] = la[3];
          lineb[0] = lb[0];
          lineb[1] = lb[1];
          lineb[2] = lb[2];
          lineb[3] = lb[3];
          sc_flag++;
          }
        }
      }
}
    }
  }
  return sc_flag;
}