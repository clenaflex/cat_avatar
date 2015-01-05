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
// #include <boost/lexical_cast.hpp>

using namespace std;

double calc_tan_two_line(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4){
  double la = (y2 - y1)/(x2 - x1);
  double lb = y2 - la * x2;
  double lc = (y4 - y3)/(x4 - x3);
  double ld = y4 - lc * x4;
  return fabs((lc -la)/(1+la*lc));
}

void detect_ear_line(cv::Mat& img,int linea[4],int lineb[4]){
  cv::Mat org_img = img.clone();
  cv::cvtColor(img, img, CV_BGR2GRAY);
  cv::Canny(img,img, 50, 200, 3);
  // 確率的Hough変換
  std::vector<cv::Vec4i> lines;
  std::vector<cv::Vec4i> lines2;
  // 入力画像，出力，距離分解能，角度分解能，閾値，線分の最小長さ，
  // 2点が同一線分上にあると見なす場合に許容される最大距離
  cv::HoughLinesP(img, lines, 1, CV_PI/180, 50, 40, 5);
  cv::HoughLinesP(img, lines2, 1, CV_PI/180, 50, 40, 5);

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
      if(0.6 < tan_calc && tan_calc <0.9){
        if(tan_calc > tan_ans){
          tan_ans = tan_calc;
          linea[0] = la[0];
          linea[1] = la[1];
          linea[2] = la[2];
          linea[3] = la[3];
          lineb[0] = lb[0];
          lineb[1] = lb[1];
          lineb[2] = lb[2];
          lineb[3] = lb[3];
        }
      }

    }
  }
  cv::line(org_img, cv::Point(linea[0], linea[1]), cv::Point(linea[2], linea[3]), cv::Scalar(0,0,255), 2, CV_AA);
  cv::line(org_img, cv::Point(lineb[0], lineb[1]), cv::Point(lineb[2], lineb[3]), cv::Scalar(0,0,255), 2, CV_AA);
  cout << "("<< linea[0] << "," << linea[1] << ")" << " to " << "("<< linea[2] << "," << linea[3] << ")" << endl;
  cout << "("<< lineb[0] << "," << lineb[1] << ")" << " to " << "("<< lineb[2] << "," << lineb[3] << ")" << endl;
  img = org_img.clone();
  // std::string filename = boost::lexical_cast<string>(i);
  // filename += ".jpg";
}

int main(int argc, char** argv) {
  cv::Mat src_img = cv::imread(argv[1], 1);
  if(!src_img.data) return -1;
  int linea[4];
  int lineb[4];
  detect_ear_line(src_img,linea,lineb);  
  cv::namedWindow("Ear", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  std::string filename = "result.jpg";
  cv::imshow("Ear",src_img);
  cv::imwrite(filename,src_img);
  cv::waitKey(0);
}

