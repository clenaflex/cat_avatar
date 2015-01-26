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

double which_min(double a,double b){
  if(a>b){
    return b;
  }else if(b>a){
    return a;
  }else{
    return a;
  }
}

double which_max(double a,double b){
  if(a<b){
    return b;
  }else if(b<a){
    return a;
  }else{
    return a;
  }
}

void ellipse_wh_calc(int a, int b ,int theta,int width_height[2]){
  cv::Mat ellipse_box;
  int center;
  if(a>b){
    center = a;
  }else{
    center = b;
  }
  int max_x = 0;
  int max_y = 0;
  ellipse_box = cv::Mat::zeros(cv::Size(3*center, 3*center), CV_8UC3);
  cv::ellipse(ellipse_box,cv::Point(3*center/2,3*center/2), cv::Size(a,b),theta,0,360,cv::Scalar(255,255,255), -1, 4);
  // std::string filename = boost::lexical_cast<string>(a);
  // filename += ".jpg";
  // cv::imwrite(filename, ellipse_box);
  for(int y = 0 ; y < ellipse_box.rows; y++){
    for(int x = 0 ; x < ellipse_box.cols; x++){
      cv::Vec3b bgr = ellipse_box.at<cv::Vec3b>(y,x);
      if (bgr[0] == 255){
        if( x > max_x){
          max_x = x;
        }
      if( y > max_y){
          max_y = y;
        }
      }
    }
  }
  width_height[0] = (max_x - (3*center/2))*2;
  width_height[1] = (max_y - (3*center/2))*2;
}

int detect_eye_ellipse(cv::Mat img,double eye[10]){
  /*
  eye[0]:p
  eye[1]:q
  eye[2]:a
  eye[3]:b
  eye[4]:theta
  */
  cv::Mat img_in = img.clone();
  cv::Mat org_img_in = img.clone();
  cv::Mat img_gray;
  cv::cvtColor(img_in, img_gray, CV_RGB2GRAY);
  cv::Mat sobelX, sobelY, norm, dir;
  cv::Sobel(img_gray, sobelX, CV_32F, 1, 0);
  cv::Sobel(img_gray, sobelY, CV_32F, 0, 1);
  cv::cartToPolar(sobelX, sobelY, norm, dir);
  int flag = 2;
  int flag_2 = 2;
  
  double sob_min, sob_max;
  cv::minMaxLoc(norm, &sob_min, &sob_max);
  
  cv::Mat img_sobel;
  norm.convertTo(img_sobel, CV_8U, -255.0 / sob_max, 255);
  cv::Mat gray_img, bin_img;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<std::vector<cv::Point> > contours_2;
  // 画像の二値化
  cout << "error" << endl;
  cv::threshold(img_sobel, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

  cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  cv::findContours(bin_img, contours_2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  int ellipse_width_height[2];
  int img_width = img_in.size().width;
  int img_height = img_in.size().height;
  int decide_count=0;
  for(int i = 0; i < contours.size(); ++i) {

    size_t count = contours[i].size();
    if(count < 100 || count > 1000) continue; // （小さすぎる|大きすぎる）輪郭を除外
    cv::Mat org_img = img_in.clone();
    cv::Mat pointsf;
    cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
    // 楕円フィッティング
      cout << "error4" << endl;
    cv::RotatedRect box = cv::fitEllipse(pointsf);
          cout << "error5" << endl;
    if (box.center.y > 0 && box.center.x >0){
    ellipse_wh_calc(box.size.width/2,box.size.height/2,box.angle,ellipse_width_height);
      if(box.center.x- (ellipse_width_height[0]/2) > 0 && box.center.y-(ellipse_width_height[1]/2) > 0 && box.center.x+ (ellipse_width_height[0]/2)< img_width && box.center.y+(ellipse_width_height[1]/2) < img_height ){

        //目の輪郭描画
        // cv::ellipse(org_img, box, cv::Scalar(0,0,255), 2, CV_AA);

        eye[0] = box.center.x;
        eye[1] = box.center.y;
        eye[2] = box.size.width/2;
        eye[3] = box.size.height/2;
        eye[4] = box.angle;
        flag++;
        decide_count = count;
        //瞳孔描画サンプル
        // cv::ellipse(org_img,cv::Point(box.center.x,box.center.y), cv::Size(which_max(box.size.height,box.size.width)/5,which_min(box.size.height,box.size.width)*2/5), 0 ,0,360,cv::Scalar(0,0,255), -1, 4);
        // cv::ellipse(org_img_in, box, cv::Scalar(0,0,255), 2, CV_AA);
        // int num = count;
        // cout<< "Eye" << num << endl;
        // std::string filename = boost::lexical_cast<string>(num)+".png";
        // cv::imwrite(filename,org_img_in);
        // org_img_in = img_in.clone();
      }

    }

  }
  cout << "decide_count" <<decide_count << endl;
  int allow =  which_min(eye[2],eye[3])/4;
  for(int i = 0; i < contours_2.size(); ++i) {

    size_t count_2 = contours_2[i].size();
cout << "count_2: " << count_2 << endl;
    if(count_2 > decide_count  ||  count_2 < 10) continue; //  輪郭より大きいのを除外
    // cv::Mat org_img = img_in.clone();
    cv::Mat pointsf_2;
    cv::Mat(contours_2[i]).convertTo(pointsf_2, CV_32F);
    // 楕円フィッティング
    cv::RotatedRect box_2 = cv::fitEllipse(pointsf_2);


        cv::ellipse(org_img_in, box_2, cv::Scalar(0,0,255), 2, CV_AA);
        int num = count_2;
        cout<< "Eye" << num << endl;
        std::string filename = boost::lexical_cast<string>(num)+".png";
        cv::imwrite(filename,org_img_in);
        org_img_in = img_in.clone();

   if(  which_min(eye[2],eye[3]) > which_max(box_2.size.width/2,box_2.size.height/2) && flag>2 ){
    if (box_2.center.y>(eye[1]-allow) && box_2.center.y<(eye[1]+allow)){
      if(box_2.center.x>(eye[0]-allow) && box_2.center.x<(eye[0]+allow)){
        eye[5] = box_2.center.x;
        eye[6] = box_2.center.y;
        eye[7] = box_2.size.width/2;
        eye[8] = box_2.size.height/2;
        eye[9] = box_2.angle;
        flag_2++;

        cv::ellipse(org_img_in, box_2, cv::Scalar(0,0,255), 2, CV_AA);
        int num = count_2;
        cout<< "Eye" << num << endl;
        std::string filename = boost::lexical_cast<string>(num)+".png";
        cv::imwrite(filename,org_img_in);
        org_img_in = img_in.clone();
      }
    }
    }
  }
  if(flag>2 && flag_2<3){
    eye[5] = eye[0];
    eye[6] = eye[1];
    eye[7] = which_min(eye[2],eye[3])*0.8;
    eye[8] = which_max(eye[2],eye[3])*0.2;
    eye[9] = 90;
    cout<< "Can't detect pupil" << endl;
    flag = 99;
  }

  cout<< "Eye detect finished" << endl;
  return flag;
}


int main(int argc, char **argv) {
     cout << "error" << endl;
  cv::Mat img_in = cv::imread(argv[1], 1);
  double eye[10];
   cout << "error" << endl;
  detect_eye_ellipse(img_in,eye);
  cv::ellipse(img_in,cv::Point(eye[0],eye[1]), cv::Size(eye[2],eye[3]), eye[4] ,0,360,cv::Scalar(0,0,255), 3, 4);
  cv::ellipse(img_in,cv::Point(eye[5],eye[6]), cv::Size(eye[7],eye[8]), eye[9] ,0,360,cv::Scalar(0,255,0), 3, 4);
  cv::namedWindow("Eye", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  // std::string filename = boost::lexical_cast<string>(変数);
  std::string filename = "result.jpg";
  cv::imshow("Eye", img_in);
  cv::imwrite(filename,img_in);
  cv::waitKey(0);
}