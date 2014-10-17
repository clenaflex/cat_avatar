#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;

int main(int argc, char** argv) {
  cv::Mat img_in = cv::imread(argv[1], 1);
  cv::Mat img_gray;
  cv::cvtColor(img_in, img_gray, CV_RGB2GRAY);
  cv::Mat sobelX, sobelY, norm, dir;
  cv::Sobel(img_gray, sobelX, CV_32F, 1, 0);
  cv::Sobel(img_gray, sobelY, CV_32F, 0, 1);
  cv::cartToPolar(sobelX, sobelY, norm, dir);
  
  double sob_min, sob_max;
  cv::minMaxLoc(norm, &sob_min, &sob_max);
  
  cv::Mat img_sobel;
  norm.convertTo(img_sobel, CV_8U, -255.0 / sob_max, 255);

  cv::Mat gray_img, bin_img;
  //cv::cvtColor(img_sobel, gray_img, CV_BGR2GRAY);

  std::vector<std::vector<cv::Point> > contours;
  // 画像の二値化
  cv::threshold(img_sobel, bin_img, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);

  cv::findContours(bin_img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
  
  for(int i = 0; i < contours.size(); ++i) {
    size_t count = contours[i].size();
    if(count < 100 || count > 1000) continue; // （小さすぎる|大きすぎる）輪郭を除外

    cv::Mat pointsf;
    cv::Mat(contours[i]).convertTo(pointsf, CV_32F);
    // 楕円フィッティング
    cv::RotatedRect box = cv::fitEllipse(pointsf);
    // 楕円の描画
    cv::ellipse(img_in, box, cv::Scalar(0,0,255), 2, CV_AA);
  }

  cv::namedWindow("fit ellipse", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("fit ellipse", img_in);
  cv::imwrite("eye_circle.jpg", img_in);
  cv::waitKey(0);

  return 1;
}