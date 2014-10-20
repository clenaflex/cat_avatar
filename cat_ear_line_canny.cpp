#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char** argv) {
  cv::Mat src_img = cv::imread(argv[1], 1);
  if(!src_img.data) return -1;

  cv::Mat dst_img, work_img;
  dst_img = src_img.clone();
  cv::cvtColor(src_img, work_img, CV_BGR2GRAY);
  cv::Canny(work_img, work_img, 50, 200, 3);
  cv::imwrite("img_canny.jpg", work_img);
  // 確率的Hough変換
  std::vector<cv::Vec4i> lines;
  // 入力画像，出力，距離分解能，角度分解能，閾値，線分の最小長さ，
  // 2点が同一線分上にあると見なす場合に許容される最大距離
  cv::HoughLinesP(work_img, lines, 1, CV_PI/180, 50, 50, 5);

  std::vector<cv::Vec4i>::iterator it = lines.begin();
  for(; it!=lines.end(); ++it) {
    cv::Vec4i l = *it;
    cv::line(dst_img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 2, CV_AA);
    cout << "("<< l[0] << "," << l[1] << ")" << " to " << "("<< l[2] << "," << l[3] << ")" << endl;
  }

  cv::namedWindow("HoughLinesP", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
  cv::imshow("HoughLinesP", dst_img);
  cv::imwrite("cat_ear_lines_canny.jpg", dst_img);
  cv::waitKey(0);
}