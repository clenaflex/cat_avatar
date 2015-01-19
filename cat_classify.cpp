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

using namespace cv;
using namespace std;

void get_color(cv::Mat img,cv:: Point img_point,cv::Scalar color[1]){
	cv::Mat dst_img = img.clone();
	cv::Mat3b dotImg = dst_img;
	Vec3b p = dotImg(img_point);
	color[0] = cv::Scalar(p[0],p[1],p[2]);
}


void rgb_gray(cv::Mat& img,int threshold){
	cv::Mat3b dotImg = img;
	for(int y = 0 ; y < img.rows; y++){
		for(int x = 0 ; x < img.cols; x++){
			Vec3b p = dotImg(cv::Point(x,y));
			if ( (p[0]+p[1]+p[2])/3 > threshold){
				dotImg(cv::Point(x,y)) = cv::Vec3b(255,255,255);
			}else{
				dotImg(cv::Point(x,y)) = cv::Vec3b(0,0,0);
			}
		}
	}
}

void k_means(cv::Mat& src_img,int cluster_count){
  #define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
  #define OPENCV_VERSION_CODE OPENCV_VERSION(CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION)
    // 画像を1列の行列に変形
  cv::Mat points;
  src_img.convertTo(points, CV_32FC3);
  points = points.reshape(3, src_img.rows*src_img.cols);

  // RGB空間でk-meansを実行
  cv::Mat_<int> clusters(points.size(), CV_32SC1);
  cv::Mat centers;
  // クラスタ対象，クラスタ数，（出力）クラスタインデックス，
  // 停止基準，k-meansの実行回数，手法，（出力）クラスタ中心値
  #if OPENCV_VERSION_CODE<OPENCV_VERSION(2,3,0)
  cv::kmeans(points, cluster_count, clusters,cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, &centers);
  #else
  cv::kmeans(points, cluster_count, clusters, cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);
  #endif

  // すべてのピクセル値をクラスタ中心値で置き換え
  cv::Mat dst_img(src_img.size(), src_img.type());
  cv::MatIterator_<cv::Vec3b> itd = dst_img.begin<cv::Vec3b>(), itd_end = dst_img.end<cv::Vec3b>();
  for(int i=0; itd != itd_end; ++itd, ++i) {
    cv::Vec3f &color = centers.at<cv::Vec3f>(clusters(i), 0);
    (*itd)[0] = cv::saturate_cast<uchar>(color[0]);
    (*itd)[1] = cv::saturate_cast<uchar>(color[1]);
    (*itd)[2] = cv::saturate_cast<uchar>(color[2]);
  }
  cv::medianBlur(dst_img, dst_img, 11);
  src_img = dst_img.clone();
}

int cat_classify(cv::Mat& img,cv::Point eye_r,int eye_r_wh,cv::Point eye_l,int eye_l_wh,cv::Point mouth,int mouth_wh,cv::Scalar face_color[3]){
	int type;
	/*
	all:10
	all+shima:11
	hachiware:20
	hachi+shima:21
	hachi+shima+mike:22
	pointed:30
	*/
	cv::Mat org_img = img.clone();
	cv::Mat low_color_img = img.clone();
	cv::Mat gray_img = img.clone();
	rgb_gray(gray_img,100);
	k_means(low_color_img,2);
	cv::Mat3b orgdotImg = org_img;
	cout << org_img.rows << " " << org_img.cols << endl;
 	cv::Mat3b lowdotImg = low_color_img;
	cv::Mat3b graydotImg = gray_img;

	cv::Point point_1 = cv::Point(eye_r.x+eye_r_wh/2,eye_r.y-eye_r_wh/2);
	cv::Point point_2 = cv::Point(eye_l.x+eye_l_wh/2,eye_l.y-eye_l_wh/2);
	cv::Point point_3 = cv::Point((eye_r.x+eye_r_wh+eye_l.x+eye_l_wh)/2,(eye_r.y+eye_r_wh+eye_l.y+eye_l_wh)/2);
	cv::Point point_4 = cv::Point((eye_r.x+eye_r_wh/2+mouth.x)/2,mouth.y+mouth_wh/2);
	cv::Point point_5 = cv::Point((eye_l.x+eye_l_wh/2+mouth.x+mouth_wh)/2,mouth.y+mouth_wh/2);
	
	cv::Vec3b orgp1 = orgdotImg(point_1);	
	cv::Vec3b orgp2 = orgdotImg(point_2);
	cv::Vec3b orgp3 = orgdotImg(point_3);

	cv::Vec3b lp1 = lowdotImg(point_1);
	cv::Vec3b lp2 = lowdotImg(point_2);
	cv::Vec3b lp3 = lowdotImg(point_3);
	cv::Vec3b lp4 = lowdotImg(point_4);
	cv::Vec3b lp5 = lowdotImg(point_5);
	
	cv::Vec3b gp3 = graydotImg(point_3);
	cv::Vec3b gp4 = graydotImg(point_4);
	cv::Vec3b gp5 = graydotImg(point_5);

	if(lp1[0] == lp2[0] && lp2[0] == lp3[0] && lp3[0] == lp4[0] && lp4[0] == lp5[0]){
		if(lp1[1] == lp2[1] && lp2[1] == lp3[1] && lp3[1] == lp4[1] && lp4[1] == lp5[1]){
			if(lp1[2] == lp2[2] && lp2[2] == lp3[2] && lp3[2] == lp4[2] && lp4[2] == lp5[2]){
				type = 10;
			}
		}
	}else if(gp3[0] >250 && gp4[0] >250  && gp5[0] >250 && gp3[0] == gp4[0] && gp4[0] == gp5[0]){
		type = 20;
	}else{
		type = 30;
	}
	face_color[0] = cv::Scalar(orgp1[0],orgp1[1],orgp1[2]);
	face_color[1] = cv::Scalar(orgp2[0],orgp2[1],orgp2[2]);
	face_color[2] = cv::Scalar(orgp3[0],orgp3[1],orgp3[2]);
	return type;
}

void get_eye_color(cv::Mat eye_img,cv::Point ellipse_pq,int a ,int b,int theta,cv::Scalar eye_color[1]){
	int r;
	cv::Mat dst_img = eye_img.clone();
	if (a>b){
		r = a*0.85/2;
	}else{
		r = b*0.85/2;
	}
	get_color(dst_img,cv::Point(ellipse_pq.x + r*cos(theta*180/3.14),ellipse_pq.y + r*sin(theta*180/3.14)),eye_color);
}
void get_ear_color(cv::Mat ear_img,cv::Scalar ear_color[1]){
	cv::Mat dst_img = ear_img.clone();
	get_color(dst_img,cv::Point(dst_img.cols/2,dst_img.rows/2),ear_color);
}

void get_mouth_color(cv::Mat mouth_img,cv::Scalar mouth_color[1]){
	cv::Mat dst_img = mouth_img.clone();
	get_color(dst_img,cv::Point(dst_img.cols/2,dst_img.rows*2/5),mouth_color);
}

// int main(int argc, char** argv)
// {
// 	cv::Mat img_in = cv::imread(argv[1], 1);
// 	cv::Mat img = img_in.clone();
// 	cout << img.rows << " " << img.cols << endl;
// 	cv::Scalar face_color[3];
// 	int result = cat_classify(img,cv::Point(atoi(argv[2]),atoi(argv[3])),atoi(argv[4]),cv::Point(atoi(argv[5]),atoi(argv[6])),atoi(argv[7]),cv::Point(atoi(argv[8]),atoi(argv[9])),atoi(argv[10]),face_color);
// 	// cout << face_color[0][0] << endl;:
// 	cout << result << endl;
// 	return 0;
// }