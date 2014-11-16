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

int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}

void cat_avatar_face(Mat dst_img, int face_x, int face_y,int face_width,int face_height,int ear_x, int ear_y,int ear_height,int face_ellipse[4]){
	double angle;
	angle = 0;
	/// 画像，中心座標，（長径・短径），回転角度，円弧開始角度，円弧終了角度，色，線太さ，連結
	// Red，太さ3，4近傍連結
	int face_ellipse_height =  face_height* 9/10 - ear_height* 3/4;
	int face_ellipse_width = face_width * 9/10;
	int ellipse_y = face_y + ear_height * 3/4 + face_ellipse_height / 2;
	int ellipse_x = face_x + face_width / 2;
	face_ellipse[0] = ellipse_x;
	face_ellipse[1] = ellipse_y;
	face_ellipse[2] = face_ellipse_width/2;
	face_ellipse[3] = face_ellipse_height/2;
	cv::ellipse(dst_img, cv::Point(ellipse_x, ellipse_y), cv::Size(face_ellipse_width/2,face_ellipse_height/2), angle, angle, angle + 360, cv::Scalar(0,0,200), 3, 4);
}

void cat_avatar_ear_line(Mat dst_img,int cross_x,int cross_x, int to_point_x,int face_ellipse[4]){
	double to_point_y;
	to_point_y = ellipse_point_calc(face_ellipse[2],face_ellipse[3],face_ellipse[0],face_ellipse[1],to_point_x)
	cv::line(dst_img, cv::Point(cross_x, cross_x), cv::Point(to_point_x,to_point_y), cv::Scalar(0,0,255), 2, CV_AA);
}

double ellipse_point_calc(int a, int b, int p ,int q ,int x){
		double temp;
		int temp_cast;
		temp = (a*a*b*b) - b*b*(x-p)^2;
		temp = sqrt(temp) / a;
		temp_cast = (q - temp)*10;
		temp = temp_cast / 0.1;
		return temp;
}

void cut_ellipse_img(Mat dst_img,face_ellipse[4]){
	Mat cp_img = dst_img.clone();
	double angle;
	angle = 0;
	int dst_width = dst_img.size().width;
	int dst_height = dst_img.size().height;
	Mat cut_img = Mat::zeros(Size(dst_width,dst_height), CV_8UC3);
	cv::ellipse(cut_img, cv::Point(face_ellipse[0], face_ellipse[1]), cv::Size(face_ellipse[2],face_ellipse[3]), angle, angle, angle + 360, cv::Scalar(0,0,200),-1, CV_AA);

	for(int y = 0 ; y < cut_img.rows; y++){
		for(int x = 0 ; x < cut_img.cols; x++){
			Vec3b &p = cut_img.at<uchar>( y, x );
			if (p[0] == 0 && p[1] == 0 && p[2] == 0){
				Vec3b &q = cp_img.at<uchar>( y, x );
				q[0] = 255;
				q[1] = 255;
				q[2] = 255;
			}
		}
	}
	dst_img = cp_img.clone();
}