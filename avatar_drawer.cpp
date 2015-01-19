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
#include "prototype.h"

using namespace std;
using namespace cv;

int reverse_point_x(int point_x,int width){
	int reverse_point;
	if(point_x > width/2){
		reverse_point = point_x - 2*(point_x - width/2);
	}else if (point_x < width/2){
		reverse_point = point_x + 2*(width/2 - point_x);
	}else {
		reverse_point = point_x;
	}
	return reverse_point;
}

void avatar_drawer(cv::Mat cat_face_img,cv::Point ear_r,int ear_r_wh,cv::Point ear_l,int ear_l_wh,cv::Point eye_r,int eye_r_wh,cv::Point eye_l,int eye_l_wh,cv::Point mouth,int mouth_wh){
	// cols:width
	// rows:height
	// rは向かって左、猫からしたら右
	cv::Mat dst_cat_face_img = cat_face_img.clone();
	cv::Mat avatar(dst_cat_face_img.rows,dst_cat_face_img.cols,CV_8UC3,cv::Scalar(194,194,194));
	int p, q, a, b;
	a = dst_cat_face_img.cols*0.9/2;
	p = dst_cat_face_img.cols/2;
	if ( (ear_r.y+ear_r_wh)<(ear_l.y+ear_l_wh) ){
		b = (dst_cat_face_img.rows - (ear_r.y+ear_r_wh))*1.1/2;
	}else{
		b = (dst_cat_face_img.rows - (ear_l.y+ear_l_wh))/2;
	}
	q = dst_cat_face_img.rows - b - 5;


	//顔の色を先に取得、及びタイプ取得
	cv::Scalar face_color[3];
	int type = cat_classify(dst_cat_face_img,eye_r,eye_r_wh,eye_l,eye_l_wh,mouth,mouth_wh,face_color);


  // (x,y)=(200,200), (width,height)=(100,100)
  // cv::Mat roi_img(dst_cat_face_img, cv::Rect(200, 200, 100, 100));
	cv::Mat ear_r_img(dst_cat_face_img, cv::Rect(ear_r.x, ear_r.y, ear_r_wh, ear_r_wh));
	cv::imwrite("ear_r.png",ear_r_img);
	cv::Mat ear_l_img(dst_cat_face_img, cv::Rect(ear_l.x, ear_l.y, ear_l_wh, ear_l_wh));
	cv::imwrite("ear_l.png",ear_l_img);
	cv::Mat eye_r_img(dst_cat_face_img, cv::Rect(eye_r.x, eye_r.y, eye_r_wh, eye_r_wh));
	cv::imwrite("eye_r.png",eye_r_img);
	cv::Mat eye_l_img(dst_cat_face_img, cv::Rect(eye_l.x, eye_l.y, eye_l_wh, eye_l_wh));
	cv::imwrite("eye_l.png",eye_l_img);
	cv::Mat mouth_img(dst_cat_face_img, cv::Rect(mouth.x, mouth.y, mouth_wh, mouth_wh));
	cv::imwrite("mouth.png",mouth_img);

	cv::Scalar l_ear_color[1];
	cv::Scalar r_ear_color[1];
	get_ear_color(ear_l_img,l_ear_color);
	get_ear_color(ear_r_img,r_ear_color);

	int r_linea[4],r_lineb[4];
	int l_linea[4],l_lineb[4];
	int sc_flag_r,sc_flag_l;
	//sc_flag は3以上であれば耳の形状の検出が成功している
	sc_flag_r = detect_ear_line(ear_r_img,r_linea,r_lineb);
	sc_flag_l =detect_ear_line(ear_l_img,l_linea,l_lineb);

	int img_width = dst_cat_face_img.cols;
	if (sc_flag_r > 2 && sc_flag_l > 2 ){
		l_linea[0] = ear_l.x+l_linea[0];
		l_linea[1] = ear_l.y+l_linea[1];
		l_linea[2] = ear_l.x+l_linea[2];
		l_linea[3] = ear_l.y+l_linea[3];
		l_lineb[0] = ear_l.x+l_lineb[0];
		l_lineb[1] = ear_l.y+l_lineb[1];
		l_lineb[2] = ear_l.x+l_lineb[2];
		l_lineb[3] = ear_l.y+l_lineb[3];

		r_linea[0] = ear_r.x+r_linea[0];
		r_linea[1] = ear_r.y+r_linea[1];
		r_linea[2] = ear_r.x+r_linea[2];
		r_linea[3] = ear_r.y+r_linea[3];
		r_lineb[0] = ear_r.x+r_lineb[0];
		r_lineb[1] = ear_r.y+r_lineb[1];
		r_lineb[2] = ear_r.x+r_lineb[2];
		r_lineb[3] = ear_r.y+r_lineb[3];
		cout << "Both ear detect success" << endl;
	} else if(sc_flag_r > 2 && sc_flag_l < 3){
		r_linea[0] = ear_r.x+r_linea[0];
		r_linea[1] = ear_r.y+r_linea[1];
		r_linea[2] = ear_r.x+r_linea[2];
		r_linea[3] = ear_r.y+r_linea[3];
		r_lineb[0] = ear_r.x+r_lineb[0];
		r_lineb[1] = ear_r.y+r_lineb[1];
		r_lineb[2] = ear_r.x+r_lineb[2];
		r_lineb[3] = ear_r.y+r_lineb[3];

		l_linea[0] = reverse_point_x(r_linea[0],img_width);
		l_linea[1] = r_linea[1];
		l_linea[2] = reverse_point_x(r_linea[2],img_width);
		l_linea[3] = r_linea[3];
		l_lineb[0] = reverse_point_x(r_lineb[0],img_width);
		l_lineb[1] = r_lineb[1];
		l_lineb[2] = reverse_point_x(r_lineb[2],img_width);
		l_lineb[3] = r_lineb[3];
		cout << "Only r detect success" << endl;
	} else if(sc_flag_r < 3 && sc_flag_l > 2){
		l_linea[0] = ear_l.x+l_linea[0];
		l_linea[1] = ear_l.y+l_linea[1];
		l_linea[2] = ear_l.x+l_linea[2];
		l_linea[3] = ear_l.y+l_linea[3];
		l_lineb[0] = ear_l.x+l_lineb[0];
		l_lineb[1] = ear_l.y+l_lineb[1];
		l_lineb[2] = ear_l.x+l_lineb[2];
		l_lineb[3] = ear_l.y+l_lineb[3];

		r_linea[0] = reverse_point_x(l_linea[0],img_width);
		r_linea[1] = l_linea[1];
		r_linea[2] = reverse_point_x(l_linea[2],img_width);
		r_linea[3] = l_linea[3];
		r_lineb[0] = reverse_point_x(l_lineb[0],img_width);
		r_lineb[1] = l_lineb[1];
		r_lineb[2] = reverse_point_x(l_lineb[2],img_width);
		r_lineb[3] = l_lineb[3];
		cout << "Only l detect success" << endl;
	} 

	double ans[12];
	ellipse_ear_calc(p,q,a,b,l_linea[0],l_linea[1],l_linea[2],l_linea[3],l_lineb[0],l_lineb[1],l_lineb[2],l_lineb[3],r_linea[0],r_linea[1],r_linea[2],r_linea[3],r_lineb[0],r_lineb[1],r_lineb[2],r_lineb[3],ans);

	cv::Point l_pt[3];
	cv::Point r_pt[3];
	l_pt[0] = cv::Point(ans[0],ans[1]);
	l_pt[1] = cv::Point(ans[2],ans[3]);
	l_pt[2] = cv::Point(ans[4],ans[5]);

	r_pt[0] = cv::Point(ans[6],ans[7]);
	r_pt[1] = cv::Point(ans[8],ans[9]);
	r_pt[2] = cv::Point(ans[10],ans[11]);

    cv::Scalar l_bgr[2];
    l_bgr[0] = face_color[1];
    l_bgr[1] = l_ear_color[0];


    cv::Scalar r_bgr[2];
    r_bgr[0] = face_color[0];
    r_bgr[1] = r_ear_color[0];

    draw_ear_two_triangle(avatar,l_pt,l_bgr,r_pt,r_bgr);
    draw_face_ellipse(avatar,p,q,a,b,face_color,type);

	double r_eye[5],l_eye[5];
	detect_eye_ellipse(eye_r_img,r_eye);
	detect_eye_ellipse(eye_l_img,l_eye);

	cv::Scalar l_eye_color[1];
	cv::Scalar r_eye_color[1];
    get_eye_color(eye_l_img,cv::Point(l_eye[0],l_eye[1]),l_eye[2],l_eye[3],l_eye[4],l_eye_color);
    get_eye_color(eye_r_img,cv::Point(r_eye[0],r_eye[1]),r_eye[2],r_eye[3],r_eye[4],r_eye_color);

    cv::Scalar le_bgr[2];
    le_bgr[0] = l_eye_color[0];
    le_bgr[1] = cv::Scalar(0,0,0);

    cv::Scalar re_bgr[2];
    re_bgr[0] = r_eye_color[0];
    re_bgr[1] = cv::Scalar(0,0,0);

    draw_eye_two_ellipse(avatar,cv::Point(eye_l.x+l_eye[0],eye_l.y+l_eye[1]),cv::Size(l_eye[2],l_eye[3]),l_eye[4],cv::Point(eye_l.x+l_eye[0],eye_l.y+l_eye[1]),cv::Size(which_min(l_eye[2],l_eye[3])*0.9,which_max(l_eye[2],l_eye[3])/4),90,le_bgr,cv::Point(eye_r.x+r_eye[0],eye_r.y+r_eye[1]),cv::Size(r_eye[2],r_eye[3]),r_eye[4],cv::Point(eye_r.x+r_eye[0],eye_r.y+r_eye[1]),cv::Size(which_min(r_eye[2],r_eye[3])*0.9,which_max(r_eye[2],r_eye[3])/4),90,re_bgr);
    
    cv::Scalar mouth_color[1];
    get_mouth_color(mouth_img,mouth_color);
    cv::Scalar c_m = mouth_color[0];
    draw_mouth(avatar,mouth,mouth_wh,mouth_wh,c_m);

    cv::imwrite("avatar.png",avatar);
    cv::namedWindow("avatar", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
    cv::imshow("avatar", avatar);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
	cv::Mat src_img = cv::imread(argv[1], 1);
	if(!src_img.data) return -1;
	cv::Mat cat_face_img = src_img.clone();
	avatar_drawer(cat_face_img,cv::Point(atoi(argv[2]),atoi(argv[3])),atoi(argv[4]),cv::Point(atoi(argv[5]),atoi(argv[6])),atoi(argv[7]),cv::Point(atoi(argv[8]),atoi(argv[9])),atoi(argv[10]),cv::Point(atoi(argv[11]),atoi(argv[12])),atoi(argv[13]),cv::Point(atoi(argv[14]),atoi(argv[15])),atoi(argv[16]));
	return 0;
}