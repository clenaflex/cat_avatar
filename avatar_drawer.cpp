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

void draw_outline(cv::Mat& avatar,cv::Scalar background_color){
	cv::Mat3b dotImg = avatar;
	for(int y = 1 ; y < avatar.rows-1; y++){
		for(int x = 1 ; x < avatar.cols-1; x++){
			Vec3b p = dotImg(cv::Point(x,y));
			Vec3b u = dotImg(cv::Point(x,y-1));
			Vec3b d = dotImg(cv::Point(x,y+1));
			Vec3b l = dotImg(cv::Point(x-1,y));
			Vec3b r = dotImg(cv::Point(x+1,y));

			if (p[0] !=  background_color[0]  && p[1] !=  background_color[1] && p[2] !=  background_color[2]){
				if(u[0] == background_color[0]  && u[1] == background_color[1] && u[2] == background_color[2]){
					dotImg(cv::Point(x,y)) = cv::Vec3b(0,0,0);
				}
				if(d[0] == background_color[0]  && d[1] == background_color[1] && d[2] == background_color[2]){
					dotImg(cv::Point(x,y)) = cv::Vec3b(0,0,0);
				}
				if(l[0] == background_color[0]  && l[1] == background_color[1] && l[2] == background_color[2]){
					dotImg(cv::Point(x,y)) = cv::Vec3b(0,0,0);
				}
				if(r[0] == background_color[0]  && r[1] == background_color[1] && r[2] == background_color[2]){
					dotImg(cv::Point(x,y)) = cv::Vec3b(0,0,0);
				}
			}
		}
	}	
}

void avatar_drawer(cv::Mat cat_face_img,cv::Point ear_r,int ear_r_wh,cv::Point ear_l,int ear_l_wh,cv::Point eye_r,int eye_r_wh,cv::Point eye_l,int eye_l_wh,cv::Point mouth,int mouth_wh,std::string savename){
	// cols:width
	// rows:height
	// rは向かって左、猫からしたら右
	cv::Mat dst_cat_face_img = cat_face_img.clone();
	cv::Scalar background_color = cv::Scalar(194,194,194);
	cv::Mat avatar(dst_cat_face_img.rows,dst_cat_face_img.cols,CV_8UC3,background_color);
	int p, q, a, b;
	a = dst_cat_face_img.cols*0.78/2;
	p = dst_cat_face_img.cols/2;
	if ( (ear_r.y+ear_r_wh)<(ear_l.y+ear_l_wh) ){
		b = (dst_cat_face_img.rows - (ear_r.y+ear_r_wh))*1.1/2;
	}else{
		b = (dst_cat_face_img.rows - (ear_l.y+ear_l_wh))*1.1/2;
	}
	q = dst_cat_face_img.rows - b - 10;


	//顔の色を先に取得、及びタイプ取得
	cv::Scalar face_color[3];
	int type = cat_classify(dst_cat_face_img,eye_r,eye_r_wh,eye_l,eye_l_wh,mouth,mouth_wh,face_color);
	if(type == 20 || type == 21){
		face_color[2] = cv::Scalar(255,255,255);
	}
	
	cv::Mat ear_r_img(dst_cat_face_img, cv::Rect(ear_r.x, ear_r.y, ear_r_wh, ear_r_wh));
	
	cv::Mat ear_l_img(dst_cat_face_img, cv::Rect(ear_l.x, ear_l.y, ear_l_wh, ear_l_wh));
	
	cv::Mat eye_r_img(dst_cat_face_img, cv::Rect(eye_r.x, eye_r.y, eye_r_wh, eye_r_wh));
	
	cv::Mat eye_l_img(dst_cat_face_img, cv::Rect(eye_l.x, eye_l.y, eye_l_wh, eye_l_wh));
	
	cv::Mat mouth_img(dst_cat_face_img, cv::Rect(mouth.x, mouth.y, mouth_wh, mouth_wh));
	
	cv::Scalar l_ear_color[1];
	cv::Scalar r_ear_color[1];
	get_ear_color(ear_l_img,l_ear_color);
	get_ear_color(ear_r_img,r_ear_color);

	int r_linea[4],r_lineb[4];
	int l_linea[4],l_lineb[4];
	int sc_flag_r,sc_flag_l;
	//sc_flag は3以上であれば耳or目の形状の検出が成功している
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
	} else{
		l_linea[0] = ear_l.x+ear_l_wh*0.8;
		l_linea[1] = ear_l.y+ear_l_wh*0.2;
		l_linea[2] = ear_l.x+ear_l_wh*0.1;
		l_linea[3] = ear_l.y+ear_l_wh*0.75;
		l_lineb[0] = ear_l.x+ear_l_wh*0.8;
		l_lineb[1] = ear_l.y+ear_l_wh*0.2;
		l_lineb[2] = ear_l.x+ear_l_wh*0.75;
		l_lineb[3] = ear_l.y+ear_l_wh;

		r_linea[0] = reverse_point_x(l_linea[0],img_width);
		r_linea[1] = l_linea[1];
		r_linea[2] = reverse_point_x(l_linea[2],img_width);
		r_linea[3] = l_linea[3];
		r_lineb[0] = reverse_point_x(l_lineb[0],img_width);
		r_lineb[1] = l_lineb[1];
		r_lineb[2] = reverse_point_x(l_lineb[2],img_width);
		r_lineb[3] = l_lineb[3];
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



    if(type == 10){
    	l_bgr[0] = face_color[0];
    	r_bgr[0] = face_color[0];
     }

    draw_ear_two_triangle(avatar,l_pt,l_bgr,r_pt,r_bgr);
    int m_y_param;
    m_y_param = (b/7);
    draw_face_ellipse(avatar,p,q-m_y_param,a,b,face_color,type);

	double r_eye[10],l_eye[10];
	sc_flag_r = detect_eye_ellipse(eye_r_img,r_eye);
	sc_flag_l = detect_eye_ellipse(eye_l_img,l_eye);

	if (sc_flag_r > 2 && sc_flag_l > 2 ){
		if(sc_flag_r ==  99 && sc_flag_l < 99 ){
		l_eye[5] = l_eye[0];
		l_eye[6] = l_eye[1];
		l_eye[7] = r_eye[7];
		l_eye[8] = r_eye[8];
		l_eye[9] = 180 - r_eye[9];	
		}else if(sc_flag_l ==  99 && sc_flag_r < 99){
		r_eye[5] = r_eye[0];
		r_eye[6] = r_eye[1];
		r_eye[7] = l_eye[7];
		r_eye[8] = l_eye[8];
		r_eye[9] = 180 - l_eye[9];
		}
	}else if(sc_flag_r > 2 && sc_flag_l < 3){
		l_eye[0] = reverse_point_x(eye_r.x + r_eye[0],img_width)-eye_l.x;
		l_eye[1] = r_eye[1] + eye_r.y - eye_l.y;
		l_eye[2] = r_eye[2];
		l_eye[3] = r_eye[3];
		l_eye[4] = 180 - r_eye[4];
		l_eye[5] = reverse_point_x(eye_r.x + r_eye[5],img_width)-eye_l.x;
		l_eye[6] = r_eye[6] + eye_r.y - eye_l.y;
		l_eye[7] = r_eye[7];
		l_eye[8] = r_eye[8];
		l_eye[9] = 180 - r_eye[9];	
	}else if(sc_flag_r < 3 && sc_flag_l > 2){
		r_eye[0] = reverse_point_x(eye_l.x + l_eye[0],img_width)-eye_r.x;
		r_eye[1] = l_eye[1] + eye_l.y - eye_r.y;
		r_eye[2] = l_eye[2];
		r_eye[3] = l_eye[3];
		r_eye[4] = 180 - l_eye[4];
		r_eye[5] = reverse_point_x(eye_l.x + l_eye[5],img_width)-eye_r.x;
		r_eye[6] = l_eye[6] + eye_l.y - eye_r.y;
		r_eye[7] = l_eye[7];
		r_eye[8] = l_eye[8];
		r_eye[9] = 180 - l_eye[9];
	}else{
		l_eye[0] = eye_l_wh/2;
		l_eye[1] = eye_l_wh/2;
		l_eye[2] = eye_l_wh*0.8/2;
		l_eye[3] = eye_l_wh*0.3/2;
		l_eye[4] = -10;

		l_eye[5] = l_eye[0];
		l_eye[6] = l_eye[1];
		l_eye[7] = which_min(l_eye[2],l_eye[3])*0.9;
		l_eye[8] = which_max(l_eye[2],l_eye[3])/2;
		l_eye[9] = 90;


		r_eye[0] = reverse_point_x(eye_l.x + l_eye[0],img_width)-eye_r.x;
		r_eye[1] = l_eye[1] + eye_l.y - eye_r.y;
		r_eye[2] = eye_l_wh*0.8/2;
		r_eye[3] = eye_l_wh*0.3/2;
		r_eye[4] = 10;

		r_eye[5] = r_eye[0];
		r_eye[6] = r_eye[1];
		r_eye[7] = l_eye[7];
		r_eye[8] = l_eye[8];
		r_eye[9] = 90;
	}

	cv::Scalar l_eye_color[2];
	cv::Scalar r_eye_color[2];
    get_eye_color(eye_l_img,cv::Point(l_eye[0],l_eye[1]),l_eye[2],l_eye[3],l_eye[4],l_eye_color);
    get_eye_color(eye_r_img,cv::Point(r_eye[0],r_eye[1]),r_eye[2],r_eye[3],r_eye[4],r_eye_color);

    cv::Scalar le_bgr[2];
    le_bgr[0] = l_eye_color[0];

    cv::Scalar re_bgr[2];
    re_bgr[0] = r_eye_color[0];

    if( (l_eye_color[1][0]+l_eye_color[1][1]+l_eye_color[1][2]) < (r_eye_color[1][0]+r_eye_color[1][1]+r_eye_color[1][2])){
    	le_bgr[1] = l_eye_color[1];
    	re_bgr[1] = l_eye_color[1];
    }else{
    	le_bgr[1] = r_eye_color[1];
    	re_bgr[1] = r_eye_color[1];
    }

	if (sc_flag_r > 2 && sc_flag_l > 2 ){
	}else if(sc_flag_r > 2 && sc_flag_l < 3){
		le_bgr[0] = re_bgr[0];
	}else if(sc_flag_r < 3 && sc_flag_l > 2){
		re_bgr[0] = le_bgr[0];
	}
    draw_eye_two_ellipse(avatar,cv::Point(eye_l.x+l_eye[0],eye_l.y+l_eye[1]-m_y_param),cv::Size(l_eye[2],l_eye[3]),l_eye[4],cv::Point(eye_l.x+l_eye[5],eye_l.y+l_eye[6]-m_y_param),cv::Size(l_eye[7],l_eye[8]),l_eye[9],le_bgr,cv::Point(eye_r.x+r_eye[0],eye_r.y+r_eye[1]-m_y_param),cv::Size(r_eye[2],r_eye[3]),r_eye[4],cv::Point(eye_r.x+r_eye[5],eye_r.y+r_eye[6]-m_y_param),cv::Size(r_eye[7],r_eye[8]),r_eye[9],re_bgr);
    
    cv::Scalar mouth_color[1];
    get_mouth_color(mouth_img,mouth_color);
    cv::Scalar c_m = mouth_color[0];
    int mouth_ed = p - (mouth_wh/2);
    draw_mouth(avatar,cv::Point(mouth_ed,mouth.y-m_y_param),mouth_wh,mouth_wh,c_m);
    draw_outline(avatar,background_color);
    savename +="_avatar.png";
    savename = "avatars_result/" + savename;
    cv::imwrite(savename,avatar);
}

int main(int argc, char** argv)
{
	cv::Mat src_img = cv::imread(argv[1], 1);
	cv::Mat dst_img = src_img.clone();
	if(!src_img.data) return -1;
	std::string filename = argv[1];
	int place = filename.find_last_of("/");
	int result[4];
	int result_ear[2][4];
	int result_eye[2][4];
	int result_mouth[4];
	std::string sub = filename.substr(place+1, filename.size()-place-5);

	detect_cat_face(dst_img,result);
	cv::Mat roi_img(dst_img, cv::Rect(result[0],result[1],result[2],result[3]));
	detect_cat_ears(roi_img,result_ear);
	detect_cat_eyes(roi_img,result_eye);
	detect_cat_mouth(roi_img,result_mouth);
	
	// //4.1と4.2の実験時にはこの部分のコメントアウトを外す。実験用コードここから

	// //4.1と4.2の実験共用ここから
	// std::string sub_cp = sub;
	// cv::Mat roi_img_cp = roi_img;
	// cv::Scalar face_color[3];
	// //4.1と4.2の実験共用ここまで

	// //4.1の実験用コードここから
	// rectangle(roi_img_cp, cv::Rect(result_ear[0][0],result_ear[0][1],result_ear[0][2],result_ear[0][3]), Scalar(0,0,255), 8);
	// rectangle(roi_img_cp, cv::Rect(result_ear[1][0],result_ear[1][1],result_ear[1][2],result_ear[1][3]), Scalar(0,0,255), 8);
	// rectangle(roi_img_cp, cv::Rect(result_eye[0][0],result_eye[0][1],result_eye[0][2],result_eye[0][3]), Scalar(255,0,0), 8);
	// rectangle(roi_img_cp, cv::Rect(result_eye[1][0],result_eye[1][1],result_eye[1][2],result_eye[1][3]), Scalar(255,0,0), 8);
	// rectangle(roi_img_cp, cv::Rect(result_mouth[0],result_mouth[1],result_mouth[2],result_mouth[3]), Scalar(0,255,0), 8);
	// sub_cp = "detection_result/"+sub_cp + "_result" + ".jpg";
	// cout << "cp ";
	// cout << sub_cp ;
	// cout << " images/" << endl;
	// cv::imwrite(sub_cp, roi_img_cp);

	// //4.1の実験用コードここまで

	// //4.2の実験用コードここから
	
	// // int type = cat_classify(roi_img_cp,cv::Point(result_eye[0][0],result_eye[0][1]),result_eye[0][2],cv::Point(result_eye[1][0],result_eye[1][1]),result_eye[1][2],cv::Point(result_mouth[0],result_mouth[1]),result_mouth[2],face_color);
	
	// // cout << sub_cp << ".jpg";
	// // if (type == 10){
	// // 	cout << " type:" << "単色" << endl;
	// // }else if(type == 11){
	// // 	cout << " type:" << "単色＋縞" << endl;
	// // }else if(type == 20){
	// // 	cout << " type:" << "ハチワレ" << endl;
	// // }else if(type == 21){
	// // 	cout << " type:" << "ハチワレ＋縞" << endl;
	// // }else if(type == 30 || type ==31){
	// // 	cout << " type:" << "ポインテッド" << endl;
	// // }
	// //4.2の実験用コードここまで

	// //実験用コードここまで


	//実験時にはアバター生成用のコードをコメントアウトする。ここから

	avatar_drawer(roi_img,cv::Point(result_ear[0][0],result_ear[0][1]),result_ear[0][2]-10,cv::Point(result_ear[1][0],result_ear[1][1]),result_ear[1][2]-10,cv::Point(result_eye[0][0],result_eye[0][1]),result_eye[0][2],cv::Point(result_eye[1][0],result_eye[1][1]),result_eye[1][2],cv::Point(result_mouth[0],result_mouth[1]),result_mouth[2],sub);

	//アバター生成用のコードここまで

	return 0;
}