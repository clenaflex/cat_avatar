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

int shima_detect(cv::Mat img){
  cv::Mat dst_img = img.clone();
  string cascade_file = string("./cascades/cascade_shima_hog.xml");
  cv::CascadeClassifier cascade;
  cascade.load(cascade_file);
  // cols:width
  // rows:height
  int img_width,img_height;
  img_width = dst_img.cols;
  img_height = dst_img.rows;
  if(cascade.empty()) {
    cerr << "cannot load cascade file" << endl;
    exit(-1);
  }
  vector<cv::Rect> objects;
  cascade.detectMultiScale(dst_img, objects, 1.1, 4,0,cv::Size(img_width/5,img_height/5));
  vector<cv::Rect>::const_iterator iter = objects.begin();
  int shima_count=0;
  while(iter!=objects.end()) {
    if(iter->y <img_height*2/3 && 0 < iter->y){
      if(iter->x > (img_width/3) && iter->x<(img_width*2/3)){
        shima_count++;
        cv::rectangle(dst_img,cv::Rect(iter->x, iter->y, iter->width, iter->height),cv::Scalar(0, 0, 255), 2);
      }
    }
    ++iter;
  }
  return shima_count;
}


int max_3(int a,int b,int c){
	if(a>b){
		if(a>c){
			return 0;
		}else{
			return 2;
		}

	}else if(b>c){
		if(b>a){
			return 1;
		}else{
			return 0;
		}
	}else if (c>a){
		if(c>b){
			return 2;
		}else{
			return 1;
		}
	}else{
		return 3;
	} 
}

int min_3(int a,int b,int c){
	if(a>b){
		if(c>b){
			return 1;
		}else{
			return 2;
		}

	}else if(b>c){
		if(a>c){
			return 2;
		}else{
			return 0;
		}
	}else if (c>a){
		if(b>a){
			return 0;
		}else{
			return 1;
		}
	}else{
		return 3;
	}
}


void hsv_to_rgb(int hsv[3],int rgb[3]){
	int max = hsv[2];
	int min = max-(max*(hsv[1]/255.0));
	if( -1 <hsv[0] && hsv[0] <61){
		rgb[0] = max;
		rgb[1] = (hsv[0]/60.0)*(max-min)+min;
		rgb[2] = min;
	}else if(60<hsv[0] && hsv[0]<121){
		rgb[0] = ((120 - hsv[0])/60.0)*(max-min)+min;
		rgb[1] = max;
		rgb[2] = min;
	}else if(120<hsv[0] && hsv[0]<181){
		rgb[0] = min;
		rgb[1] = max;
		rgb[2] = ((hsv[0]-120)/60.0 )*(max-min)+min;
	}else if(180<hsv[0] && hsv[0]<241){
		rgb[0] = min;
		rgb[1] = ((240-hsv[0])/60.0)*(max-min)+min;
		rgb[2] = max;
	}else if(240<hsv[0] && hsv[0]<301){
		rgb[0] = ((hsv[0]-240)/60.0)*(max-min)+min;
		rgb[1] = min;
		rgb[2] = max;
	}else if(300<hsv[0] && hsv[0]<361){
		rgb[0] = max;
		rgb[1] = min;
		rgb[2] = ((360-hsv[0])/60.0)*(max-min)+min; 
	}
}

void rgb_to_hsv(int rgb[3],int hsv[3]){
	int max_s = max_3(rgb[0],rgb[1],rgb[2]);
	int min_s = min_3(rgb[0],rgb[1],rgb[2]);
	if (max_s == 0){
		hsv[0] = 60 * (rgb[1]-rgb[2])/ ((rgb[max_s]-rgb[min_s])*0.1*10) ;
	}else if (max_s == 1){
		hsv[0] = 120 + 60 * (rgb[2]-rgb[0])/( (rgb[max_s]-rgb[min_s])*0.1*10) ;
	}else if (max_s == 2){
		hsv[0] = 240 + 60 * (rgb[0]-rgb[1])/((rgb[max_s]-rgb[min_s])*0.1*10);
	}else{
		hsv[0] = 0;
	}
	if(hsv[0] < 0){
		hsv[0] +=360;
	}
	//S,V 's ranges are 0~255
	hsv[1] = 255*(rgb[max_s]-rgb[min_s])/(rgb[max_s]*0.1*10);
	hsv[2] = rgb[max_s];
}


void get_color(cv::Mat img,cv:: Point img_point,cv::Scalar color[1]){
	cv::Mat dst_img = img.clone();
	cv::Mat3b dotImg = dst_img;
	Vec3b p = dotImg(img_point);
	color[0] = cv::Scalar(p[0],p[1],p[2]);
}

void get_color_resize(cv::Mat img,cv::Scalar color[1]){
	cv::Mat src_img = img.clone(); 
	cv::Mat dst_img2(1,1, src_img.type());
	// color[1];
	cv::resize(src_img, dst_img2, dst_img2.size(), cv::INTER_CUBIC);
	get_color(dst_img2,cv::Point(0,0),color);
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
  cv::Mat points;
  src_img.convertTo(points, CV_32FC3);
  points = points.reshape(3, src_img.rows*src_img.cols);
  cv::Mat_<int> clusters(points.size(), CV_32SC1);
  cv::Mat centers;
  #if OPENCV_VERSION_CODE<OPENCV_VERSION(2,3,0)
  cv::kmeans(points, cluster_count, clusters,cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, &centers);
  #else
  cv::kmeans(points, cluster_count, clusters, cvTermCriteria(CV_TERMCRIT_EPS|CV_TERMCRIT_ITER, 10, 1.0), 1, cv::KMEANS_PP_CENTERS, centers);
  #endif
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
	pointed:30
	*/
	cv::Mat org_img = img.clone();
	cv::Mat low_color_img = img.clone();
	cv::Mat gray_img = img.clone();
	rgb_gray(gray_img,150);
	k_means(low_color_img,2);
	cv::Mat3b orgdotImg = org_img;
 	cv::Mat3b lowdotImg = low_color_img;
	cv::Mat3b graydotImg = gray_img;

	cv::Point point_1 = cv::Point(eye_r.x+eye_r_wh/2,eye_r.y-eye_r_wh/2);
	cv::Point point_2 = cv::Point(eye_l.x+eye_l_wh/2-12,eye_l.y-eye_l_wh/2);
	cv::Point point_3 = cv::Point((eye_r.x+eye_r_wh+eye_l.x+eye_l_wh)/2,(eye_r.y+eye_r_wh+eye_l.y+eye_l_wh)/2);
	cv::Point point_4 = cv::Point((eye_r.x+eye_r_wh/2+mouth.x)/2,mouth.y+mouth_wh/4);
	cv::Point point_5 = cv::Point((eye_l.x+eye_l_wh/2+mouth.x+mouth_wh)/2,mouth.y+mouth_wh/4);
	cv::Mat area1(org_img, cv::Rect(eye_r.x,eye_r.y-eye_r_wh,eye_r_wh,eye_r_wh));
	cv::Mat area2(org_img, cv::Rect(eye_l.x,eye_l.y-eye_l_wh,eye_l_wh,eye_l_wh));
	cv::Mat area3(org_img, cv::Rect(eye_r.x+eye_r_wh,eye_r.y,eye_r_wh,eye_r_wh));
	
	cv::Vec3b lp1 = lowdotImg(point_1);
	cv::Vec3b lp2 = lowdotImg(point_2);
	cv::Vec3b lp3 = lowdotImg(point_3);
	cv::Vec3b lp4 = lowdotImg(point_4);
	cv::Vec3b lp5 = lowdotImg(point_5);
	
	cv::Vec3b gp3 = graydotImg(point_3);
	cv::Vec3b gp4 = graydotImg(point_4);
	cv::Vec3b gp5 = graydotImg(point_5);
	int s_type  = shima_detect(org_img);


	if(lp1[0] == lp2[0] && lp2[0] == lp3[0] && lp3[0] == lp4[0] && lp4[0] == lp5[0]){
		if(lp1[1] == lp2[1] && lp2[1] == lp3[1] && lp3[1] == lp4[1] && lp4[1] == lp5[1]){
			if(lp1[2] == lp2[2] && lp2[2] == lp3[2] && lp3[2] == lp4[2] && lp4[2] == lp5[2]){
				type = 10;
				if(s_type > 0){
					type =11;
				}
			}
		}
	}else if(gp3[0] >250 && gp4[0] >250  && gp5[0] >250){
		type = 20;
		if(s_type > 0){
					type =21;
				}
	}else{
		type = 30;
	if(s_type > 0){
		type =31;
				}
	}
	cv::Scalar color[1];
	get_color_resize(area1,color);
	face_color[0] = color[0];
	get_color_resize(area2,color);
	face_color[1] = color[0];
	get_color_resize(area3,color);
	face_color[2] = color[0];

	return type;
}

void get_eye_color(cv::Mat eye_img,cv::Point ellipse_pq,int a ,int b,int theta,cv::Scalar eye_color[2]){
	int r;
	int param = 3;
	cv::Mat dst_img = eye_img.clone();
	if (a>b){
		r = a*0.65;
	}else{
		r = b*0.65;
		theta = -1*(theta-90);
	}
	cv::Scalar center_color[1];
	cv::Scalar out_color[1];
	theta = 0;
	get_color(dst_img,cv::Point(ellipse_pq.x+param + r*cos(theta*180/3.14),ellipse_pq.y + r*sin(theta*180/3.14)),out_color);
	get_color(dst_img,cv::Point(ellipse_pq.x+param,ellipse_pq.y),center_color);
	eye_color[0][0] = out_color[0][0];
	eye_color[0][1] = out_color[0][1];
	eye_color[0][2] = out_color[0][2];
	eye_color[1][0] = center_color[0][0];
	eye_color[1][1] = center_color[0][1];
	eye_color[1][2] = center_color[0][2];
	int rgb[3],hsv[3];
	rgb[0]=eye_color[0][2];
	rgb[1]=eye_color[0][1];
	rgb[2]=eye_color[0][0];
	rgb_to_hsv(rgb,hsv);
	hsv[2] +=90;
	hsv_to_rgb(hsv,rgb);
	eye_color[0][2]=rgb[0];
	eye_color[0][1]=rgb[1];
	eye_color[0][0]=rgb[2];

	rgb[0]=eye_color[1][2];
	rgb[1]=eye_color[1][1];
	rgb[2]=eye_color[1][0];
	eye_color[1][2]=rgb[0];
	eye_color[1][1]=rgb[1];
	eye_color[1][0]=rgb[2];
}
void get_ear_color(cv::Mat ear_img,cv::Scalar ear_color[1]){
	cv::Mat dst_img = ear_img.clone();
	get_color(dst_img,cv::Point(dst_img.cols/2,dst_img.rows/2),ear_color);
}

void get_mouth_color(cv::Mat mouth_img,cv::Scalar mouth_color[1]){
	cv::Mat dst_img = mouth_img.clone();
	get_color(dst_img,cv::Point(dst_img.cols/3,dst_img.rows*1/5),mouth_color);
}