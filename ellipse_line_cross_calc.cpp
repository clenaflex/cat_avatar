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

using namespace std;

int which_big_num(int a,int b){
  if(a<b){
    return 1;
  }else if(b<a){
    return 0;
  }else{
    return 0;
  }
}

int which_small_num(int a,int b){
  if(a<b){
    return 0;
  }else if(b<a){
    return 1;
  }else{
    return 0;
  }
}


void perpendicular_calc(int cx,int cy,int x1,int y1,int x2,int y2,int fp[2]){
  double a = (y2 - y1)/( (x2 - x1)*10*0.1 );
  double b = -1.0;
  double c = y2 - a * x2;
  double coef = (a*cx + b*cy + c)/( (a*a+b*b)*10*0.1 );
  fp[0] = (cx - coef*a);
  fp[1] = (cy - coef*b);
}

double dist_two_point(double x1,double y1,double x2,double y2){
 return sqrt( pow( (x1 - x2) ,2 ) + pow( (y1 - y2) ,2));
}

double calc_tan(double x1,double y1,double x2,double y2){
  double la = (y2 - y1)/( (x2 - x1)*10*0.1);
  return la;
}

void ellipse_line_cross_calc(double p,double q,double a,double b,double x1,double y1, double x2,double y2,double ans[2]){
  double la = (y2 - y1)/(x2 - x1);
  double lb = y2 - la * x2;
  //判別式においては
  //aa*x^2+bb*x+cc=0
  //の形式
  double aa = (a*a*la*la + b*b);
  double bb = 2*la*a*a*lb -2*la*a*a*q -2*b*b*p;
  double cc = b*b*p*p + a*a*lb*lb - 2*a*a*lb*q + a*a*q*q - a*a*b*b;
  double d = bb*bb - 4*aa*cc;
  cout << "d:" << d << endl;
  if (d > 0){
   double px = (-1* bb + sqrt(d)) / ( 2*aa );
   double mx = (-1* bb - sqrt(d)) / ( 2*aa );
   double py = la * px +lb;
   double my = la * mx +lb;
   if (py > my){
    //楕円との交点
    ans[0] = mx;
    ans[1] = my;
   }else{
    ans[0] = px;
    ans[1] = py;
   }
   cout << "on-ellipse-x:" << ans[0] << " " << "on-ellipse-y:" << ans[1] << endl;
  }else
  {
    cout << "判別式の値が不正です。" << endl;

  }
}

void line_cross_point(double x1,double y1,double x2,double y2,double x3,double y3,double x4,double y4,double ans[2]){
  double la = (y2 - y1)/(x2 - x1);
  double lb = y2 - la * x2;
  double lc = (y4 - y3)/(x4 - x3);
  double ld = y4 - lc * x4;
  double x = (ld - lb) / (la - lc);
  double y = la * x + lb;
  ans[0] = x;
  ans[1] = y;
  cout << "cross_point:" << ans[0] << " " << ans[1] << endl;
}



void ellipse_ear_calc(double p,double q,double a,double b,double lx1,double ly1,double lx2,double ly2,double lx3,double ly3,double lx4,double ly4,double rx1,double ry1,double rx2,double ry2,double rx3,double ry3,double rx4,double ry4,double ans[12]){
  /*
  ans[0] = 左交点x
  ans[1] = 左交点y
  ans[2] = 左楕円上1x
  ans[3] = 左楕円上1y
  ans[4] = 左楕円上2x
  ans[5] = 左楕円上2y
  ans[6] = 右交点x
  ans[7] = 右交点y
  ans[8] = 右楕円上1x
  ans[9] = 右楕円上1y
  ans[10] = 右楕円上2x
  ans[11] = 右楕円上2y
  */
  double l1ans[2];
  double l2ans[2];
  double r1ans[2];
  double r2ans[2];
  double l12ans[2];
  double r12ans[2];

  ellipse_line_cross_calc(p,q,a,b,lx1,ly1, lx2,ly2,l1ans);
  ellipse_line_cross_calc(p,q,a,b,lx3,ly3, lx4,ly4,l2ans);
  ellipse_line_cross_calc(p,q,a,b,rx1,ry1, rx2,ry2,r1ans);
  ellipse_line_cross_calc(p,q,a,b,rx3,ry3, rx4,ry4,r2ans);

  line_cross_point(lx1,ly1,lx2,ly2,lx3,ly3,lx4,ly4,l12ans);
  line_cross_point(rx1,ry1,rx2,ry2,rx3,ry3,rx4,ry4,r12ans);

  ans[0] = l1ans[0];
  ans[1] = l1ans[1];
  ans[2] = l2ans[0];
  ans[3] = l2ans[1];
  ans[4] = l12ans[0];
  ans[5] = l12ans[1];
  ans[6] = r1ans[0];
  ans[7] = r1ans[1];
  ans[8] = r2ans[0];
  ans[9] = r2ans[1];
  ans[10] = r12ans[0];
  ans[11] = r12ans[1];
}

void draw_face_ellipse(cv::Mat& img,double p,double q,double a,double b,cv::Scalar bgr[3],int type){
//B,G,Rの順に格納すること

  /// 画像，中心座標，（長径・短径），回転角度，円弧開始角度，円弧終了角度，色，線太さ，連結
  // Red，太さ3，4近傍連結
  //bgr[0]は向かって左
  //bgr[1]は向かって右
  //bgr[2]はハチワレ部
  /*
  all:10
  all+shima:11
  hachiware:20
  hachi+shima:21
  hachi+shima+mike:22
  pointed:30
  */
    cv::Point pt[3];
    pt[0] = cv::Point(p,q-(4*b/5));
    pt[1] = cv::Point(p+a/4,q+(b)/2);
    pt[2] = cv::Point(p-a/4,q+(b)/2);
   // cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,150,270,bgr[0], -1, 4);
   // cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,-90,30,bgr[1], -1, 4);
   // cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,30,150,bgr[2], -1, 4);
   // cv::fillConvexPoly(img, pt, 3,bgr[2]);
   if (type == 10){
    cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,180,360,bgr[0], -1, 4);
    cv::ellipse(img,cv::Point(p,q), cv::Size(a,b*1.1),0,0,180,bgr[0], -1, 4);
   }else if(type == 20 || type == 30 ){
       cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,150,270,bgr[0], -1, 4);
       cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,-90,30,bgr[1], -1, 4);
       cv::ellipse(img,cv::Point(p,q), cv::Size(a,b),0,30,150,bgr[2], -1, 4);
       cv::fillConvexPoly(img, pt, 3,bgr[2]);
   }
}

void draw_ear_two_triangle(cv::Mat& img,cv::Point l_pt[3],cv::Scalar l_bgr[2],cv::Point r_pt[3],cv::Scalar r_bgr[2]){
  cv::Point pt[3];
  double x;
  double y;
  // 左外

  if (l_pt[2].y < 0){
    l_pt[2].y += 2*abs(l_pt[2].y);
  }
  if (r_pt[2].y < 0){
    r_pt[2].y += 2*abs(r_pt[2].y);
  }

  cv::fillConvexPoly(img, l_pt, 3,l_bgr[0]);
  double l_0_a,l_0_b,l_1_a,l_1_b;
  double r_0_a,r_0_b,r_1_a,r_1_b;
  double l_theta_0,l_theta_1;
  double r_theta_0,r_theta_1;

  l_theta_0 = atan(calc_tan(l_pt[0].x,l_pt[0].y,l_pt[2].x,l_pt[2].y))*180/3.14;
  l_theta_1 = atan(calc_tan(l_pt[1].x,l_pt[1].y,l_pt[2].x,l_pt[2].y))*180/3.14;
  cout << "theta" << l_theta_0 << " " << l_theta_1 << endl;
  l_0_a = dist_two_point(l_pt[0].x,l_pt[0].y,l_pt[2].x ,l_pt[2].y)/2;
  l_0_b = l_0_a*2/10;
  l_1_a = dist_two_point(l_pt[1].x,l_pt[1].y,l_pt[2].x ,l_pt[2].y)/2;
  l_1_b = l_0_a*2/10;
 
  cv::ellipse(img,cv::Point( (l_pt[0].x+l_pt[2].x)/2.0 ,(l_pt[0].y+l_pt[2].y)/2.0 ), cv::Size(l_0_a,l_0_b),l_theta_0,0,360,l_bgr[0], -1, 8);

  cv::ellipse(img,cv::Point( (l_pt[1].x+l_pt[2].x)/2.0 ,(l_pt[1].y+l_pt[2].y)/2.0  ), cv::Size(l_1_a,l_1_b),l_theta_1,0,360,l_bgr[0], -1, 8);
  

  cv::Point hl_pt[3];
  hl_pt[0] = cv::Point( l_pt[2].x-(l_pt[2].x - l_pt[0].x)*2/5 ,l_pt[2].y-(l_pt[2].y - l_pt[0].y)*2/5);
  hl_pt[1] = cv::Point(l_pt[2].x-(l_pt[2].x - l_pt[1].x)*2/5 ,l_pt[2].y-(l_pt[2].y - l_pt[1].y)*2/5);
  hl_pt[2] = cv::Point(l_pt[2].x,l_pt[2].y);
  
  //左内
  x = (l_pt[0].x + l_pt[1].x + l_pt[2].x)/3;
  y = (l_pt[0].y + l_pt[1].y + l_pt[2].y)/3;
  pt[0] = cv::Point((x+l_pt[0].x)/2,(y+l_pt[0].y)/2); 
  pt[1] = cv::Point((x+l_pt[1].x)/2,(y+l_pt[1].y)/2);
  pt[2] = cv::Point((x+l_pt[2].x)/2,(y+l_pt[2].y)/2);
  cv::fillConvexPoly(img, pt, 3,l_bgr[1]);
  cv::line(img, pt[0], pt[1],cv::Scalar(0,0,0), 2, 4);
  cv::line(img, pt[1], pt[2],cv::Scalar(0,0,0), 2, 4);
  cv::line(img, pt[2], pt[0],cv::Scalar(0,0,0), 2, 4);

  //右外
  cv::fillConvexPoly(img, r_pt, 3,r_bgr[0]);
  cv::Point hr_pt[3];
  hr_pt[0] = cv::Point( r_pt[2].x-(r_pt[2].x - r_pt[0].x)*2/5 ,r_pt[2].y-(r_pt[2].y - r_pt[0].y)*2/5);
  hr_pt[1] = cv::Point(r_pt[2].x-(r_pt[2].x - r_pt[1].x)*2/5 ,r_pt[2].y-(r_pt[2].y - r_pt[1].y)*2/5);
  hr_pt[2] = cv::Point(r_pt[2].x,r_pt[2].y);

  r_theta_0 = atan(calc_tan(r_pt[0].x,r_pt[0].y,r_pt[2].x,r_pt[2].y))*180/3.14;
  r_theta_1 = atan(calc_tan(r_pt[1].x,r_pt[1].y,r_pt[2].x,r_pt[2].y))*180/3.14;
  cout << "theta" << r_theta_0 << " " << r_theta_1 << endl;
  r_0_a = dist_two_point(r_pt[0].x,r_pt[0].y,r_pt[2].x ,r_pt[2].y)/2.0;
  r_0_b = r_0_a*2/10.0;
  r_1_a = dist_two_point(r_pt[1].x,r_pt[1].y,r_pt[2].x ,r_pt[2].y)/2.0;
  r_1_b = r_0_a*2/10.0;
 

  cv::ellipse(img,cv::Point( (r_pt[0].x+r_pt[2].x)/2.0 ,(r_pt[0].y+r_pt[2].y)/2.0 ), cv::Size(r_0_a,r_0_b),r_theta_0,0,360,r_bgr[0], -1, 8);
  cv::ellipse(img,cv::Point( (r_pt[1].x+r_pt[2].x)/2.0 ,(r_pt[1].y+r_pt[2].y)/2.0  ), cv::Size(r_1_a,r_1_b),r_theta_1,0,360,r_bgr[0], -1, 8);

  //右内
  x = (r_pt[0].x + r_pt[1].x + r_pt[2].x)/3;
  y = (r_pt[0].y + r_pt[1].y + r_pt[2].y)/3;
  pt[0] = cv::Point((x+r_pt[0].x)/2,(y+r_pt[0].y)/2); 
  pt[1] = cv::Point((x+r_pt[1].x)/2,(y+r_pt[1].y)/2);
  pt[2] = cv::Point((x+r_pt[2].x)/2,(y+r_pt[2].y)/2);
  cv::fillConvexPoly(img, pt, 3,r_bgr[1]);
  cv::line(img, pt[0], pt[1], cv::Scalar(0,0,0), 2, 4);
  cv::line(img, pt[1], pt[2], cv::Scalar(0,0,0), 2, 4);
  cv::line(img, pt[2], pt[0], cv::Scalar(0,0,0), 2, 4);

}

void draw_eye_two_ellipse(cv::Mat& img,cv::Point pt_lo,cv::Size sz_lo,int theta_lo,cv::Point pt_li,cv::Size sz_li,int theta_li,cv::Scalar le_bgr[2],cv::Point pt_ro,cv::Size sz_ro,int theta_ro,cv::Point pt_ri,cv::Size sz_ri,int theta_ri,cv::Scalar re_bgr[2]){
  //lo楕円外側左目
  //li楕円内側左目
  double in_a,in_b;

  /// 画像，中心座標，（長径・短径），回転角度，円弧開始角度，円弧終了角度，色，線太さ，連結
  
  cv::ellipse(img,pt_lo, sz_lo,theta_lo,0,360,le_bgr[0], -1, 8);
  cv::ellipse(img,pt_lo, sz_lo,theta_lo,0,360,cv::Scalar(0,0,0), 2, 8);
  // cout << pt_lo << " "<< sz_lo << " "<<  theta_lo << " "<< le_bgr[0] << endl;
  cv::ellipse(img,pt_li, sz_li,theta_li,0,360,le_bgr[1], -1, 8);
  cv::ellipse(img,pt_li, sz_li,theta_li,0,360,cv::Scalar(0,0,0), 2, 8);

  if(which_small_num(sz_li.width,sz_li.height) == 0 ){
    in_a = sz_li.height;
    in_b = sz_li.width;
  }else{
    in_b = sz_li.height;
    in_a = sz_li.width;
  }
  cv::circle(img, cv::Point(pt_li.x-(in_b/4),pt_li.y-(in_a/4)), in_b/4, cv::Scalar(255,255,255), -1, 8);

  // cout << pt_li << " "<< sz_li << " "<<  theta_li << " "<< le_bgr[1] << endl;
  cv::ellipse(img,pt_ro, sz_ro,theta_ro,0,360,re_bgr[0], -1, 8);
  cv::ellipse(img,pt_ro, sz_ro,theta_ro,0,360,cv::Scalar(0,0,0), 2, 8);
  // cout << pt_ro << " "<< sz_ro << " "<<  theta_ro << " "<< re_bgr[0] << endl;
  cv::ellipse(img,pt_ri, sz_ri,theta_ri,0,360,re_bgr[1], -1, 8);
  cv::ellipse(img,pt_ri, sz_ri,theta_ri,0,360,cv::Scalar(0,0,0), 2, 8);

 if(which_small_num(sz_ri.width,sz_ri.height) == 0 ){
    in_a = sz_ri.height;
    in_b = sz_ri.width;
  }else{
    in_b = sz_ri.height;
    in_a = sz_ri.width;
  }
  cv::circle(img, cv::Point(pt_ri.x-(in_b/4),pt_ri.y-(in_a/4)), in_b/4, cv::Scalar(255,255,255), -1, 8);

  // cout << pt_ri << " "<< sz_ri << " "<<  theta_ri << " "<< re_bgr[1] << endl;
}

void draw_mouth(cv::Mat& img,cv::Point pt_m,int w, int h,cv::Scalar c_m){
  cv::line(img, cv::Point(pt_m.x+w/2, pt_m.y+h/4), cv::Point(pt_m.x+w/2, pt_m.y+h/2), cv::Scalar(0,0,0), 3, 4);  
  cv::line(img, cv::Point(pt_m.x+w/2, pt_m.y+h/2), cv::Point(pt_m.x+w/4, pt_m.y+5*h/8), cv::Scalar(0,0,0), 3, 4);  
  cv::line(img, cv::Point(pt_m.x+w/2, pt_m.y+h/2), cv::Point(pt_m.x+w*3/4, pt_m.y+5*h/8), cv::Scalar(0,0,0), 3, 4);  
  cv::circle(img, cv::Point(pt_m.x+w/2,pt_m.y+h/4),h/8,c_m, -1, CV_AA);
}

// int main(int argc, char** argv) {
//    double param = 1;
//    int p = atoi(argv[1])*param;
//    int q = atoi(argv[2])*param;
//    int a = atoi(argv[3])*param;
//    int b = atoi(argv[4])*param; 
//    // double lx1 = atoi(argv[5])*5;
//    // double ly1 = atoi(argv[6])*5;
//    // double lx2 = atoi(argv[7])*5;
//    // double ly2 = atoi(argv[8])*5;
//    // double lx3 = atoi(argv[9])*5;
//    // double ly3 = atoi(argv[10])*5;
//    // double lx4 = atoi(argv[11])*5;
//    // double ly4 = atoi(argv[12])*5;
//    // double rx1 = atoi(argv[13])*5;
//    // double ry1 = atoi(argv[14])*5;
//    // double rx2 = atoi(argv[15])*5;
//    // double ry2 = atoi(argv[16])*5;
//    // double rx3 = atoi(argv[17])*5;
//    // double ry3 = atoi(argv[18])*5;
//    // double rx4 = atoi(argv[19])*5;
//    // double ry4 = atoi(argv[20])*5;

//    double ans[12];

//    ellipse_ear_calc(p,q,a,b,atoi(argv[5])*param,atoi(argv[6])*param,atoi(argv[7])*param,atoi(argv[8])*param,atoi(argv[9])*param,atoi(argv[10])*param,atoi(argv[11])*param,atoi(argv[12])*param,atoi(argv[13])*param,atoi(argv[14])*param,atoi(argv[15])*param,atoi(argv[16])*param,atoi(argv[17])*param,atoi(argv[18])*param,atoi(argv[19])*param,atoi(argv[20])*param,ans);

//    // cv::Mat frame = cv::Mat::zeros(cv::Size(800, 800), CV_8UC3);
//    cv::Mat frame(1000, 1000,CV_8UC3,cv::Scalar(194,194,194));
//    cv::Point l_pt[3];
//    cv::Point r_pt[3];
//     l_pt[0] = cv::Point(ans[0],ans[1]);
//     l_pt[1] = cv::Point(ans[2],ans[3]);
//     l_pt[2] = cv::Point(ans[4],ans[5]);

//     r_pt[0] = cv::Point(ans[6],ans[7]);
//     r_pt[1] = cv::Point(ans[8],ans[9]);
//     r_pt[2] = cv::Point(ans[10],ans[11]);

//     cv::Scalar l_bgr[2];
//     l_bgr[0] = cv::Scalar(0,0,0);
//     l_bgr[1] = cv::Scalar(92,107,109);

//     cv::Scalar r_bgr[2];
//     r_bgr[0] = cv::Scalar(0,0,0);
//     r_bgr[1] = cv::Scalar(92,107,109);

//     draw_ear_two_triangle(frame,l_pt,l_bgr,r_pt,r_bgr);

//     // cout << "pt-l1:" << "(" << ans[0] << "," << ans[1] << ")" << endl;
//     // cout << "pt-l2:" << "(" << ans[2] << "," << ans[3] << ")" << endl;
//     // cout << "pt-l3:" << "(" << ans[4] << "," << ans[5] << ")" << endl;
//     // // cv::fillConvexPoly(frame, pt, 3, cv::Scalar(0,0,200));

//     // // cv::fillConvexPoly(frame, pt, 3, cv::Scalar(0,0,200));
//     // cout << "pt-r1:" << "(" << ans[6] << "," << ans[7] << ")" << endl;
//     // cout << "pt-r2:" << "(" << ans[8] << "," << ans[9] << ")" << endl;
//     // cout << "pt-r3:" << "(" << ans[10] << "," << ans[11] << ")" << endl;

//     cv::Scalar bgr[3];
//     bgr[0] = cv::Scalar(0,0,0);
//     bgr[1] = cv::Scalar(0,0,0);
//     bgr[2] = cv::Scalar(255,255,255);
//     int type = 20;
//    draw_face_ellipse(frame,p,q,a,b,bgr,type);
   
//    cv::Scalar le_bgr[2];
//    le_bgr[0] = cv::Scalar(97,109,109);
//    le_bgr[1] = cv::Scalar(0,0,0);

//    cv::Scalar re_bgr[2];
//    re_bgr[0] = cv::Scalar(97,109,109);
//    re_bgr[1] = cv::Scalar(0,0,0);

//    draw_eye_two_ellipse(frame,cv::Point(atoi(argv[21])*param,atoi(argv[22])*param),cv::Size(atoi(argv[23])*param,atoi(argv[24])*param),atoi(argv[25]),cv::Point(atoi(argv[26])*param,atoi(argv[27])*param),cv::Size(atoi(argv[28])*param,atoi(argv[29])*param),atoi(argv[30]),le_bgr,cv::Point(atoi(argv[31])*param,atoi(argv[32])*param),cv::Size(atoi(argv[33])*param,atoi(argv[34])*param),atoi(argv[35]),cv::Point(atoi(argv[36])*param,atoi(argv[37])*param),cv::Size(atoi(argv[38])*param,atoi(argv[39])*param),atoi(argv[40]),re_bgr);
   
//    cv::Scalar c_m = cv::Scalar(204,209,210);
   
//    draw_mouth(frame,cv::Point(atoi(argv[41])*param,atoi(argv[42])*param),atoi(argv[43])*param,atoi(argv[44])*param,c_m);

//    cv::imwrite("test_img.png",frame);
//    cv::namedWindow("Capture", CV_WINDOW_AUTOSIZE|CV_WINDOW_FREERATIO);
//    cv::imshow("Capture", frame);
//    cv::waitKey(0);
// }

