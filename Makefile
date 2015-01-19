CC = g++
CFLAGS = -g -Wall
COPENCV = -I/path/to/opencv/include -L/path/to/opencv/lib -lopencv_core -lopencv_highgui  -lopencv_objdetect -lopencv_features2d -lopencv_imgproc

ALL: avatar_drawer.o cat_eye_circle_sobel.o ellipse_line_cross_calc.o cat_ear_line_canny.o cat_classify.o
	$(CC) $(CFLAGS) -o avatar_drawer avatar_drawer.o cat_eye_circle_sobel.o ellipse_line_cross_calc.o cat_ear_line_canny.o cat_classify.o $(COPENCV)

avatar_drawer.o: avatar_drawer.cpp prototype.h
	$(CC) $(CFLAGS) -o avatar_drawer.o -c avatar_drawer.cpp $(COPENCV)

cat_eye_circle_sobel.o: cat_eye_circle_sobel.cpp
	$(CC) $(CFLAGS) -o cat_eye_circle_sobel.o -c cat_eye_circle_sobel.cpp $(COPENCV)

ellipse_line_cross_calc.o: ellipse_line_cross_calc.cpp
	$(CC) $(CFLAGS) -o ellipse_line_cross_calc.o -c ellipse_line_cross_calc.cpp $(COPENCV)

cat_ear_line_canny.o: cat_ear_line_canny.cpp
	$(CC) $(CFLAGS) -o cat_ear_line_canny.o -c cat_ear_line_canny.cpp $(COPENCV)

cat_classify.o: cat_classify.cpp
	$(CC) $(CFLAGS) -o cat_classify.o -c cat_classify.cpp $(COPENCV)