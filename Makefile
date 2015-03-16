CC = g++
CFLAGS = -g -w
COPENCV = -I/path/to/opencv/include -L/path/to/opencv/lib -lopencv_core -lopencv_highgui  -lopencv_objdetect -lopencv_features2d -lopencv_imgproc

ALL: avatar_drawer.o cat_eye_circle_sobel.o ellipse_line_cross_calc.o cat_ear_line_canny.o cat_classify.o detect_cat_face.o
	$(CC) $(CFLAGS) -o cat_avatar avatar_drawer.o cat_eye_circle_sobel.o ellipse_line_cross_calc.o cat_ear_line_canny.o cat_classify.o detect_cat_face.o $(COPENCV)

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

detect_cat_face.o: detect_cat_face.cpp
	$(CC) $(CFLAGS) -o detect_cat_face.o -c detect_cat_face.cpp $(COPENCV)	 