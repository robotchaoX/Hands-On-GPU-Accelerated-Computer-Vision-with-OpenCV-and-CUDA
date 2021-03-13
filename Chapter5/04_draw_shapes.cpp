#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	// Create blank white color Image with size 256x256
	Mat img(512, 512, CV_8UC3, Scalar(0, 0, 0));
	line(img, Point(0, 0), Point(511, 511), Scalar(0, 255, 0), 7);
	rectangle(img, Point(384, 0), Point(510, 128), Scalar(255, 255, 0), 5);
	circle(img, Point(447, 63), 63, Scalar(0, 0, 255), -1);
	ellipse(img, Point(256, 256), Point(100, 100), 0, 0, 180, 255, -1);
	putText(img, "OpenCV!", Point(10, 500), FONT_HERSHEY_SIMPLEX, 3, Scalar(255, 255, 255), 5, 8);
	// Name of the window
	String win_name = "Blank Blue Color Image";
	// Create a window
	namedWindow(win_name);
	// Show our image inside the created window.
	imshow(win_name, img);
	// Wait for any keystroke in the window
	waitKey(0);
	// destroy the created window
	destroyWindow(win_name);
	return 0;
}
