#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
	// open the Webcam
	VideoCapture cap(0);
	// if not success, exit program
	if (cap.isOpened() == false)
	{
		cout << "Cannot open Webcam" << endl;
		return -1;
	}
	// get the frames rate of the video
	double fps = cap.get(CAP_PROP_FPS);
	cout << "Frames per seconds : " << fps << endl;
	cout << "Press Q to Quit" << endl;
	String win_name = "Webcam Video";
	// create a window
	namedWindow(win_name);
	// set frame size according to camera
	Size frame_size(640, 640);
	int frames_per_second = 30; // video fps
	int delay = 1000 / frames_per_second;
	// open the video writer
	VideoWriter v_writer("images/video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, frame_size, true);
	if (!v_writer.isOpened())
	{
		cout << "Cannot open VideoWriter" << endl;
		return -1;
	}
	while (true)
	{
		Mat frame;
		// read a new frame from video
		bool flag = cap.read(frame);
		// write a frame to file
		v_writer.write(frame);
		// show the frame in the created window
		imshow(win_name, frame);
		if (waitKey(delay) == 'q')
		{
			break;
		}
		if (!cap.grab())
		{
			cout << "grab failed!" << endl;
			break;
		}
	}
	// After finishing video write
	v_writer.release();
	return 0;
}
