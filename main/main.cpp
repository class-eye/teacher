#include <thread>
#include <fstream>
#include <iostream>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "cv.h"  
#include "teacher/teacher.hpp"
using namespace cv;
using namespace std;
using namespace caffe;
int main(){
	if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 0);
	}
	Net net1("../models/pose_deploy.prototxt");
	Net net2("../models/test_agnostic_50.prototxt");
	jfda::JfdaDetector detector("../models/p.prototxt", "../models/p.caffemodel", "../models/r.prototxt", "../models/r.caffemodel", \
		"../models/o.prototxt", "../models/o.caffemodel", "../models/l.prototxt", "../models/l.caffemodel");
	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
	net2.CopyTrainedLayersFrom("../models/resnet50_rfcn_final.caffemodel");

	VideoCapture capture("../2017.9.25##classroom7channel15__10.0--11.45.mp4");
	int frameH = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int frameW = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	Rect box_pre;
	box_pre.x = 0;
	box_pre.y = 0;
	box_pre.width = frameW;
	box_pre.height = frameH;
	Teacher_Info teacher_info;

	if (!capture.isOpened())
	{
		printf("video loading fail");
	}
	Mat frame;
	int n = 0;
	while (true)
	{
		if (!capture.read(frame)){
			break;
		}
		teacher_info = teacher_detect(net1, net2, detector, frame, box_pre, n);
		n++;
	}
	//teacher_info.bbox是老师的位置
	//teacher_info.writing是一个bool，说老师是否在板书
	//teacher_info.have_teacher是一个bool，说是否检测到老师
}