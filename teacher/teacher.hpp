#ifndef _TEACHER_H_
#define _TEACHER_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "pose.hpp"
#include "rfcn.hpp"
#include "jfda.hpp"
using namespace cv;
using namespace caffe;
struct Teacher_Info{
	Point location;
	bool writing=false;
	bool front_pointing=false;
	bool back_pointing = false;
	int num;
	bool teacher_in_screen = false;
	vector<Point2f>all_points;
};
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre, int &n);
extern string videoname;
#endif