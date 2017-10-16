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
	cv::Rect bbox;
	bool writing=false;
	bool pointing=false;
};
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre, int &n);
#endif