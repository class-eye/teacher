#ifndef _TEACHER_H_
#define _TEACHER_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "jfda.hpp"
#include "pose.hpp"
#include "rfcn.hpp"
using namespace cv;
using namespace caffe;
struct Teacher_Info{
	cv::Rect bbox;
	bool writing;
	bool have_teacher;
};
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre,int &n);
#endif