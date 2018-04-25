#ifndef _TEACHER_H_
#define _TEACHER_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include<queue>
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
	bool interaction = false;
	int num;
	bool teacher_in_screen = false;
	vector<Point2f>all_points;
};
//Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre, int &n);
//extern string videoname;

class Teacher_analy{
public:
	Teacher_analy(const string& pose_net, const string& pose_model, const string& rfcn_net, const string& rfcn_model,int gpu_device=-1);
	~Teacher_analy();
	Teacher_Info teacher_detect(jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre, int &n);

private:
	Net *posenet;
	Net *rfcnnet;

	int raising_time = 0;
	int raising_total_time = 0;
	int interaction_time = 0;
	int interaction_total_time = 0;
	vector<int>have_face;

	int no_teacher = 0;
};

#endif