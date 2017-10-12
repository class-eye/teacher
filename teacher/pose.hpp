#ifndef _POSE_H_
#define _POSE_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
using namespace cv;
using namespace caffe;
vector<vector<float>> pose_detect(Net &net,cv::Mat &oriImg);
#endif