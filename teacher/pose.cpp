#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "teacher/teacher.hpp"
using namespace cv;
using namespace std;
using namespace caffe;
vector<vector<float>> pose_detect(Net &net,Mat &oriImg){
	
	const int stride = 8;
	/*Net net("../models/pose_deploy.prototxt");
	net.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");*/
	
	float scale = 368.0 / oriImg.size[0];
	Mat imagetotest;
	cv::resize(oriImg, imagetotest, Size(0, 0), scale, scale);
	vector<Mat> bgr;
	cv::split(imagetotest, bgr);
	bgr[0].convertTo(bgr[0], CV_32F, 1/256.f, -0.5);
	bgr[1].convertTo(bgr[1], CV_32F, 1/256.f, -0.5);
	bgr[2].convertTo(bgr[2], CV_32F, 1/256.f, -0.5);
	
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, imagetotest.rows, imagetotest.cols);
	
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	net.Forward();
	shared_ptr<Blob> output_blobs = net.blob_by_name("Mconv7_stage6_L2");
	
	Mat heatmap = Mat::zeros(output_blobs->height(), output_blobs->width(), CV_32FC(19));
	
	for (int i = 0; i < output_blobs->channels(); i++){
		for (int j = 0; j < output_blobs->height(); j++){
			for (int k = 0; k < output_blobs->width(); k++){
				heatmap.at<float>(j, 19*k+i) = output_blobs->data_at(0, i, j, k);	
			
			}
		}
	}
	
	cv::resize(heatmap, heatmap, Size(0, 0), stride, stride);
	cv::resize(heatmap, heatmap, cv::Size(oriImg.size[1], oriImg.size[0]));
	/*for (int i = 0; i < 9; i++){
		cout << heatmap.at<float>(800, i) << "  " << endl;
	}*/
	Mat compare1, compare2, compare3, compare4, compare5;
	Mat bool_1, bool_2, bool_3, bool_4;
	Mat map_1(oriImg.size[0], oriImg.size[1], CV_32F, cv::Scalar::all(0.1));
	vector<vector<float>>all_peaks;
	vector<float>peaks;
	Point max_loc;
	
	double max_val=0;
	Mat map_ori = Mat::zeros(oriImg.size[0], oriImg.size[1],CV_32F);
	//cout << oriImg.size[0] << oriImg.size[1] << endl;
	
	for (int i = 0; i < 9; i++){
		for (int j = 0; j < oriImg.size[0]; j++){
			for (int k = 0; k < oriImg.size[1]; k++){
				map_ori.at<float>(j, k) = heatmap.at<float>(j, 19 * k + i);
			}
		}
		Mat map;
		GaussianBlur(map_ori, map, Size(7,7),3, 3);

		Mat map_left = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.rowRange(0, oriImg.size[0] - 1).copyTo(map_left.rowRange(1, oriImg.size[0]));

		Mat map_right = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.rowRange(1, oriImg.size[0]).copyTo(map_right.rowRange(0, oriImg.size[0] - 1));

		Mat map_up = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.colRange(0, oriImg.size[1] - 1).copyTo(map_up.colRange(1, oriImg.size[1]));

		Mat map_down = Mat::zeros(oriImg.size[0], oriImg.size[1], CV_32F);
		map.colRange(1, oriImg.size[1]).copyTo(map_down.colRange(0, oriImg.size[1] - 1));

		//cout << map.rowRange(0, oriImg.size[0] - 1) << endl;
		//cout << map_left.rowRange(1, oriImg.size[0]) << endl;

		compare(map, map_left, compare1, CMP_GE);
		compare(map, map_right, compare2, CMP_GE);
		compare(map, map_up, compare3, CMP_GE);
		compare(map, map_down, compare4, CMP_GE);
		compare(map, map_1, compare5, CMP_GT);
		
		bitwise_and(compare1, compare2, bool_1);
		bitwise_and(compare3, bool_1, bool_2);
		bitwise_and(compare4, bool_2, bool_3);
		bitwise_and(compare5, bool_3, bool_4);
		
		minMaxLoc(bool_4, 0, 0, 0, &max_loc);
	
		peaks.push_back(max_loc.x);
		peaks.push_back(max_loc.y);
		peaks.push_back(map_ori.at<float>(max_loc.y, max_loc.x));
		all_peaks.push_back(peaks);
		peaks.clear();

	}
	/*for (int i = 2; i < 8; i++){
		if (all_peaks[i].size() != 0 && all_peaks[i][2]>0.15){
			cv::circle(oriImg, Point(all_peaks[i][0], all_peaks[i][1]), 2, cv::Scalar(0, 255, 0), -1);
		}
	}*/
	//cv::imwrite("1.jpg", oriImg);
	//cout << heatmap.at<float>(3, 19 * 3 + 6) << endl;
	//std::cout << output_blobs->num() << output_blobs->channels() << output_blobs->height() << output_blobs->width()<<endl;
	return all_peaks;
}