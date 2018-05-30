#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "teacher/teacher.hpp"
#include "teacher/Timer.hpp"
using namespace cv;
using namespace std;
using namespace caffe;
vector<vector<float>> pose_detect(Net &net,Mat &oriImg){
	Timer timer;
	float scale = 0.8*368.0 / oriImg.size[0];
	Mat imagetotest;
	cv::resize(oriImg, imagetotest, Size(0, 0), scale, scale);
	vector<Mat> bgr;
	cv::split(imagetotest, bgr);
	/*bgr[0].convertTo(bgr[0], CV_32F, 1/256.f, -0.5);
	bgr[1].convertTo(bgr[1], CV_32F, 1/256.f, -0.5);
	bgr[2].convertTo(bgr[2], CV_32F, 1/256.f, -0.5);*/

	bgr[0].convertTo(bgr[0], CV_32F, 1.f, -103.939);
	bgr[1].convertTo(bgr[1], CV_32F, 1.f, -116.779);
	bgr[2].convertTo(bgr[2], CV_32F, 1.f, -123.68);
	
	shared_ptr<Blob> data = net.blob_by_name("data");
	data->Reshape(1, 3, imagetotest.rows, imagetotest.cols);
	
	const int bias = data->offset(0, 1, 0, 0);
	const int bytes = bias*sizeof(float);
	
	memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
	memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
	memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
	//timer.Tic();
	net.Forward();
	/*timer.Toc();
	cout <<"forward cost " << timer.Elasped() / 1000.0 << "s" << endl;*/
	shared_ptr<Blob> output_blobs = net.blob_by_name("Mconv7_stage5_L2");
	
	vector<Mat>all_heatmap;
	const int bias1 = output_blobs->offset(0, 1, 0, 0);
	const int bytes1 = bias1*sizeof(float);
	for (int i = 0; i < output_blobs->channels(); i++){
		const float *data = output_blobs->cpu_data();
		Mat img(output_blobs->height(), output_blobs->width(), CV_32FC1, static_cast<void*>(const_cast<float*>(data + i*bias1)));
		cv::resize(img, img, cv::Size(oriImg.size[1], oriImg.size[0]));
		all_heatmap.push_back(img);
	}

	vector<vector<float>>all_peaks;
	for (int i = 0; i < 8; i++){
		vector<float>peaks;
		Mat map;
		GaussianBlur(all_heatmap[i], map, Size(13, 13), 5, 5);

		for (int j = 1; j < map.size().height - 1; j++){
			for (int k = 1; k < map.size().width - 1; k++){
				if (map.at<float>(j, k) > map.at<float>(j - 1, k) && map.at<float>(j, k) > map.at<float>(j + 1, k) && map.at<float>(j, k) > map.at<float>(j, k - 1) && map.at<float>(j, k) > map.at<float>(j, k + 1) && map.at<float>(j, k) > 0.1){
					peaks.push_back(k);
					peaks.push_back(j);
					peaks.push_back(all_heatmap[i].at<float>(j, k));
				}
			}
		}
		all_peaks.push_back(peaks);

	}
	return all_peaks;
}