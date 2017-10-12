
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
#include "teacher/Timer.hpp"


using namespace cv;
using namespace std;
using namespace caffe;

static float CalculateVectorAngle(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float x_1 = x2 - x1;
	float x_2 = x3 - x2;
	float y_1 = y2 - y1;
	float y_2 = y3 - y2;
	float lx = sqrt(x_1*x_1 + y_1*y_1);
	float ly = sqrt(x_2*x_2 + y_2*y_2);
	return 180.0 - acos((x_1*x_2 + y_1*y_2) / lx / ly) * 180 / 3.1415926;
}
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre,int &n){
	//cout << box_pre.x << " " << box_pre.y << " " << box_pre.width << " " << box_pre.height << endl;
	Teacher_Info teacher_info;
	Rect bbox_new;
	vector<Rect>all_bbox;
	Rect bbox_new1;
	bool yes_or_no;
	Timer timer;
	if (n % 25 == 0){
		//Í¿µô²¿·ÖÇøÓò
		Mat img_copy;
		img_copy = img;
		int height = 800;
		cv::rectangle(img_copy, Point(0, height), Point(img.size[1],img.size[0]),Scalar(0,0,0),-1,8,0);

		timer.Tic();
		//detect body
		all_bbox = im_detect(net2, img_copy, box_pre);
		cv::rectangle(img, box_pre, cv::Scalar(0, 0, 255), 2);
		//timer.Toc();
		//cout << "detect body cost " << timer.Elasped() / 1000.0 << "s" << endl;
		/*if (all_bbox.size() > 1){
			for (Rect box : all_bbox){
				cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);
			}
		}*/
		if (all_bbox.size() == 1){
			bbox_new = all_bbox[0];
			if (bbox_new.x != box_pre.x || bbox_new.y != box_pre.y || bbox_new.width != box_pre.width || bbox_new.height != box_pre.height){
				cv::rectangle(img, bbox_new, cv::Scalar(0, 255, 0), 2);
			}

			teacher_info.bbox = bbox_new;
			//detect pose
			bbox_new1.height = bbox_new.height;
			bbox_new1.width = int(bbox_new.height * 1);
			bbox_new1.x = bbox_new.x - (bbox_new1.width - bbox_new.width) / 2;
			bbox_new1.y = bbox_new.y;
			bbox_new1 = refine(img, bbox_new1);
			Mat im_body = img(bbox_new1);
			vector<vector<float>>all_peaks = pose_detect(net1, im_body);

			for (int i = 2; i < 8; i++){
				if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
					cv::circle(img, Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y), 4, cv::Scalar(0, 0, 255), -1);
				}
			}

			//detect face
			vector<FaceInfoInternal>facem;
			vector<FaceInfo> faces = detector.Detect(im_body, facem);

			//pose estimation
			int symbol_l = 0;
			int symbol_r = 0;
			if (all_peaks[4].size() != 0 && all_peaks[3].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2 && all_peaks[3][2] > 0.2 && all_peaks[2][2] > 0.2){
				if (faces.size() == 0){
					if (((all_peaks[4][1] < all_peaks[3][1]) && (all_peaks[3][1] < all_peaks[2][1]))){
						symbol_r = 1;
					}
				}
				float angle_r = CalculateVectorAngle(all_peaks[2][0], all_peaks[2][1], all_peaks[3][0], all_peaks[3][1], all_peaks[4][0], all_peaks[4][1]);
				if (all_peaks[4][1] <= all_peaks[2][1]){		
					if (angle_r > 120)symbol_r = 1;
				}
			}
			if (all_peaks[7].size() != 0 && all_peaks[6].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2 && all_peaks[6][2] > 0.2 && all_peaks[5][2] > 0.2){
				if (faces.size() == 0){
					if ((all_peaks[7][1] < all_peaks[6][1]) && (all_peaks[6][1] < all_peaks[5][1])){
						symbol_l = 1;
					}
				}
				float angle_l = CalculateVectorAngle(all_peaks[5][0], all_peaks[5][1], all_peaks[6][0], all_peaks[6][1], all_peaks[7][0], all_peaks[7][1]);
				if (all_peaks[7][1] <= all_peaks[5][1]){		
					if (angle_l > 120)symbol_l = 1;
				}
			}
			
			if (symbol_l == 1 || symbol_r == 1){
				string status = "Doing Action";
				yes_or_no = true;
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2), FONT_HERSHEY_DUPLEX,1,Scalar(255,255,255));

			}
			else{
				yes_or_no = false;
			}
			teacher_info.writing = yes_or_no;
			
			string output = "../output";
			char buff[300];
			sprintf(buff, "%s/%d.jpg", output.c_str(), n);
			cv::imwrite(buff, img);
			timer.Toc();
			cout << "Frame " << n << " cost " << timer.Elasped() / 1000.0 << "s" << endl;
			return teacher_info;
		}
	}
}

