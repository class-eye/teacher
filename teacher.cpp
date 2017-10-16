
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

int raising_time = 0;
int raising_total_time = 0;
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector,cv::Mat &img, Rect &box_pre, int &n){
	Teacher_Info teacher_info;
	Rect bbox_new;
	vector<Rect>all_bbox;
	Rect bbox_new1;
	Timer timer;
	
	if (n % 25 == 0){
		Mat img_copy;
		img.copyTo(img_copy);
		/*int height = 800;
		cv::rectangle(img_copy, Point(0, height), Point(img.size[1],img.size[0]),Scalar(0,0,0),-1,8,0);*/

		timer.Tic();
		//detect body
		all_bbox = im_detect(net2, img_copy, box_pre);
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
			int front_symbol_l = 0;
			int front_symbol_r = 0;
			if (all_peaks[4].size() != 0 && all_peaks[3].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2 && all_peaks[3][2] > 0.2 && all_peaks[2][2] > 0.2){
				if (faces.size() == 0 || faces[0].score<0.9){
					if (((all_peaks[4][1]<=all_peaks[2][1]))){
						symbol_r = 1;
					}
				}
				else{
					//float angle_r = CalculateVectorAngle(all_peaks[2][0], all_peaks[2][1], all_peaks[3][0], all_peaks[3][1], all_peaks[4][0], all_peaks[4][1]);
					if (all_peaks[4][1] <= all_peaks[2][1]){
						/*if (angle_r > 120)*/front_symbol_r = 1;
					}
				}
			}
			if (all_peaks[7].size() != 0 && all_peaks[6].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2 && all_peaks[6][2] > 0.2 && all_peaks[5][2] > 0.2){
				if (faces.size() == 0 || faces[0].score<0.9){
					if ((all_peaks[7][1] <= all_peaks[5][1])){
						symbol_l = 1;
					}
				}
				else{
					//float angle_l = CalculateVectorAngle(all_peaks[5][0], all_peaks[5][1], all_peaks[6][0], all_peaks[6][1], all_peaks[7][0], all_peaks[7][1]);
					if (all_peaks[7][1] <= all_peaks[5][1]){
						/*if (angle_l > 120)*/front_symbol_l = 1;
					}
				}
			}
			if (front_symbol_l || front_symbol_r){  //front pointing
				teacher_info.pointing = true;
			}
			if (symbol_l || symbol_r)raising_time += 1;	
			else raising_time = 0;
						
			raising_total_time = raising_time;
			if (raising_total_time >= 4){       //back raising_time>=4   writing
				teacher_info.writing = true;
				teacher_info.pointing = false;
			}
			else{
				teacher_info.writing = false;
			}
			if (raising_total_time >= 1 && raising_total_time<4){  //back   1=<rasing_time<4  pointing
				teacher_info.pointing = true;
			}
			
			/*string output = "../output";
			char buff[300];
			sprintf(buff, "%s/%d.jpg", output.c_str(), n);
			cv::imwrite(buff, img);*/
			timer.Toc();
			cout << "Frame " << n << " cost " << timer.Elasped() / 1000.0 << "s" << endl;
			return teacher_info;
		}
	}
}

