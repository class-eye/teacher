
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
#ifdef __unix__
#include <json/json.h>
//#include <python2.7/Python.h>
#endif

using namespace cv;
using namespace std;
using namespace caffe;

Json::Value root_all;
void writeJson(Teacher_Info &teacher_info, int &n){
	Json::Value root;
	root["Frame"] = n;
	root["Teacher's location"].append(teacher_info.location.x);
	root["Teacher's location"].append(teacher_info.location.y);
	for (int j = 0; j < 8; j++){
		Json::Value part_loc;
		int x = teacher_info.all_points[j].x;
		int y = teacher_info.all_points[j].y;
		part_loc.append(x);
		part_loc.append(y);
		root["Parts'location"].append(part_loc);
	}
	root["pointing"] = teacher_info.front_pointing || teacher_info.back_pointing;
	root["teacher_in_screen"] = teacher_info.teacher_in_screen;
	root_all.append(root);

	ofstream out;
	string jsonfile = "../output_json/" + videoname + ".json";
	out.open(jsonfile);
	Json::StyledWriter sw;
	out << sw.write(root_all);
	out.close();

}
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
int raising_time = 0;
int raising_total_time = 0;
Teacher_Info teacher_detect(Net &net1, Net &net2, jfda::JfdaDetector &detector,cv::Mat &img, Rect &box_pre, int &n){
	//cout << box_pre.x << " " << box_pre.y << " " << box_pre.width << " " << box_pre.height << endl;
	Teacher_Info teacher_info;
	Rect bbox_new;
	vector<Rect>all_bbox;
	Rect bbox_new1;
	Timer timer;
	
	if (n % 25 == 0){
		//Í¿µô²¿·ÖÇøÓò
		Mat img_copy;
		img.copyTo(img_copy);
		int height = 670*2/3;
		cv::rectangle(img_copy, Point(0, height), Point(img.size[1], img.size[0]), Scalar(0, 0, 0), -1, 8, 0);

		timer.Tic();
		//detect body
		
		all_bbox = im_detect(net2, img_copy, box_pre);
		
		//cv::rectangle(img, box_pre, cv::Scalar(0, 0, 255), 2);
		/*timer.Toc();
		cout << "detect body cost " << timer.Elasped() / 1000.0 << "s" << endl;*/
		if (all_bbox.size() > 1){
			teacher_info.num = all_bbox.size();
			return teacher_info;
		}
		if (all_bbox.size() == 1){
			bbox_new = all_bbox[0];
			if (bbox_new.x != box_pre.x || bbox_new.y != box_pre.y || bbox_new.width != box_pre.width || bbox_new.height != box_pre.height){
				teacher_info.num = 1;
				cv::rectangle(img, bbox_new, cv::Scalar(0, 255, 0), 2);
			}
			else teacher_info.num = 0;
		
			//detect pose
			bbox_new1.height = bbox_new.height;
			bbox_new1.width = int(bbox_new.height * 1);
			bbox_new1.x = bbox_new.x - (bbox_new1.width - bbox_new.width) / 2;
			bbox_new1.y = bbox_new.y;
			bbox_new1=refine(img, bbox_new1);
			Mat im_body = img(bbox_new1);
			vector<vector<float>>all_peaks = pose_detect(net1, im_body);
			if (all_peaks[1].size() != 0){
				teacher_info.location.x = all_peaks[1][0] + bbox_new1.x;
				teacher_info.location.y = all_peaks[1][1] + bbox_new1.y;
			}
			for (int i = 0; i < 8; i++){
				if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
					teacher_info.all_points.push_back(Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y));
					cv::circle(img, Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y), 4, cv::Scalar(0, 0, 255), -1);
				}
				else{
					teacher_info.all_points.push_back(Point(0,0));
				}
			}

			if (teacher_info.num == 0){
				for (int i = 0; i < 8; i++){
					if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
						teacher_info.teacher_in_screen = true;
					}
				}
			}

			//detect face

			vector<FaceInfoInternal>facem;
			vector<FaceInfo> faces = detector.Detect(im_body, facem);
			if (faces.size() != 0 && faces[0].score>0.9){
				string score = to_string(faces[0].score);
				cv::rectangle(img, Point(faces[0].bbox.x + bbox_new1.x, faces[0].bbox.y + bbox_new1.y), Point(faces[0].bbox.x + bbox_new1.x + faces[0].bbox.width, faces[0].bbox.y + bbox_new1.y + faces[0].bbox.height), cv::Scalar(0, 0, 255), 2);
				//cv::putText(img, score, Point(faces[0].bbox.x + bbox_new1.x, faces[0].bbox.y + bbox_new1.y-15), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}

			//pose estimation
			int symbol_l = 0;
			int symbol_r = 0;
			int front_symbol_l = 0;
			int front_symbol_r = 0;
			if (all_peaks[4].size() != 0 && all_peaks[3].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2 && all_peaks[3][2] > 0.2 && all_peaks[2][2] > 0.2){
				if (faces.size() == 0 || faces[0].score<0.9){
					if (all_peaks[4][1]<=all_peaks[2][1])symbol_r = 1;
					if (all_peaks[4][1] <= (all_peaks[2][1] + all_peaks[3][1]) / 2)symbol_r = 1;
				}
				else{
					//float angle_r = CalculateVectorAngle(all_peaks[2][0], all_peaks[2][1], all_peaks[3][0], all_peaks[3][1], all_peaks[4][0], all_peaks[4][1]);
					if (all_peaks[4][1] <= all_peaks[2][1])front_symbol_r = 1;			
				}
			}
			if (all_peaks[7].size() != 0 && all_peaks[6].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2 && all_peaks[6][2] > 0.2 && all_peaks[5][2] > 0.2){
				if (faces.size() == 0 || faces[0].score<0.9){
					if (all_peaks[7][1] <= all_peaks[5][1])symbol_l = 1;
					if (all_peaks[7][1] <= (all_peaks[5][1] + all_peaks[6][1]) / 2)symbol_l = 1;		
				}
				else{
					//float angle_l = CalculateVectorAngle(all_peaks[5][0], all_peaks[5][1], all_peaks[6][0], all_peaks[6][1], all_peaks[7][0], all_peaks[7][1]);
					if (all_peaks[7][1] <= all_peaks[5][1])front_symbol_l = 1;
				}
			}
			if (front_symbol_l || front_symbol_r){  //front pointing
				teacher_info.front_pointing = true;
			}
			if (symbol_l || symbol_r){
				teacher_info.back_pointing = true;
			}
			//if (symbol_l || symbol_r)raising_time += 1;	
			//else raising_time = 0;
			//			
			//raising_total_time = raising_time;
			//if (raising_total_time >= 4){       //back raising_time>=4   writing
			//	teacher_info.writing = true;
			//	teacher_info.back_pointing = false;
			//}
			//else{
			//	teacher_info.writing = false;
			//}
			//if (raising_total_time >= 1 && raising_total_time<4){  //back   1=<rasing_time<4  pointing
			//	teacher_info.back_pointing = true;
			//}

			/*writeJson(teacher_info,n);


			if (teacher_info.front_pointing){
				string status = "pointing";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}
			if (teacher_info.back_pointing){
				string status = "pointing";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}*/
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

