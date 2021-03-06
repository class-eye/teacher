
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

//Json::Value root_all;
//void writeJson(Teacher_Info &teacher_info, int &n){
//	Json::Value root;
//	root["Frame"] = n;
//	root["Teacher's location"].append(teacher_info.location.x);
//	root["Teacher's location"].append(teacher_info.location.y);
//	for (int j = 0; j < 8; j++){
//		Json::Value part_loc;
//		int x = teacher_info.all_points[j].x;
//		int y = teacher_info.all_points[j].y;
//		part_loc.append(x);
//		part_loc.append(y);
//		root["Parts'location"].append(part_loc);
//	}
//	root["pointing"] = teacher_info.front_pointing || teacher_info.back_pointing;
//	root["teacher_in_screen"] = teacher_info.teacher_in_screen;
//	root_all.append(root);
//
//	ofstream out;
//	string jsonfile = "../output_json/" + videoname + ".json";
//	out.open(jsonfile);
//	Json::StyledWriter sw;
//	out << sw.write(root_all);
//	out.close();
//}

Teacher_analy::Teacher_analy(const string& pose_net, const string& pose_model, const string& rfcn_net, const string& rfcn_model,int gpu_device){
	if (gpu_device < 0) {
		caffe::SetMode(caffe::CPU, -1);
	}
	else {
		if (caffe::GPUAvailable()){
			caffe::SetMode(caffe::GPU, gpu_device);
		}
	}
	posenet = new caffe::Net(pose_net);
	posenet->CopyTrainedLayersFrom(pose_model);
	rfcnnet = new caffe::Net(rfcn_net);
	rfcnnet->CopyTrainedLayersFrom(rfcn_model);
}
Teacher_analy::~Teacher_analy(){
	delete posenet;
	delete rfcnnet;
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

Teacher_Info Teacher_analy::teacher_detect(jfda::JfdaDetector &detector, cv::Mat &img, Rect &box_pre, int &n){
	//cout << box_pre.x << " " << box_pre.y << " " << box_pre.width << " " << box_pre.height << endl;
	Teacher_Info teacher_info;
	Rect bbox_new;
	vector<Rect>all_bbox;
	Rect bbox_new1;
	Timer timer;


	//Mat img_copy;
	//img.copyTo(img_copy);
	//int height = 344 * 2 / 3;
	//cv::rectangle(img_copy, Point(0, height), Point(img.size[1], img.size[0]), Scalar(0, 0, 0), -1, 8, 0);
	//timer.Tic();
	//vector<vector<float>>all_peaks = pose_detect(*posenet, img_copy);
	//timer.Toc();
	//cout << "detect pose cost " << timer.Elasped() / 1000.0 << "s" << endl;
	//for (int i = 0; i < 8; i++){
	//	if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
	//		cv::circle(img, Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y), 3, cv::Scalar(0, 0, 255), -1);
	//		//string score = to_string(all_peaks[i][2]);
	//		//cv::putText(img, score, Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y), FONT_HERSHEY_DUPLEX,0.5, cv::Scalar(0, 0, 255));
	//	}
	//}	
	//string output = "../output";
	//char buff[300];
	//sprintf(buff, "%s/%d.jpg", output.c_str(), n);
	//cv::imwrite(buff, img);

#if 1
	timer.Tic();

	//Ϳ����������
	Mat img_copy;
	img.copyTo(img_copy);
	//int height = 344*2/3;
	int width = 38;
	cv::rectangle(img_copy, Point(0, 0), Point(width, img.size[0]), Scalar(0, 0, 0), -1, 8, 0);

	timer.Tic();
	//detect body
	if (no_teacher >= 5){
		box_pre = Rect(0, 0, img_copy.size().width, img_copy.size().height);
	}
	
	all_bbox = im_detect(*rfcnnet, img_copy, box_pre);

	//cv::rectangle(img, box_pre, cv::Scalar(0, 0, 255), 2);
	timer.Toc();
	cout << "detect body cost " << timer.Elasped() / 1000.0 << "s" << endl;


	if (all_bbox.size() > 1){
		teacher_info.num = all_bbox.size();
		/*for (int i = 0; i < all_bbox.size(); i++){
			Rect bbbox = all_bbox[i];		
			cv::rectangle(img, bbbox, cv::Scalar(0, 255, 0), 2);
		}
		cv::putText(img, to_string(all_bbox.size()), Point(100,100), FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 255));*/
		return teacher_info;
	}
	if (all_bbox.size() == 1){
		bbox_new = all_bbox[0];
		if (bbox_new.x != box_pre.x || bbox_new.y != box_pre.y || bbox_new.width != box_pre.width || bbox_new.height != box_pre.height){
			teacher_info.num = 1;
			cv::rectangle(img, bbox_new, cv::Scalar(0, 255, 0), 2);
			no_teacher = 0;
		}
		else {
			no_teacher++;
			teacher_info.num = 0;
		}

		//detect pose
		if (teacher_info.num == 1){
			bbox_new1.height = bbox_new.height;
			bbox_new1.width = int(bbox_new.height / 1.4);
			bbox_new1.x = bbox_new.x - (bbox_new1.width - bbox_new.width) / 2;
			bbox_new1.y = bbox_new.y;
			bbox_new1 = refine(img, bbox_new1);

			Mat im_body = img(bbox_new1);
			
			timer.Tic();
			vector<vector<float>>all_peaks = pose_detect(*posenet, im_body);
			timer.Toc();
			cout << "detect pose cost " << timer.Elasped() / 1000.0 << "s" << endl;
			if (all_peaks[1].size() != 0){
				teacher_info.location.x = all_peaks[1][0] + bbox_new1.x;
				teacher_info.location.y = all_peaks[1][1] + bbox_new1.y;
			}
			for (int i = 0; i < 8; i++){
				if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
					teacher_info.all_points.push_back(Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y));
					cv::circle(img, Point(all_peaks[i][0] + bbox_new1.x, all_peaks[i][1] + bbox_new1.y), 3, cv::Scalar(0, 0, 255), -1);
				}
				else{
					teacher_info.all_points.push_back(Point(0, 0));
				}
			}

			/*if (teacher_info.num == 0){
				for (int i = 0; i < 8; i++){
					if (all_peaks[i].size() != 0 && all_peaks[i][2] > 0.2){
						teacher_info.teacher_in_screen = true;
					}
				}
			}*/

			//detect face

			vector<FaceInfoInternal>facem;
			vector<FaceInfo> faces = detector.Detect(im_body, facem);
			if (faces.size() != 0 && faces[0].score > 0.9){
				string score = to_string(faces[0].score);
				cv::rectangle(img, Point(faces[0].bbox.x + bbox_new1.x, faces[0].bbox.y + bbox_new1.y), Point(faces[0].bbox.x + bbox_new1.x + faces[0].bbox.width, faces[0].bbox.y + bbox_new1.y + faces[0].bbox.height), cv::Scalar(0, 0, 255), 2);
				//cv::putText(img, score, Point(faces[0].bbox.x + bbox_new1.x, faces[0].bbox.y + bbox_new1.y - 15), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}

			//pose estimation
			int symbol_l_back = 0;
			int symbol_r_back = 0;
			int symbol_l_front = 0;
			int symbol_r_front = 0;
			int interaction_l = 0;
			int interaction_r = 0;
			if (have_face.size() == 5)have_face.erase(have_face.begin());
			if (faces.size() == 1 && faces[0].score > 0.9)have_face.push_back(1);
			if (faces.size() == 0 || (faces.size() == 1 && faces[0].score < 0.9))have_face.push_back(0);

			//----------------------------right----------------------------------------
			if (all_peaks[2].size() != 0 && all_peaks[5].size() != 0 && all_peaks[2][2] > 0.2 && all_peaks[5][2] > 0.2){
				if (all_peaks[2][0] > all_peaks[5][0]){
					if (all_peaks[4].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2&& all_peaks[2][2] > 0.2){
						if (all_peaks[4][1] <= all_peaks[2][1])symbol_r_back = 1;
					}
					if (all_peaks[4].size() != 0 && all_peaks[3].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2 && all_peaks[3][2] > 0.2 && all_peaks[2][2] > 0.2){
						if (all_peaks[4][1] <= (all_peaks[2][1] + all_peaks[3][1]) / 2)symbol_r_back = 1;
					}
				}
			}
			int count_ = count(have_face.begin(), have_face.end(), 0);
			//cv::putText(img, to_string(count_), Point(200,200), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			if (all_peaks[4].size() != 0 && all_peaks[3].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2 && all_peaks[3][2] > 0.2 && all_peaks[2][2] > 0.2){
				if ((all_peaks[4][1] <= (all_peaks[2][1] + all_peaks[3][1]) / 2) && count_ >= 2)interaction_r = 1;
			}
			if (all_peaks[4].size() != 0 && all_peaks[2].size() != 0 && all_peaks[4][2] > 0.2&& all_peaks[2][2] > 0.2){
				if ((all_peaks[4][1] <= all_peaks[2][1]) && count_ >= 2)interaction_r = 1;
			}

			//------------------------------left-----------------------------------------
			if (all_peaks[2].size() != 0 && all_peaks[5].size() != 0 && all_peaks[2][2] > 0.2 && all_peaks[5][2] > 0.2){
				if (all_peaks[2][0] > all_peaks[5][0]){
					if (all_peaks[7].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2&& all_peaks[5][2] > 0.2){
						if (all_peaks[7][1] <= all_peaks[5][1])symbol_l_back = 1;
					}
					if (all_peaks[7].size() != 0 && all_peaks[6].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2 && all_peaks[6][2] > 0.2 && all_peaks[5][2] > 0.2){
						if (all_peaks[7][1] <= (all_peaks[5][1] + all_peaks[6][1]) / 2)symbol_l_back = 1;
					}
				}
			}
			
			if (all_peaks[7].size() != 0 && all_peaks[6].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2 && all_peaks[6][2] > 0.2 && all_peaks[5][2] > 0.2){
				if ((all_peaks[7][1] <= (all_peaks[5][1] + all_peaks[6][1]) / 2) && count_ >= 2)interaction_l = 1;
			}
			if (all_peaks[7].size() != 0 && all_peaks[5].size() != 0 && all_peaks[7][2] > 0.2&& all_peaks[5][2] > 0.2){
				if ((all_peaks[7][1] <= all_peaks[5][1]) && count_ >= 2)interaction_l = 1;
			}

			//------------------------------------------------------------------------
			if (symbol_l_front || symbol_r_front || symbol_l_back || symbol_r_back){
				teacher_info.back_pointing = true;
			}
			if (symbol_l_back || symbol_r_back){
				raising_time += 1;
			}
			else raising_time = 0;
			raising_total_time = raising_time;

			//----------------------------------------------------------------------
		
			if (interaction_l || interaction_r){
				teacher_info.interaction = true;
				teacher_info.back_pointing = false;
			}
			//------------------------------------------------------------------------
			//char buf[300];
			/*sprintf(buf, "interaction_l:%d  interaction_r:%d", interaction_l, interaction_r);
			cv::putText(img, buf, Point(100, 100), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255));*/
			/*sprintf(buf, "%d/%d", count_, have_face.size());
			cv::putText(img, buf, Point(100, 200), FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255));*/

			if (raising_total_time >= 4){       //back raising_time>=4   writing
				teacher_info.writing = true;
				teacher_info.back_pointing = false;
				teacher_info.interaction = false;
			}
			else{
				teacher_info.writing = false;
			}
			//if (raising_total_time >= 1 && raising_total_time<4){  //back   1=<rasing_time<4  pointing
			//	teacher_info.back_pointing = true;
			//}

			//writeJson(teacher_info,n);

			if (teacher_info.writing){
				string status = "writing+focus";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2 - 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}
			if (teacher_info.interaction){
				string status = "interaction+focus";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2 + 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}
			/*if (teacher_info.front_pointing){
				string status = "pointing";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
				}*/
			/*if (teacher_info.back_pointing){
				string status = "pointing";
				cv::putText(img, status, Point(img.size[1] / 2, img.size[0] / 2), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
			}*/
			
		}
	}
	timer.Toc();
	cout << "Frame " << n << " cost " << timer.Elasped() / 1000.0 << "s" << endl;

	string output = "../output";
	char buff[300];
	sprintf(buff, "%s/%d.jpg", output.c_str(), n);
	cv::imwrite(buff, img);
	return teacher_info;
#endif
}

