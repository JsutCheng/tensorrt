#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "segcpp.hpp"
#include <map>

using namespace std;
using namespace cv;

vector<vector<uint8_t>> get_color_map();
Mat get_by_id(vector<int>, int id);

int main(int argc, const char** argv)
{
	vector<vector<uint8_t>> color_map = get_color_map();
	SampleSegmentation sample("/home/user/project/tensorRT/model.trt");
	Mat frame = imread("/home/user/project/tensorRT/testdata/test_image.png");
	if (frame.empty())
		return 0;
	cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
	vector<int> res = sample.infer(frame);

	//可以注释这部分，把下面一行注释去掉根据id获取对应标签的mask
	int oW = 1024;
	int oH = 1024;
	Mat pred(1024, 1024, CV_8UC3);
	int idx{0};

	for (int i{0}; i < oH; ++i) {
		uint8_t *ptr = pred.ptr<uint8_t>(i);
		for (int j{0}; j < oW; ++j) {
			ptr[0] = color_map[res[idx]][0];
			ptr[1] = color_map[res[idx]][1];
			ptr[2] = color_map[res[idx]][2];
			ptr += 3;
			++ idx;

		}
	}

	// Mat pred = get_by_id(res, 1);
	imshow("original image", frame);
	imshow("predict image", pred);
	waitKey(0);
	return 0;
}

vector<vector<uint8_t>> get_color_map() {
    vector<vector<uint8_t>> color_map(256, vector<uint8_t>(3));
    std::minstd_rand rand_eng(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (int i{0}; i < 256; ++i) {
        for (int j{0}; j < 3; ++j) {
            color_map[i][j] = u(rand_eng);
        }
    }
    return color_map;
}

Mat get_by_id(vector<int> res, int id){
	int oW = 1024;
	int oH = 1024;
	
	map<int, Mat> mp;
	for(int i{0}; i < 11; i++){
		mp[i] = Mat(1024, 1024, CV_8UC1, Scalar(0));
	}
	int idx{0};
	for (int i{0}; i < oH; ++i) {
		for (int j{0}; j < oW; ++j) {
			mp[res[idx]].ptr<uchar>(i)[j] = 255;
			++ idx;
		}
	}
	return mp[id];
}