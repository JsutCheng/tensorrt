#ifndef _SEG_CPP_
#define _SEG_CPP_
#include <opencv2/opencv.hpp>
#include "trt_dep.hpp"

using cv::Mat;


class SampleSegmentation
{

public:
    SampleSegmentation(const std::string& engineFilename);
    vector<int> infer(Mat frame);

private:
    std::string mEngineFilename;
    TrtSharedEnginePtr engine; //!< The TensorRT engine used to run the network
};

#endif
