#include "segcpp.hpp"


using nvinfer1::IHostMemory;
using nvinfer1::IBuilder;
using nvinfer1::INetworkDefinition;
using nvinfer1::ICudaEngine;
using nvinfer1::IInt8Calibrator;
using nvinfer1::IBuilderConfig;
using nvinfer1::IRuntime;
using nvinfer1::IExecutionContext;
using nvinfer1::ILogger;
using nvinfer1::Dims3;
using nvinfer1::Dims2;
using Severity = nvinfer1::ILogger::Severity;

using std::string;
using std::ios;
using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;
using std::array;

using cv::Mat;
using namespace cv;

SampleSegmentation::SampleSegmentation(const std::string& engineFilename)
    : mEngineFilename(engineFilename)
    , engine(nullptr)
{
    engine = deserialize(mEngineFilename);
}

vector<int> SampleSegmentation::infer(Mat frame) {
    Dims3 i_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    Dims3 o_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));
    const int iH{i_dims.d[2]}, iW{i_dims.d[3]};
    const int oH{o_dims.d[1]}, oW{o_dims.d[2]};

    // prepare image and resize
    // Mat im = cv::imread(imgfile);
    Mat im = frame;
    if (im.empty()) {
        cout << "cannot read image \n";
        std::abort();
    }
    // CHECK (!im.empty()) << "cannot read image \n";
    int orgH{im.rows}, orgW{im.cols};
    if ((orgH != iH) || orgW != iW) {
        cout << "resize orignal image of (" << orgH << "," << orgW 
            << ") to (" << iH << ", " << iW << ") according to model require\n";
        cv::resize(im, im, cv::Size(iW, iH), cv::INTER_CUBIC);
    }

    // normalize and convert to rgb
    // array<float, 3> mean{0.485f, 0.456f, 0.406f};
    // array<float, 3> variance{0.229f, 0.224f, 0.225f};
    array<float, 3> mean{0.3257f, 0.3690f, 0.3223f};
    array<float, 3> variance{0.2112f, 0.2148f, 0.2115f};
    float scale = 1.f / 255.f;
    for (int i{0}; i < 3; ++ i) {
        variance[i] = 1.f / variance[i];
    }
    vector<float> data(iH * iW * 3);
    for (int h{0}; h < iH; ++h) {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w < iW; ++w) {
            for (int c{0}; c < 3; ++c) {
                int idx = (2 - c) * iH * iW + h * iW + w; // to rgb order
                data[idx] = (p[w][c] * scale - mean[c]) * variance[c];
            }
        }
    }

    // call engine
    vector<int> res = infer_with_engine(engine, data);
    return res;
}


