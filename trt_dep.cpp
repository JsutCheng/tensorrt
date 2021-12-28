
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <chrono>

#include "trt_dep.hpp"


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


Logger gLogger;


TrtSharedEnginePtr shared_engine_ptr(ICudaEngine* ptr) {
    return TrtSharedEnginePtr(ptr, TrtDeleter());
}


TrtSharedEnginePtr deserialize(string serpth) {

    ifstream ifile(serpth, ios::in | ios::binary);
    if (!ifile) {
        cout << "read serialized file failed\n";
        std::abort();
    }

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    auto runtime = TrtUniquePtr<IRuntime>(nvinfer1::createInferRuntime(gLogger));
    TrtSharedEnginePtr engine = shared_engine_ptr(
            runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr));
    return engine;
}


vector<int> infer_with_engine(TrtSharedEnginePtr engine, vector<float>& data) {
    Dims3 out_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));

    const int batchsize{1}, H{out_dims.d[1]}, W{out_dims.d[2]};
    const int in_size{static_cast<int>(data.size())};
    const int out_size{batchsize * H * W};
    vector<void*> buffs(2);
    vector<int> res(out_size);

    auto context = TrtUniquePtr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        cout << "create execution context failed\n";
        std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[1], out_size * sizeof(int));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    cudaStream_t stream;
    state = cudaStreamCreate(&stream);
    if (state) {
        cout << "create stream failed\n";
        std::abort();
    }

    state = cudaMemcpyAsync(
            buffs[0], &data[0], in_size * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    if (state) {
        cout << "transmit to device failed\n";
        std::abort();
    }
    context->enqueueV2(&buffs[0], stream, nullptr);
    // context->enqueue(1, &buffs[0], stream, nullptr);
    state = cudaMemcpyAsync(
            &res[0], buffs[1], out_size * sizeof(int), 
            cudaMemcpyDeviceToHost, stream);
    if (state) {
        cout << "transmit to host failed \n";
        std::abort();
    }
    cudaStreamSynchronize(stream);

    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaStreamDestroy(stream);

    return res;
}


