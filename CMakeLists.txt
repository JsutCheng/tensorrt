CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(seginterface)

set(CMAKE_CXX_FLAGS "-std=c++14 -O1")


link_directories(/usr/local/cuda/lib64)
# set(OpenCV_DIR "/opt/opencv/lib/cmake/opencv4")


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/home/user/TensorRT-7.2.3.4/include) #for NvInfer.h
link_directories(/home/user/TensorRT-7.2.3.4/lib/) #for cannot found -lnvinfer

add_executable(test test.cpp segcpp.cpp trt_dep.cpp)
target_include_directories(
    test PUBLIC ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    test -lnvinfer -lnvinfer_plugin -lnvparsers -lnvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    )
