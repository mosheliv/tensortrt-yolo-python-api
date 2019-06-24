# TensorRT api for the deepstream yolo implemetation

This is a work in progress, but it works for me... It is rough around the corners and lots of place for improvment - feel free to contribute:

Inference time with yolov2-tiny is 85ms/image (including resizing and NMS and decoding results) on 10W Nano. should be twice as fast on a MAXN mode Nano.


## Installing

1. sudo apt-get install libgflags-dev cmake
1. git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps
1. export YOLO_ROOT=\`pwd\`/deepstream_reference_apps/yolo
1. cd $YOLO_ROOT/apps/trt-yolo 
1. edit CMakeLists.txt and :
	1. change 'set(CMAKE_CXX_FLAGS_RELEASE "-O2")' to 'set(CMAKE_CXX_FLAGS_RELEASE "-O2 -fPIC")'
	2. add after it a line 'set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC" )'
1. mkdir build && cd build
1. cmake -D CMAKE_BUILD_TYPE=Release ..
as its read only. You can also chmod +w it
1. make && sudo make install
1. edit $YOLO-ROOT/config/yolov2-tiny.txt and change all the links to absolute paths (config_file_path, wts_file_path, labels_file_path, test_images)
1. check that the cpp app works ok by doing:
	1. cd $YOLO_ROOT/apps/trt-yolo/build
	1. put an image or two in $YOLO_ROOT/data/test_images.txt
	1. ./trt-yolo-app --flagfile=$YOLO_ROOT/config/yolov2-tiny.txt
1. cd $YOLO_ROOT/apps/trt-yolo/build/lib
1. git clone https://github.com/mosheliv/tensortrt-yolo-python-api
1. cd tensortrt-yolo-python-api
1. source link_shared.sh
1. python t.py --flagfile=$YOLO_ROOT/config/yolov2-tiny.txt your_image.jpg
