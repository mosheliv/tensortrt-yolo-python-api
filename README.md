# Tensort api for the deepstream yolo implemetation

This is a work in progress, but it works for me... It is rough around the corners and lots of place for improvment - feel free to contribute:

Inference time with yolov2-tiny is 85ms/image (including resizing and NMS and decoding results) on 10W Nano. should be twice as fast on a MAXN mode Nano.


## Installing

1. git clone https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps
1. cd deepstream_reference_apps/yolo/apps/trt-yolo
1. mkdir build && cd build
1. cmake -D CMAKE_BUILD_TYPE=Release ..
1. edit deepstream_reference_apps/yolo/apps/trt-yolo/CMakeFiles/trt-yolo-app.dir/flags.make and add -fPIC to CXX_FLAGS
1. make && sudo make install
1. check that the cpp app works ok by doing:
	1. cd ../../../
	1. put an image or two in data/test_images.txt
	1. trt-yolo-app --flagfile=config/yolov2-tiny.txt
1. cd apps/trt-yolo/lib
1. git clone XXXXXXX
1. cd XXXXXX
1. source link_shared.sh
1. python t.py --flagfile=../../../../config/yolov2-tiny.txt 
