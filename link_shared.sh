g++ -fPIC -c -I../../../../../lib -I/usr/local/cuda/include -fpermissive yolo_api.cpp
ar -x ../libyolo-lib.a 
g++ -shared *.o -o libyolo.so -L/usr/local/lib  -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_dnn -lopencv_imgcodecs -lnvinfer -lgflags -L/usr/local/cuda/lib64 -lcudart -lcudnn -lcublas -lstdc++fs -ldl -lnvinfer_plugin

