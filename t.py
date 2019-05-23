from ctypes import *
import  ctypes as C
import sys
import cv2
import time

lib=CDLL('./libyolo.so')
LP_c_char = POINTER(c_char)
LP_LP_c_char = POINTER(LP_c_char)

#lib.yoloConfigParserInit.argtypes = (c_int, # argc
#                        LP_LP_c_char) # argv

argc = len(sys.argv)
argv = (LP_c_char * (argc + 1))()
for i, arg in enumerate(sys.argv):
    enc_arg = arg.encode('utf-8')
    argv[i] = create_string_buffer(enc_arg)

#use for yoloV3
#lib.yolo3_init.restype = c_void_p
#s = lib.yolo3_init(argc, argv)
lib.yolo2_init.restype = c_void_p
s = lib.yolo2_init(argc, argv)

img = cv2.imread(sys.argv[-1])
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
ret_buf = create_string_buffer(1000)
lib.yolo_inference.argtypes = [c_void_p, c_uint, c_uint, c_void_p, c_char_p, c_uint]
for _ in range(10):
    t = time.time()
    res = lib.yolo_inference(s, img.shape[0], img.shape[1], img.ctypes.data_as(C.POINTER(C.c_ubyte)), ret_buf, 1000)
    for tt in ret_buf.value.split("|"):
        print(tt)
    print(time.time()-t)
#res = lib.yolo_inference(5)

