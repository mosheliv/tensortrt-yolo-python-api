#include "ds_image.h"
#include "trt_utils.h"
#include "yolo.h"
#include "yolo_config_parser.h"
#include "yolov2.h"
#include "yolov3.h"

#include <experimental/filesystem>
#include <fstream>
#include <string>
#include <sys/time.h>

char stat_s[] = "this is a string";



extern "C" void* yolo2_tiny_init(int argc, char* argv[])
{
    std::cout << "in init";
    // Flag set in the command line overrides the value in the flagfile
    gflags::SetUsageMessage(
        "Usage : trt-yolo-app --flagfile=</path/to/config_file.txt> --<flag>=value ...");

    // parse config params
    yoloConfigParserInit(argc, argv);
    NetworkInfo yoloInfo = getYoloNetworkInfo();
    InferParams yoloInferParams = getYoloInferParams();
    uint64_t seed = getSeed();
    std::string networkType = getNetworkType();
    std::string precision = getPrecision();
    Yolo* inferNet = new YoloV2(1, yoloInfo, yoloInferParams);
    return (void*) inferNet;
}

extern "C" int yolo_inference(Yolo* engine, unsigned int rows, unsigned int cols, unsigned char* image, char* result_buf)
//extern "C" int yolo_inference(int t)
{
	cv::Mat img(rows, cols, CV_8UC3, (void *) image);
	cv::Mat blob;
	std::vector<cv::Mat> letterboxStack(1);

        float dim = std::max(rows, cols);
        int resizeH = ((rows / dim) * 416);
        int resizeW = ((cols / dim) * 416);
        float ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(rows);
        

        // Additional checks for images with non even dims
        if ((416 - resizeW) % 2) resizeW--;
        if ((416 - resizeH) % 2) resizeH--;

        int m_XOffset = (416 - resizeW) / 2;
        int m_YOffset = (416 - resizeH) / 2;

        // resizing
        cv::resize(img, img, cv::Size(resizeW, resizeH), 0, 0, cv::INTER_CUBIC);
        // letterboxing
        cv::copyMakeBorder(img, img, m_YOffset, m_YOffset, m_XOffset,
                       m_XOffset, cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
//	cv::imshow("t", img);
//	cv::waitKey(0);

        // converting to RGB
        cv::cvtColor(img, img, CV_BGR2RGB);


	letterboxStack[0] = img;
	blob = cv::dnn::blobFromImages(letterboxStack, 1.0, cv::Size(416, 416),
                                   cv::Scalar(0.0, 0.0, 0.0), false, false);
        Yolo* inferNet = engine;

        inferNet->doInference(blob.data, 1);

        auto binfo = inferNet->decodeDetections(0, rows, cols);
        auto remaining
                    = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
	int res_ptr = 0;

	char tmp_res[1000];

        for (auto b : remaining)
        {
		char *cn = &inferNet->getClassName(b.label)[0];
		sprintf(&tmp_res[0], "%s,%d,%f,%d,%d,%d,%d|",
			cn,
			b.label,
			b.prob,
			(int)b.box.x1,
			(int)b.box.y1,
			(int)b.box.x2,
			(int)b.box.y2
		);
		strcpy(&result_buf[res_ptr], tmp_res);
		res_ptr += strlen(tmp_res);

        }
	return 1;
}

	

