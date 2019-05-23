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

extern "C" int yolo_inference(void* engine, unsigned int rows, unsigned int cols, unsigned char* image)
//extern "C" int yolo_inference(int t)
{
        struct timeval inferStart, inferEnd;
	double inferElapsed = 0;
	unsigned long ts;
	unsigned long ots;
	gettimeofday(&inferStart, NULL);

	ots = inferStart.tv_sec+inferStart.tv_usec;
	std::cout << "in inference" << "\n";
	cv::Mat img(rows, cols, CV_8UC3, (void *) image);
	cv::Mat blob;
	std::vector<cv::Mat> letterboxStack(1);

	std::cout << "engine ptr is " << (long) engine << "\n";
	std::cout << "rows is " << rows << "\n";
	std::cout << "cols is " << cols << "\n";

	gettimeofday(&inferStart, NULL);
	ts = inferStart.tv_sec+inferStart.tv_usec;
	std::cout << "#1 " << ts-ots <<"\n";
	ots = ts;

        float dim = std::max(rows, cols);
        int resizeH = ((rows / dim) * 416);
	std::cout << "resizeH is " << resizeH << "\n";
        int resizeW = ((cols / dim) * 416);
	std::cout << "resizeW is " << resizeW << "\n";
        float ScalingFactor = static_cast<float>(resizeH) / static_cast<float>(rows);
	std::cout << "scaling factor  is " << ScalingFactor << "\n";
        

        // Additional checks for images with non even dims
        if ((416 - resizeW) % 2) resizeW--;
        if ((416 - resizeH) % 2) resizeH--;

        int m_XOffset = (416 - resizeW) / 2;
        int m_YOffset = (416 - resizeH) / 2;
	std::cout << "m_XOffset is " << m_XOffset << "\n";
	std::cout << "m_YOffset is " << m_YOffset << "\n";

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
	gettimeofday(&inferStart, NULL);
	ts = inferStart.tv_sec+inferStart.tv_usec;
	std::cout << "#2 " << ts-ots <<"\n";
	ots = ts;
        Yolo* inferNet = engine;

        gettimeofday(&inferStart, NULL);
        inferNet->doInference(blob.data, 1);
        gettimeofday(&inferEnd, NULL);
        inferElapsed += ((inferEnd.tv_sec - inferStart.tv_sec)
                         + (inferEnd.tv_usec - inferStart.tv_usec) / 1000000.0)
            * 1000;
	std::cout << "infer time in cpp " << inferElapsed;
	gettimeofday(&inferStart, NULL);
	ts = inferStart.tv_sec+inferStart.tv_usec;
	std::cout << "#3 " << ts-ots <<"\n";
	ots = ts;
        auto binfo = inferNet->decodeDetections(0, rows, cols);
	std::cout << "after decode" << "\n";
	std::cout << "nms th " << inferNet->getNMSThresh() << "\n";
	std::cout << "num classes " << inferNet->getNumClasses() << "\n";
        auto remaining
                    = nmsAllClasses(inferNet->getNMSThresh(), binfo, inferNet->getNumClasses());
        for (auto b : remaining)
        {
		    std::cout << "in loop" << "\n";
                    printPredictions(b, inferNet->getClassName(b.label));
        }
	gettimeofday(&inferStart, NULL);
	ts = inferStart.tv_sec+inferStart.tv_usec;
	std::cout << "#4 " << ts-ots <<"\n";
	ots = ts;
	return 1;
}

	

