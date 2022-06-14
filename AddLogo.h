#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>

#include <iostream>
#include <time.h>


using namespace cv;
using namespace std;


Mat src, logo;
cuda::GpuMat srcGpu, logoGpu, outputGpu;

void MixLogo(cuda::GpuMat img, cuda::GpuMat logo);


class AddLogo {

	//cuda::GpuMat src, logo;
public:
	


	

};