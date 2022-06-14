#include "AddLogo.h"


int main(int ac, char** av) {

	src = imread("src.png");
	logo = imread("logo1.png", IMREAD_UNCHANGED);

	if (src.cols == 3840)
		resize(src, src, Size(1920, 1080));

	srcGpu.upload(src);
	logoGpu.upload(logo);


	while (1) {
		MixLogo(srcGpu, logoGpu);
	}



	return 0;
}

void MixLogo(cuda::GpuMat img, cuda::GpuMat logo)
{
	auto t0 = std::chrono::system_clock::now();


	cuda::GpuMat mask, mask_temp, logo_temp, logo_dst, temp;
	cuda::GpuMat bgra[4];

	cuda::split(logo, bgra);
	cuda::threshold(bgra[3], mask_temp, 0, 255, THRESH_BINARY);

	cuda::cvtColor(mask_temp, mask, COLOR_GRAY2BGR);
	cuda::cvtColor(logo, logo_temp, COLOR_BGRA2BGR);

	logo_temp.copyTo(logo_dst, mask);

	cuda::subtract(img, mask, temp);
	cuda::add(temp, logo_dst, outputGpu);

	std::chrono::milliseconds delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t0);
	cout << "time: " << delta.count() << endl;

	Mat output;
	outputGpu.download(output);
	imshow("output", output);
	waitKey(1);



}
