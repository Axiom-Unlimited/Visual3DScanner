// Visual3DScanner.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <tuple>

#include "DataFile.h"
#include "DataStructs.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"

void displayPntCloud(std::vector<cv::Point3d> pointCloud)
{
	cv::viz::Viz3d myWindow("Window");

	cv::viz::WCloud cloud(pointCloud);

	myWindow.showWidget("bunny",cloud);

	myWindow.spin();
}

int main()
{
	DataFile image_data{"D:\\Data\\dino multiview dataset","dino_par.txt"};

	std::vector<std::shared_ptr<ImageModel>> imageModels(image_data.getDataSize());

	cv::Mat image, projMat, intrinsicMat, extrinsicMat;
	for (int i = 0; i < image_data.getDataSize(); ++i)
	{
		std::tie(image,projMat,intrinsicMat,extrinsicMat) = image_data.getNext(i);
		imageModels.push_back(std::make_shared<ImageModel>(std::move(image), std::move(projMat), std::move(intrinsicMat), std::move(extrinsicMat)));
	}


    return 0;
}

