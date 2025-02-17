// Visual3DScanner.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <tuple>

#include "DataFile.h"
#include "DataStructs.h"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/viz.hpp"
#include "DetectFeatures.h"

void displayPntCloud(std::vector<cv::Point3d> pointCloud)
{
	cv::viz::Viz3d myWindow("Window");

	cv::viz::WCloud const  cloud(pointCloud);

	myWindow.showWidget("bunny",cloud);

	myWindow.spin();
}

int main()
{
	DataFile image_data{"D:\\Data\\dino multiview dataset","dino_par - Copy.txt"};

	std::vector<std::shared_ptr<ImageModel>> imageModels(image_data.getDataSize());

	for (int i = 0; i < image_data.getDataSize(); ++i)
	{
		auto [image, projMat, intrinsicMat, extrinsicMat] = image_data.getNext(i);
		imageModels[i] = std::make_shared<ImageModel>(std::move(image), std::move(projMat), std::move(intrinsicMat), std::move(extrinsicMat));
		MVS_calc::DetectFeatures(imageModels[i],i);
	}

	MVS_calc::MatchKeyPoints(0, 2, imageModels);

    return 0;
}

