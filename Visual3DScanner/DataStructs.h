#pragma once

#ifndef _DATASTRUCTS_H_
#define _DATAsTRUCTS_H_

#include "opencv2/core.hpp"

struct ImageStruct
{
	int index;
	cv::Mat Image;
	cv::Mat ProjectionMat;
	std::vector<cv::Point> featurePoints;
};

#endif

