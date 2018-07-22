#pragma once

#ifndef _DATASTRUCTS_H_
#define _DATAsTRUCTS_H_

#include "opencv2/core.hpp"

struct PatchModel
{
	int meu_patch_size = 5;
	cv::Point center;
	cv::Vec3d normal; 
	std::vector<std::shared_ptr<cv::Mat>> T_p; // other images where this patch is also visible
	std::shared_ptr<cv::Mat> referenceImage; // a pointer to the reference image.
};

struct Cell
{
	int image_index;
	int first_x;
	int first_y;
	int size;
	std::vector<cv::KeyPoint> key_points;
	cv::Mat features;
	std::vector<double> Q_t_depths;
	std::vector<std::shared_ptr<PatchModel>> Q_t; // a patch is stored here if the patch is truly visible in this cell
	std::vector<std::shared_ptr<PatchModel>> Q_f; // a patch is stored here if the patch is occluded by something but would otherwise be struc by a ray cast from the camera
};

struct ImageModel
{
	cv::Mat image;
	cv::Mat projectionMat;
	cv::Mat intrinsicParams;
	cv::Mat extrinsicParams;
	int beta_cell_size = 32;
	std::vector<std::shared_ptr<Cell>> cells;

	ImageModel(cv::Mat&& in_image, cv::Mat&& in_projMat,cv::Mat&& in_intrParam,cv::Mat&& in_extParam ) : 
		image(std::move(in_image)),projectionMat(std::move(in_projMat)),intrinsicParams(std::move(in_intrParam)), extrinsicParams(std::move(in_extParam)){};

};

#endif

