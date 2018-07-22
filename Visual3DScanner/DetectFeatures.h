#pragma once
#ifndef _DETECT_FEATURES_H_
#define _DETECT_FEATURES_H_
#include "DataStructs.h"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "opencv2/features2d.hpp"
#include <map>
#include "CommonUtils.h"

namespace MVS_calc
{
	static void DetectFeatures(std::shared_ptr<ImageModel> imgModel,int img_idx,int numOfFeatures = 4)
	{
		// by default cell size is set to 32 
		const auto cellSize = imgModel->beta_cell_size; 
		const auto xNumOfShift = imgModel->image.cols / cellSize;
		const auto yNumOfShift = imgModel->image.rows / cellSize;

		auto sift = cv::xfeatures2d::SIFT::create(numOfFeatures);

		//nested for loop to calculate features for each cell in the image
		for (int rows = 0; rows < yNumOfShift; ++rows)
		{
			for (int cols = 0; cols < xNumOfShift; ++cols)
			{
				// calculate the cell shift
				const auto xPos = cellSize * cols;
				const auto yPos = cellSize * rows;

				// get a referene to the cell
				cv::Mat cellRef = imgModel->image(cv::Rect(xPos,yPos,cellSize,cellSize));

				// calculate the keypoints and features
				std::vector<cv::KeyPoint> key_points;
				cv::Mat features;
				sift->detect(cellRef, key_points);
				sift->compute(cellRef, key_points, features);

				// if there is at least one key point then add it to cell
				if (!key_points.empty())
				{
					auto cell = std::make_shared<Cell>();
					cell->image_index = img_idx;
					cell->first_x = xPos;
					cell->first_y = yPos;
					cell->size = cellSize;
					cell->key_points = key_points;
					cell->features = features;
					imgModel->cells.push_back(cell);
				}
			}
		}
	}



	static void MatchKeyPoints(int index,int numOfImages2Match, std::vector<std::shared_ptr<ImageModel>>& image_Models)
	{
		// Initial sparse set of patches P will go here

		// for each image I with optical center O
		for (int imgIdx = 0; imgIdx < image_Models.size()-numOfImages2Match; ++imgIdx)
		{
			// for each feature f detected in I 
			for (int cellIdx = 0; cellIdx < image_Models[imgIdx]->cells.size(); ++cellIdx)
			{
				// calculate F features for the image at imgIdx for each feature point in the image
				for (auto ptCellIdx = 0; ptCellIdx < image_Models[imgIdx]->cells[cellIdx]->key_points.size(); ptCellIdx++)
				{
					auto it = image_Models.begin();
					auto F = Utils::getCorrespondingFeaturesSetF(
						ptCellIdx
						, image_Models[imgIdx]->cells[cellIdx]
						, image_Models[imgIdx]->intrinsicParams
						, image_Models[imgIdx]->extrinsicParams
						, image_Models.begin() + imgIdx + 1
						, image_Models.begin() + imgIdx + 1 + numOfImages2Match);

				}
	
			}
		}
		
		
	}

	
}
#endif

