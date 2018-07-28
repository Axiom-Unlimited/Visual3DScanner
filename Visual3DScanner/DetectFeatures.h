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
	/**
	 * @param imgModel : the ImageModel to get sift feature for
	 * @param img_idx : the index of the image model of interest
	 * @param numOfFeatures : the number of max number of feature per cell to detect
	 */
	static void DetectFeatures(std::shared_ptr<ImageModel> imgModel,int img_idx,int numOfFeatures = 4)
	{
		// by default cell size is set to 32 
		auto const cellSize = imgModel->beta_cell_size; 
		auto const xNumOfShift = imgModel->image.cols / cellSize;
		auto const yNumOfShift = imgModel->image.rows / cellSize;

		auto sift = cv::xfeatures2d::SIFT::create(numOfFeatures);

		//nested for loop to calculate features for each cell in the image
		for (int rows = 0; rows < yNumOfShift; ++rows)
		{
			for (int cols = 0; cols < xNumOfShift; ++cols)
			{
				// calculate the cell shift
				auto const xPos = cellSize * cols;
				auto const yPos = cellSize * rows;

				// get a referene to the cell
				auto const cellRef = imgModel->image(cv::Rect(xPos,yPos,cellSize,cellSize));

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


	/**
	 * @brief : this function is taken from "Accurate, Dense, and Robust Multi-View Stereopsis"
	 * by Yasutaka Furukawa and Jean Ponce
	 * @param index : image index in the data
	 * @param numOfImages2Match : the total number of images in S(p)
	 * @param image_Models : reference to the vector of image models
	 * @param alpha_0 : the first photometric consistancy threshold
	 * @param alpha_1 : the second photometric consistancy threshold 
	 */
	static void MatchKeyPoints(int index,int numOfImages2Match, std::vector<std::shared_ptr<ImageModel>>& image_Models, double alpha_0 = 0.4,double alpha_1 = 0.7)
	{
		// Initial sparse set of patches P will go here
		std::vector<cv::Mat> P;
		// for each image I with optical center O
		for (int imgIdx = 0; imgIdx < image_Models.size()-numOfImages2Match; ++imgIdx)
		{
			cv::Point2f O(image_Models[imgIdx]->intrinsicParams.at<double>(0,2), image_Models[imgIdx]->intrinsicParams.at<double>(1, 2));

			// for each feature cell in I
			for (int cellIdx = 0; cellIdx < image_Models[imgIdx]->cells.size(); ++cellIdx)
			{
				// use normalized cross correlation to get photo consistance of the image patch in the following images
				auto photoConsistancies = Utils::calcPhotometricConsistency(
											image_Models[imgIdx]->image(cv::Rect(image_Models[imgIdx]->cells[cellIdx]->first_x
												, image_Models[imgIdx]->cells[cellIdx]->first_y
												, image_Models[imgIdx]->cells[cellIdx]->size
												, image_Models[imgIdx]->cells[cellIdx]->size))
											, image_Models.begin() + imgIdx + 1
											, image_Models.begin() + imgIdx + 1 + numOfImages2Match);
				
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

					// sort F in an increasing order of distance from O
					std::map<double, std::tuple<cv::Point2f, int, int>> F_prime;

					for (auto f : F)
					{
						auto dist = cv::norm(std::get<0>(f) - O);
						F_prime.insert(std::make_pair(dist,f));
					}

					for (auto f_prime : F_prime)
					{
						auto R_p = image_Models[imgIdx];
						// add values to T(p) that pass the photometric constraint
						std::vector<std::shared_ptr<ImageModel>> T_p;
						for (int i = 0; i < photoConsistancies.size(); ++i)
						{
							if (photoConsistancies[i] >= alpha_0)
							{
								T_p.push_back(*(image_Models.begin() + imgIdx + 1 + i));
							}
						}

						// get initial calculation for c(p)
						auto const projMat1 = image_Models[imgIdx]->projectionMat;
						auto const projMat2 = image_Models[std::get<2>(f_prime.second)]->projectionMat;
						cv::Mat const projPnt1 = (cv::Mat_<double>(2,1) << image_Models[imgIdx]->cells[cellIdx]->key_points[ptCellIdx].pt.x
																	, image_Models[imgIdx]->cells[cellIdx]->key_points[ptCellIdx].pt.y);
						cv::Mat const projPnt2 = (cv::Mat_<double>(2,1) << std::get<0>(f_prime.second).x, std::get<0>(f_prime.second).y);
						cv::Mat c_p; // should be a 4x1 mat
						cv::triangulatePoints(projMat1, projMat2, projPnt1, projPnt2, c_p);
						// initial calculation for n(p)
						cv::Mat const O_4x1 = (cv::Mat_<double>(4, 1) << O.x,O.y,1,1);
						cv::Mat n_p = c_p - O_4x1;

					}
				}
	
			}
		}
	}

	
}
#endif

