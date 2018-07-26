#pragma once
#ifndef _COMMON_UTILS_H_
#define _COMMON_UTILS_H_
#include <opencv2/core/types.hpp>
#include "opencv2/imgproc.hpp"
#include "DataStructs.h"
#include <set>

namespace Utils
{

	namespace std_transform_functions
	{
		cv::Point getPoint(cv::KeyPoint keypoint) { return keypoint.pt; }
	}

	/*
	 * this function is taken from "Accurate, Dense, and Robust Multi-View Stereopsis"
	 * by Yasutaka Furukawa and Jean Ponce
	 * @param refImgCell a cell from the current image of interest
	 * @param begin iterator to the first image to compare to the refImgCell
	 * @param end iterator to the last image
	 * @return std::vector<std::tuple<cv::Point2f,int,int>> : the point coordinates, cell index, image index
	 */
	static std::vector<std::tuple<cv::Point2f, int, int>>  getCorrespondingFeaturesSetF(
		int pointIndex
		, std::shared_ptr<Cell>& refImgCell
		, cv::Mat& refImg_cameraMat
		, cv::Mat& refImg_homogRotTrans
		, std::vector<std::shared_ptr<ImageModel>>::iterator begin
		, std::vector<std::shared_ptr<ImageModel>>::iterator  end
		, int FSetSize = 4)
	{
		
		std::map<double, std::tuple<cv::Point2f, int, int>> temp;

		const auto tmpRefPnt = refImgCell->key_points[pointIndex];
		cv::Mat q = (cv::Mat_<double>(3, 1) << tmpRefPnt.pt.x, tmpRefPnt.pt.y, 1);

		// iterate over the image models to find the the best features;
		for (auto img_it = begin; img_it != end; ++img_it)
		{
			const auto numOfCells = (*img_it)->cells.size();
			//calculate rotation of the image of interest to the next image==============================
			cv::Mat comparatorImgTransform =  (*img_it)->extrinsicParams.clone();
			//flip rotation 
			cv::Mat newCompRot = comparatorImgTransform(cv::Rect(0,0,3,3));
			newCompRot = newCompRot.t();
			//flip translation
			cv::Mat newCompTrans = comparatorImgTransform(cv::Rect(3,0,1,3));
			newCompTrans = -(newCompRot*newCompTrans);
			// recombine into comparatorImgTransform
			newCompRot.copyTo(comparatorImgTransform(cv::Rect(0, 0, 3, 3)));
			newCompTrans.copyTo(comparatorImgTransform(cv::Rect(3, 0, 1, 3)));
			// calculate transfrom to next image
			const cv::Mat transform2NxtImg = refImg_homogRotTrans * comparatorImgTransform;
			const cv::Mat rot2NxtImg = transform2NxtImg(cv::Rect(0,0,3,3));
			cv::Mat trans2NxtImg = transform2NxtImg(cv::Rect(3,0,1,3));
			const cv::Mat skew = (cv::Mat_<double>(3, 3) << 0								,-trans2NxtImg.at<double>(2,0)	, trans2NxtImg.at<double>(1,0)
															, trans2NxtImg.at<double>(2, 0)	,0								, -trans2NxtImg.at<double>(0,0)
															, -trans2NxtImg.at<double>(1,0)	,trans2NxtImg.at<double>(0,0)	, 0);

			// compute fundamental matrix F = inv(transpose(Mr))*R*S*inv(Ml)
			cv::Mat fundMat = refImg_cameraMat.t()*rot2NxtImg*skew*(*img_it)->intrinsicParams.inv();
			fundMat = fundMat.inv();

			// this is where the epipolar constraint is tested.
			for (int cell_idx = 0; cell_idx < numOfCells; ++cell_idx)
			{
				for (int pnt_idx = 0; pnt_idx < (*img_it)->cells[cell_idx]->key_points.size(); ++pnt_idx)
				{
					const auto tmp_pnt = (*img_it)->cells[cell_idx]->key_points[pnt_idx];
					const cv::Mat p = (cv::Mat_<double>(3,1) << tmp_pnt.pt.x, tmp_pnt.pt.y, 1);

					const cv::Mat v = q.t() * fundMat; // v = [a,b,c]
					auto dist_v = std::abs(v.dot(p)) / std::sqrt(std::pow(v.at<double>(0, 0), 2) + std::pow(v.at<double>(0, 1), 2));

					if (temp.size() < FSetSize)
					{
						auto tmp_tpl = std::make_tuple(tmp_pnt.pt, cell_idx, (*img_it)->cells[cell_idx]->image_index);
						temp.insert(std::make_pair(dist_v ,tmp_tpl));
					}
					else
					{
						const auto F_end = temp.rbegin();
						if (dist_v < F_end->first)
						{
							temp.erase(F_end->first);
							auto tmp_tpl = std::make_tuple(tmp_pnt.pt, cell_idx, (*img_it)->cells[cell_idx]->image_index);
							temp.insert(std::make_pair(dist_v, tmp_tpl));
						}
					}
				}
			}
		}

		std::vector<std::tuple<cv::Point2f, int, int>> F;

		std::for_each(temp.begin(), temp.end(), [&F](std::pair<double, std::tuple<cv::Point2f, int, int>> pair){F.push_back(pair.second);});
		
		return F;
	}

	static std::vector<double> calcPhotometricConsistency(cv::Mat patch
		, std::vector<std::shared_ptr<ImageModel>>::iterator begin
		, std::vector<std::shared_ptr<ImageModel>>::iterator  end)
	{
		std::vector<double> photoConsistVals;
		for (auto img_it = begin; img_it != end; ++img_it)
		{
			cv::Mat result;
			cv::matchTemplate((*img_it)->image, patch, result, cv::TM_CCORR_NORMED);
			double minval, maxval;
			cv::Point minloc, maxloc;
			cv::minMaxLoc(result, &minval, &maxval, &minloc, &maxloc);
			photoConsistVals.push_back(maxval);
		}

		return photoConsistVals;
	}
}
#endif

