#pragma once

#ifndef _DataFile_H_
#define _DataFile_H_

#include <iostream>
#include <fstream>
#include <regex>

#include "boost/filesystem.hpp"
#include "boost/iostreams/filtering_stream.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

class DataFile
{
private:
	std::vector<cv::Mat> _images;
	std::vector<cv::Mat> _projectMats;
	std::ifstream _fileStream;
	cv::Mat _K;
	cv::Mat _Rt;
	int _dataSize;
	int _index;

public:
	//constructor
	DataFile(std::string dirName, std::string fileName);
	// get next data set
	std::pair<cv::Mat, cv::Mat> getNext();

	std::pair<cv::Mat, cv::Mat> getNext(int index);
	
	int getDataSize() const;

	void resetIndex();

};

inline DataFile::DataFile(std::string dirName, std::string fileName) : _index(0)
{
	//const std::regex e("\\+");
	std::string l_dirName = dirName/*std::regex_replace(dirName, e, "/")*/;
	std::replace(l_dirName.begin(),l_dirName.end(),'\\','/');
	auto metaData = l_dirName + "/" + fileName;
	int index = 0;

	_fileStream = std::ifstream(metaData, std::ios_base::in | std::ios_base::binary);

	boost::iostreams::filtering_istream in;
	in.push(_fileStream);
	std::string dataDir;
	for (std::string str; std::getline(in, str); )
	{
		std::cout << "Processed line "<< index /*<< ", " << str*/ << '\n';
			
		std::istringstream buf(str);
		const std::istream_iterator<std::string> beg(buf), end;
		std::vector<std::string> strs(beg,end);

		if (index == 0)
		{
			_dataSize = std::stoi(strs[0]);
		}
		else if(index == 1)
		{
			dataDir = strs[0];
		}
		else
		{
			auto imgName = l_dirName + "/" + dataDir + "/" + strs[0];
			cv::Mat img = cv::imread(imgName);
			_images.push_back(img);
			
			// projection mat = K * [R t]
			_K = (cv::Mat_<double>(3, 3) <<	std::stod(strs[1])	, std::stod(strs[2]), std::stod(strs[3])
													, std::stod(strs[4]), std::stod(strs[5]), std::stod(strs[6])
													, std::stod(strs[7]), std::stod(strs[8]), std::stod(strs[9]));

			_Rt = (cv::Mat_<double>(3, 4) << std::stod(strs[10])		, std::stod(strs[11]), std::stod(strs[12]), std::stod(strs[19])
													, std::stod(strs[13])	, std::stod(strs[14]), std::stod(strs[15]), std::stod(strs[20])
													, std::stod(strs[16])	, std::stod(strs[17]), std::stod(strs[18]), std::stod(strs[21]));

			cv::Mat projMat = _K * _Rt;
			//std::cout << "projecteion matrix: \n" << projMat << std::endl << std::endl;
			_projectMats.push_back(std::move(projMat));
		}
		index++;
	}

}

inline std::pair<cv::Mat, cv::Mat> DataFile::getNext()
{
	
	if (!_images.empty() && !_projectMats.empty() && _index < _images.size() && _index < _projectMats.size())
	{
		std::pair<cv::Mat, cv::Mat> ret;
		ret.first = _images.at(_index).clone();
		ret.second = _projectMats.at(_index).clone();
		_index++;
		return ret;
	}
	return  std::pair<cv::Mat, cv::Mat>();
}

inline std::pair<cv::Mat, cv::Mat> DataFile::getNext(int index)
{
	if (!_images.empty() && !_projectMats.empty() && index < _images.size() && index < _projectMats.size())
	{
		std::pair<cv::Mat, cv::Mat> ret;
		ret.first = _images.at(index).clone();
		ret.second = _projectMats.at(index).clone();
		return ret;
	}
	return  std::pair<cv::Mat, cv::Mat>();
}

inline int DataFile::getDataSize() const
{
	return _dataSize - 1;
}

inline void DataFile::resetIndex()
{
	_index = 0;
}
#endif
