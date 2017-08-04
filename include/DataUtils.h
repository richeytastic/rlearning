/************************************************************************
 * Copyright (C) 2017 Richard Palmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#pragma once
#ifndef RLEARNING_DATA_UTILS_H
#define RLEARNING_DATA_UTILS_H

#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <cstdlib>
#include <cassert>
#include <vector>
using std::vector;


namespace RLearning
{

// Generate normally distributed N dimensional data. Dimensions are given by dimensions
// of mean and stddev matrices (whose dimensions must be equal) and single channelled.
// Input matrices mean and stddev may be multichannelled but will be flattened before
// use. They must both be of depth CV_32F.
//
// Output matrices are single row, single channel vectors. Row vectors are placed into
// output matrix drows.
// On return:
// count == drows.rows
// drows.type() == CV_32FC1
// drows.cols == mean.total() == stddev.total()
void generateRandomNormalData( boost::random::mt19937& mt,  // Mersenne Twister for pseudo-random number generation
                               const cv::Mat &mean, const cv::Mat &stddev, int count, cv::Mat &drows);


// Concatenate the collection of matrices in fvs as row vectors to xs starting at row i onwards
// giving each row instance the label lab in the corresponding position in row vector labels
// (where labels.cols == xs.rows)
void concatToMat( const vector<cv::Mat> &fvs, cv::Mat &xs, cv::Mat &labels, int &i, int lab);

}   // end namespace


#endif
