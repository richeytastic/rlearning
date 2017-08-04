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
#ifndef RLEARNING_PROB_ESTIMATOR_H
#define RLEARNING_PROB_ESTIMATOR_H

#include <vector>
using std::vector;
#include <opencv2/opencv.hpp>

typedef vector<cv::Mat> TrainingSet;

namespace RLearning
{

class ProbEstimator
{
public:
    // Estimate which class c x comes from given the Maximum A Posteriori between the classes.
    // Returns the probability of being in the given class.
    virtual double estimate( const cv::Mat &x, int &c) const = 0;

    // Estimate all class probabilities of test instance x belonging
    // to the given classes. Returns the index of the class with the
    // highest probability.
    virtual int estimateProbs( const cv::Mat &x, vector<double> &probs) const = 0;
};  // end class

}   // end namespace

#endif


