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
#ifndef RLEARNING_GAUSSIAN_MAP_ESTIMATOR_H
#define RLEARNING_GAUSSIAN_MAP_ESTIMATOR_H

#include "DiscreteNaiveBayes.h"
using RLearning::DiscreteNaiveBayes;
#include "ProbEstimator.h"
using RLearning::ProbEstimator;
#include "PCA.h"    // For calcMedian
#include <boost/foreach.hpp>



namespace RLearning
{

class GaussianMAPEstimator : public ProbEstimator
{
public:
    explicit GaussianMAPEstimator( const vector<TrainingSet> &cs);
    virtual ~GaussianMAPEstimator(){}

    virtual double estimate( const cv::Mat &x, int &c) const;

    virtual int estimateProbs( const cv::Mat &x, vector<double> &probs) const;

private:
    struct NormalParams
    {
        cv::Mat means;
        cv::Mat stdev;
    };  // end struct

    vector<NormalParams> params_; // Per class parameters for normal distributions
    vector<double> priors_; // Per class prior probabilities

    static void calcTrainingSetParams( const TrainingSet&, NormalParams&);
    static double calcLogLikelihood( const cv::Mat&, const NormalParams&);
};  // end class


}   // end namespace

#endif
