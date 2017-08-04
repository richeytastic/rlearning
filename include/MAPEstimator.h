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
#ifndef RLEARNING_MAP_ESTIMATOR_H
#define RLEARNING_MAP_ESTIMATOR_H

#include "DiscreteNaiveBayes.h"
using RLearning::DiscreteNaiveBayes;
#include "ProbEstimator.h"
using RLearning::ProbEstimator;
#include <boost/foreach.hpp>


namespace RLearning
{

class MAPEstimator : public ProbEstimator
{
public:
    MAPEstimator( const vector<TrainingSet> &cs, int nbins);
    virtual ~MAPEstimator(){}

    virtual double estimate( const cv::Mat &x, int &c) const;

    virtual int estimateProbs( const cv::Mat &x, vector<double> &probs) const;

private:
    int nbins_;
    cv::Mat minVals_, maxVals_;
    vector<cv::Mat> props_;
    vector<double> priors_;

    // uses nbins_, minVals_, maxVals_
    cv::Mat createBinnedDistribution( const vector<cv::Mat> &tset);

    // uses nbins_, minVals_, maxVals_
    double calcLogLikelihood( const cv::Mat &x, const cv::Mat &props) const;
};  // end class


}   // end namespace

#endif
