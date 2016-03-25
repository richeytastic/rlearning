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
