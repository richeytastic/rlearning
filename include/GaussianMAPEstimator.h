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
