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


