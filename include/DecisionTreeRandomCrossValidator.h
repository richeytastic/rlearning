#pragma once
#ifndef RLEARNING_DECISION_TREE_RANDOM_CROSS_VALIDATOR_H
#define RLEARNING_DECISION_TREE_RANDOM_CROSS_VALIDATOR_H

#include <boost/shared_ptr.hpp>
#include "RandomCrossValidator.h"
using RLearning::RandomCrossValidator;


namespace RLearning
{

class DecisionTreeRandomCrossValidator : public RandomCrossValidator
{
public:
    DecisionTreeRandomCrossValidator( int tcount, int numIts,
                    const cv::Mat &xs, const cv::Mat &labels, int numEVs=0);
    virtual ~DecisionTreeRandomCrossValidator(){}

protected:
    virtual void train( const cv::Mat &trainData, const cv::Mat &labels);
    virtual float validate( const cv::Mat &x);

private:
    boost::shared_ptr<CvDTree> model_;
};  // end class


}   // end namespace

#endif
