#pragma once
#ifndef RLEARNING_KNEAREST_RANDOM_CROSS_VALIDATOR_H
#define RLEARNING_KNEAREST_RANDOM_CROSS_VALIDATOR_H

#include <boost/shared_ptr.hpp>
#include "RandomCrossValidator.h"
using RLearning::RandomCrossValidator;


namespace RLearning
{

class KNearestRandomCrossValidator : public RandomCrossValidator
{
public:
    KNearestRandomCrossValidator( int tcount, int numIts,
                    const cv::Mat &xs, const cv::Mat &labels, int numEVs=0);
    virtual ~KNearestRandomCrossValidator(){}

protected:
    virtual void train( const cv::Mat &trainData, const cv::Mat &labels);
    virtual float validate( const cv::Mat &x);

private:
    boost::shared_ptr<CvKNearest> model_;
};  // end class


}   // end namespace

#endif
