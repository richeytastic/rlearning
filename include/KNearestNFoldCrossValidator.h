#pragma once
#ifndef RLEARNING_KNEAREST_NFOLD_CROSS_VALIDATOR_H
#define RLEARNING_KNEAREST_NFOLD_CROSS_VALIDATOR_H

#include <boost/shared_ptr.hpp>
#include "NFoldCrossValidator.h"
using RLearning::NFoldCrossValidator;


namespace RLearning
{

class KNearestNFoldCrossValidator : public NFoldCrossValidator
{
public:
    KNearestNFoldCrossValidator( int k, int nfold, const cv::Mat &xs, const cv::Mat &labels, int numEVs=0);
    virtual ~KNearestNFoldCrossValidator(){}

protected:
    virtual void train( const cv::Mat &trainData, const cv::Mat &labels);
    virtual float validate( const cv::Mat &x);

private:
    int _k;
    boost::shared_ptr<CvKNearest> _model;
};  // end class


}   // end namespace

#endif
