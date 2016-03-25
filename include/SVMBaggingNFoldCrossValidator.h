#pragma once
#ifndef RLEARNING_SVM_BAGGING_NFOLD_CROSS_VALIDATOR_H
#define RLEARNING_SVM_BAGGING_NFOLD_CROSS_VALIDATOR_H

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "NFoldCrossValidator.h"
using RLearning::NFoldCrossValidator;

#include "SVMParams.h"
#include "SVMTrainer.h"
#include "SVMClassifier.h"
#include "KernelFunc.h"
using RLearning::SVMParams;
using RLearning::SVMTrainer;
using RLearning::SVMClassifier;
using RLearning::KernelFunc;


namespace RLearning
{

class SVMBaggingNFoldCrossValidator : public NFoldCrossValidator
{
public:
    SVMBaggingNFoldCrossValidator( const SVMParams &svmp, int numClassifiers, int nfold,
            const cv::Mat_<float>& xs, const cv::Mat_<int> &labels, int numEVs=0);

protected:
    virtual void train( const cv::Mat_<float> &trainData, const cv::Mat_<int>& labels);
    virtual float validate( const cv::Mat_<float> &x);

private:
    const KernelFunc<cv::Mat_<float> >::Ptr _kernel;
    double _cost;
    double _eps;
    int _nfolds;
    int _numClassifiers;

    vector<SVMClassifier::Ptr> _svmcs;

    void trainGroup( int, int, const vector<cv::Mat_<float> >*, const vector<cv::Mat_<float> >*);
};  // end class

}   // end namespace

#endif
