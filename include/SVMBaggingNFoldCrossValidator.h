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
