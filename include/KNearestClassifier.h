#pragma once
#ifndef RLEARNING_KNEAREST_CLASSIFIER_H
#define RLEARNING_KNEAREST_CLASSIFIER_H

#include "Classification.h"
using RLearning::Classifier;
#include <boost/shared_ptr.hpp>


namespace RLearning
{

class KNearestClassifier : public Classifier
{
public:
    KNearestClassifier( const cv::Mat &xs, const cv::Mat &labels, int k=1);
    virtual ~KNearestClassifier(){}

    virtual float predict( const cv::Mat &x) const;

private:
    int _k;
    boost::shared_ptr<CvKNearest> _model;
};  // end class


}   // end namespace

#endif
