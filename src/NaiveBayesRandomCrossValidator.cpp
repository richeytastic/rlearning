#include "NaiveBayesRandomCrossValidator.h"
using RLearning::NaiveBayesRandomCrossValidator;
#include <cassert>


NaiveBayesRandomCrossValidator::NaiveBayesRandomCrossValidator( int tcount, int numIts,
            const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : RandomCrossValidator( tcount, numIts, xs, labels, numEVs)
{
}   // end ctor


void NaiveBayesRandomCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    assert( trainData.type() == CV_32FC1);
    assert( labels.type() == CV_32SC1);
    assert( trainData.rows == labels.cols);
    CvNormalBayesClassifier *classifier = new CvNormalBayesClassifier( trainData, labels.t());
    model_ = boost::shared_ptr<CvNormalBayesClassifier>( classifier);
}   // end trainModel



float NaiveBayesRandomCrossValidator::validate( const cv::Mat &x)
{
    return model_->predict(x);
}   // end validate

