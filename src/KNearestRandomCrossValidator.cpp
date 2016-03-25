#include "KNearestRandomCrossValidator.h"
using RLearning::KNearestRandomCrossValidator;


KNearestRandomCrossValidator::KNearestRandomCrossValidator( int tcount, int numIts,
            const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : RandomCrossValidator( tcount, numIts, xs, labels, numEVs)
{
}   // end ctor


void KNearestRandomCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    CvKNearest *classifier = new CvKNearest( trainData, labels.t());
    model_ = boost::shared_ptr<CvKNearest>( classifier);
}   // end trainModel



float KNearestRandomCrossValidator::validate( const cv::Mat &x)
{
    return model_->find_nearest(x,1);
}   // end validate

