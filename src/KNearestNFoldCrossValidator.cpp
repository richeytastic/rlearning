#include "KNearestNFoldCrossValidator.h"
using RLearning::KNearestNFoldCrossValidator;


KNearestNFoldCrossValidator::KNearestNFoldCrossValidator( int k, int nf, const cv::Mat &xs, const cv::Mat &labels, int numEVs)
    : NFoldCrossValidator( nf, xs, labels, numEVs), _k(k)
{
    if ( _k < 1) _k = 1;
    if ( _k > 32) _k = 32;
}   // end ctor



void KNearestNFoldCrossValidator::train( const cv::Mat &trainData, const cv::Mat &labels)
{
    _model = boost::shared_ptr<CvKNearest>( new CvKNearest( trainData, labels.t()));
}   // end train



float KNearestNFoldCrossValidator::validate( const cv::Mat &x)
{
    return _model->find_nearest( x, _k);
}   // end validate

