#include "KNearestClassifier.h"
using RLearning::KNearestClassifier;


KNearestClassifier::KNearestClassifier( const cv::Mat &xs, const cv::Mat &labels, int k) : _k(k)
{
    if ( _k < 1)
        _k = 1;
    if ( _k > 32)
        _k = 32;

    _model = boost::shared_ptr<CvKNearest>( new CvKNearest( xs, labels));
}   // end ctor



float KNearestClassifier::predict( const cv::Mat &x) const
{
    return _model->find_nearest( x, _k);
}   // end predict

