#include "FeatureDetector.h"
using RLearning::FeatureDetector;
#include <cassert>


FeatureDetector::FeatureDetector( Model *m, const FeatureOperator *fo)
    : model_(m), fop_(fo)
{
    assert( m != NULL);
    assert( fo != NULL);
}   // end ctor


cv::Size2f FeatureDetector::getActualSize() const
{
    return model_->getActualSize();
}   // end getActualSize


double FeatureDetector::detect( const cv::Rect &rct)
{
    const cv::Mat fv = (*fop_)( rct);   // Extract the feature vector
    cv::Mat fv32;
    fv.convertTo(fv32, CV_32F);
    const cv::Mat z = fv32.reshape(1,1);    // Single row CV_32C1
    float v = model_->predict( z);
    return v > 0 ? 1 : 0;
}   // end detect
