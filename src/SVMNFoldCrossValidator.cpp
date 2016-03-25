#include "SVMNFoldCrossValidator.h"
using RLearning::SVMNFoldCrossValidator;
#include <iostream>


SVMNFoldCrossValidator::SVMNFoldCrossValidator( const SVMParams &svmp, int nf,
        const cv::Mat_<float>& xs, const cv::Mat_<int>& labels, int numEVs)
    : NFoldCrossValidator( nf, xs, labels, numEVs),
    _kernel( svmp.makeKernel<cv::Mat_<float> >()), _cost(svmp.cost()), _eps(svmp.eps())
{
}   // end ctor



//void SVMNFoldCrossValidator::train( const vector<cv::Mat> &tpset, const vector<cv::Mat> &tnset)
void SVMNFoldCrossValidator::train( const cv::Mat_<float>& xs, const cv::Mat_<int>& labels)
{
    SVMTrainer<cv::Mat_<float> > svmt( _kernel, _cost, _eps, boost::thread::hardware_concurrency());
    svmt.enableErrorOutput( false);

    vector<cv::Mat_<float> > tpset, tnset;
    CrossValidator::splitIntoPositiveAndNegativeClasses( xs, labels, tpset, tnset);

    _svmc = svmt.train( tpset, tnset);  // Train
}   // end train


int SVMNFoldCrossValidator::getNumSVs() const
{
    return _svmc->getNumSVs();
}   // end if



float SVMNFoldCrossValidator::validate( const cv::Mat_<float> &x)
{
    return _svmc->predict(x);
}   // end validate

