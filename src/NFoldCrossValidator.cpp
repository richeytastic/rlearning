#include "NFoldCrossValidator.h"
using RLearning::NFoldCrossValidator;
using RLearning::Classification;
using RLearning::StatsGenerator;
#include <cassert>
#include <iostream>


void NFoldCrossValidator::init( int nf)
{
    nfolds_ = nf < 1 ? 1 : nf;
    iter_ = 0;
    vector<int> ccnts;
    getClassCounts( ccnts);
    assert( ccnts.size() == 2); // We know this because CrossValidator checks this anyway

    if ( ccnts[0] < nfolds_ || ccnts[1] < nfolds_)
    {
        std::cerr << "ERROR: Cannot choose N folds > size of positive or negative dataset!" << std::endl;
        assert( ccnts[0] >= nfolds_ && ccnts[1] >= nfolds_);
    }   // end if

    // Reduce cross-validation data to nfold * equal sized sets
    nSegSz_ = ccnts[0] / nfolds_; // Integer division
    pSegSz_ = ccnts[1] / nfolds_; // Integer division
}   // end init



NFoldCrossValidator::NFoldCrossValidator( int nf, const cv::Mat_<float> &xs,
                                                  const cv::Mat_<int> &labs, int numEVs)
    : CrossValidator( xs, labs, numEVs)
{
    init(nf);
}   // end ctor



int NFoldCrossValidator::createTrainingMask( const cv::Mat_<int> &labs, const vector<int> &counts, char *mask)
{
    // Negative set runs from index 0
    const int nStart = iter_ * nSegSz_;
    const int nEnd = nStart + nSegSz_;
    for ( int i = nStart; i < nEnd; ++i)
        mask[i] = 0xff;

    // Positive set runs from counts[0]
    const int pStart = counts[0] + iter_ * pSegSz_;
    const int pEnd = pStart + pSegSz_;
    for ( int i = pStart; i < pEnd; ++i)
        mask[i] = 0xff;
    
    iter_++;

    return nEnd - nStart + pEnd - pStart;
}   // end createTrainingMask



void NFoldCrossValidator::printResults( ostream &os) const
{
    os << "[ Results after validation set " << iter_ << " of " << nfolds_ << " ]" << std::endl;
    const StatsGenerator* sgen = getStatsGenerator();
    double tp, fn, tn, fp;
    sgen->calcStats( tp, fn, tn, fp);
    Classification::printResultsTable( os, tp, fn, tn, fp);
}   // end printFinalResults


bool NFoldCrossValidator::moreIterations() const
{
    return iter_ < nfolds_;
}   // end moreIterations
