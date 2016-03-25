#include "PrecisionRecallFinder.h"
using RLearning::PrecisionRecallFinder;
#include <cassert>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <iostream>


PrecisionRecallFinder::PrecisionRecallFinder( float maxConf, float tpCovPcnt)
    : _maxConf(maxConf), TP_COV_PCNT( tpCovPcnt)
{
}   // end ctor



void PrecisionRecallFinder::clear()
{
    _scores.clear();
}   // end clear



void PrecisionRecallFinder::add( float confidence, float covPcnt)
{
    assert( covPcnt >= 0 && covPcnt <= 100);
    Score s( confidence, covPcnt);
    _scores.push_back(s);
    if ( confidence > _maxConf)
        _maxConf = confidence;
}   // end add



// private
void PrecisionRecallFinder::calcRatios( float confThresh, int& tp, int& tn, int& fp, int& fn) const
{
    tp = tp = fp = fn = 0;
    const int numScores = _scores.size();
    for ( int j = 0; j < numScores; ++j)
    {
        if ( _scores[j].confidence >= confThresh)
        {
            if ( _scores[j].coveragePcnt > TP_COV_PCNT)
                tp++;
            else
                fp++;
        }   // end if
        else
        {
            if ( _scores[j].coveragePcnt > TP_COV_PCNT)
                fn++;
            else
                tn++;
        }   // end else
    }   // end for
}   // end calcRatios



void PrecisionRecallFinder::calcPrecisionRecallData( int steps, vector<float>& precision, vector<float>& recall) const
{
    std::sort( _scores.begin(), _scores.end()); // Sorts from lowest to highest
    const float threshStep = _maxConf / steps;

    for ( int i = 0; i <= steps; ++i)
    {
        const float confThresh = i * threshStep;
        int tp, tn, fp, fn; // tn not used here
        calcRatios( confThresh, tp, tn, fp, fn);

        const float p = Classification::calcMetric( tp, fn, tn, fp, Classification::Precision);
        const float r = Classification::calcMetric( tp, fn, tn, fp, Classification::Recall);
        precision.push_back(p);
        recall.push_back(r);
    }   // end for
}   // end calcPrecisionRecallData



float PrecisionRecallFinder::calcAveragePrecision( const vector<float>& precision, const vector<float>& recall)
{
    const int sz = recall.size();
    assert( sz == precision.size());

    const float RSTEP = 0.1;

    float psum = 0;
    for ( int i = 0; i <= 10; ++i)
    {
        float rmin = i*RSTEP;
        if ( i == 10)   // Ensure exact
            rmin = 1;

        // Find Pinterp(r) i.e. max p(r) : r >= rmin
        float maxp = 0;
        for ( int j = 0; j < sz; ++j)
        {
            const float p = precision[j];
            const float r = recall[j];
            assert( p >= 0 && p <= 1);
            assert( r >= 0 && r <= 1);

            if ( r >= rmin && p > maxp)
                maxp = p;
        }   // end for
        psum += maxp;
    }   // end for

    return psum/11;
}   // end calcAveragePrecision



float PrecisionRecallFinder::calcROCData( int steps, vector<float>& fprs, vector<float>& tprs) const
{
    std::sort( _scores.begin(), _scores.end());
    
    const float threshStep = _maxConf / steps;

    float auc = 0;
    float lastFPR = 0;
    float lastTPR = 0;

    for ( int i = 0; i <= steps; ++i)
    {
        const float confThresh = i * threshStep;
        int tp, tn, fp, fn;
        calcRatios( confThresh, tp, tn, fp, fn);

        const float fpr = Classification::calcMetric( tp, fn, tn, fp, Classification::Fallout);
        const float tpr = Classification::calcMetric( tp, fn, tn, fp, Classification::Recall);
        fprs.push_back(fpr);
        tprs.push_back(tpr);

        auc += fabs(fpr - lastFPR) * (tpr + lastTPR)/2;
        lastFPR = fpr;
        lastTPR = tpr;
    }   // end for

    return auc;
}   // end calcROCData
