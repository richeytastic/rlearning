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

#include "DiscreteNaiveBayes.h"
using RLearning::DiscreteNaiveBayes;
#include <boost/foreach.hpp>
#include <cmath>
#include <climits>
#include <cassert>

#include <iostream>
using std::cerr;
using std::endl;

const int DiscreteNaiveBayes::MIN_NUM_BINS(2);
const int DiscreteNaiveBayes::MIN_SMOOTHING(1);



DiscreteNaiveBayes::DiscreteNaiveBayes( int nb, double minVal, double maxVal, int smooth)
    : nbins_(nb < MIN_NUM_BINS ? MIN_NUM_BINS : nb),
      lapSmooth_(smooth < MIN_SMOOTHING ? MIN_SMOOTHING : smooth), minVal_(minVal), maxVal_(maxVal),
      frowDef_(-1), fcolDef_(-1), createdFeaturePriors_(false), cTotals_(0)
{
    if ( nb < MIN_NUM_BINS)
        cerr << "WARNING: Chosen number of bins selected for DiscreteNaiveBayes is too small!"
             << "\nUsing minimum value (" << MIN_NUM_BINS << ") instead." << endl;
}   // end ctor



// static
int DiscreteNaiveBayes::binValue( int nbins, double min, double max, double val)
{
    if ( val < min || val > max)
        return -1;
    const double totalDiff = fabs(max - min);
    const double valDiff = fabs(val - min);
    const int idx = (int)(valDiff/totalDiff * (nbins - 1) + 0.5);
    assert( idx >= 0 && idx < nbins);
    return idx;
}   // end binValue



// private
void DiscreteNaiveBayes::addNewClass( int cid)
{
    cCounts_[cid] = 0;
    fcTotals_[cid] = cv::Mat::ones( frowDef_, fcolDef_, CV_32S) * (lapSmooth_ * nbins_);
    fcCounts_[cid] = cv::Mat( frowDef_, fcolDef_, CV_32SC(nbins_));
    for ( int i = 0; i < frowDef_; ++i)
    {
        int *rowPtr = fcCounts_[cid].ptr<int>(i);
        for ( int j = 0; j < fcolDef_; ++j)
        {
            int *colPtr = &rowPtr[j*nbins_];
            for ( int k = 0; k < nbins_; ++k)
                colPtr[k] = lapSmooth_;
        }   // end for
    }   // end for - rows
}   // end addNewClass



// private
void DiscreteNaiveBayes::setupFeaturePriors()
{
    // Build the matrix that will record the distributions for the prior
    // probability of P(F=x). Note that this is only needed for normalisation
    // (i.e. getting accurate probabilities) and can be ignored for classification
    // purposes (i.e. ordering the values of P(C=c|F=x) for different c)
    fTotals_ = cv::Mat::ones( frowDef_, fcolDef_, CV_32S) * (lapSmooth_ * nbins_);
    fCounts_ = cv::Mat( frowDef_, fcolDef_, CV_32SC(nbins_));
    for ( int i = 0; i < frowDef_; ++i)
    {
        int *rowPtr = fCounts_.ptr<int>(i);
        for ( int j = 0; j < fcolDef_; ++j)
        {
            int *colPtr = &rowPtr[j*nbins_];
            for ( int k = 0; k < nbins_; ++k)
                colPtr[k] = lapSmooth_;
        }   // end for
    }   // end for - rows
    createdFeaturePriors_ = false;
}   // end setupFeaturePriors



bool DiscreteNaiveBayes::addTrainingInstance( int cid, const cv::Mat &fv)
{
    assert( fv.isContinuous());
    const int nc = fv.channels() * fv.cols;

    if ( frowDef_ == -1)
    {
        frowDef_ = fv.rows;
        fcolDef_ = nc;
        setupFeaturePriors();
    }   // end if
    else
    {
        // All feature vectors must have the same dimensionality!
        if ( frowDef_ != fv.rows || fcolDef_ != nc)
        {
            cerr << "ERROR: Feature vector provided to DiscreteNaiveBayes has a"
                << " different number of dimensions to previous training instances!" << endl;
            return false;
        }   // end if
    }   // end else

    // Ensure that feature counts are initialised for the given class
    if ( cCounts_.count(cid) == 0)
        addNewClass(cid);

    cv::Mat &fcCntMat = fcCounts_[cid];
    cv::Mat &fcTotMat = fcTotals_[cid];
    for ( int i = 0; i < frowDef_; ++i)
    {
        const double *rowPtr = fv.ptr<double>(i);

        // Rows for incrementing feature counts needed for calculating P(F|C)
        int *fcCntRowPtr = fcCntMat.ptr<int>(i);
        int *fcTotRowPtr = fcTotMat.ptr<int>(i);

        // Rows for incrementing feature counts needed for calculating P(F)
        int *fCntRowPtr = fCounts_.ptr<int>(i);
        int *fTotRowPtr = fTotals_.ptr<int>(i);

        for ( int j = 0; j < fcolDef_; ++j) // j runs over all columns and channels of fv
        {
            const double val = rowPtr[j];
            const int idx = DiscreteNaiveBayes::binValue( nbins_, minVal_, maxVal_, val);

            // Increment feature counts required for P(F|C)
            fcCntRowPtr[j*nbins_ + idx]++;
            fcTotRowPtr[j]++;

            // Increment feature counts required for P(F)
            fCntRowPtr[j*nbins_ + idx]++;
            fTotRowPtr[j]++;
        }   // end for
    }   // end for

    // Increment number of training instances needed for calculating P(C)
    cCounts_[cid]++;
    cTotals_++;

    createdFeaturePriors_ = false;
    return true;
}   // end addTrainingInstance



void DiscreteNaiveBayes::calcNormalisationFeaturePriors()
{
    fPriors_ = cv::Mat( frowDef_, fcolDef_, CV_64FC(nbins_));
    for ( int i = 0; i < frowDef_; ++i)
    {
        const int *fCntRowPtr = fCounts_.ptr<int>(i);
        const int *fTotRowPtr = fTotals_.ptr<int>(i);
        double *fPriRowPtr = fPriors_.ptr<double>(i);

        for ( int j = 0; j < fcolDef_; ++j)
        {
            const int totCnt = fTotRowPtr[j];
            for ( int k = 0; k < nbins_; ++k)
            {
                assert( totCnt >= fCntRowPtr[j*nbins_ + k]);
                assert( totCnt > 0);
                assert( fCntRowPtr[j*nbins_ + k] > 0);
                fPriRowPtr[j*nbins_ + k] = log( (double)fCntRowPtr[j*nbins_ + k] / totCnt);
            }   // end for
        }   // end for
    }   // end for
    createdFeaturePriors_ = true;
}   // end calcNormalisationFeaturePriors



double DiscreteNaiveBayes::calcNaiveLogLikelihood( const cv::Mat &x, int cid)
{
    const int nc = x.cols * x.channels();
    assert( cCounts_.count(cid) == 1);
    assert( frowDef_ == x.rows);
    assert( fcolDef_ == nc);

    double lval = 0;

    const cv::Mat& fcTots = fcTotals_[cid];
    const cv::Mat& fcCnts = fcCounts_[cid];

    for ( int i = 0; i < x.rows; ++i)
    {
        const int *fcTotRowPtr = fcTots.ptr<int>(i);
        const int *fcCntRowPtr = fcCnts.ptr<int>(i);
        const double *xRowPtr = x.ptr<double>(i);

        for ( int j = 0; j < nc; ++j)
        {
            const int idx = DiscreteNaiveBayes::binValue( nbins_, minVal_, maxVal_, xRowPtr[j]);
            assert( fcTotRowPtr[j] >= fcCntRowPtr[j*nbins_ + idx]);
            assert( fcTotRowPtr[j] > 0);
            assert( fcCntRowPtr[j*nbins_ + idx] > 0);
            lval += log( (double)fcCntRowPtr[j*nbins_ + idx] / fcTotRowPtr[j]);    // Naive assumption
        }   // end for
    }   // end for

    return lval;    // Typically a value like -487.146
}   // end calcNaiveLogLikelihood



double DiscreteNaiveBayes::calcNaiveLogFeaturePrior( const cv::Mat &x)
{
    if ( !createdFeaturePriors_)
        return 0;

    const int nc = x.cols * x.channels();
    assert( frowDef_ == x.rows);
    assert( fcolDef_ == nc);

    double lval = 0;
    for ( int i = 0; i < frowDef_; ++i)
    {
        const double *xRowPtr = x.ptr<double>(i);
        const double *fRowPtr = fPriors_.ptr<double>(i);

        for ( int j = 0; j < nc; ++j)
        {
            const int idx = DiscreteNaiveBayes::binValue( nbins_, minVal_, maxVal_, xRowPtr[j]);
            lval += fRowPtr[j*nbins_ + idx];  // Naive assumption (logs already taken)
        }   // end for
    }   // end for

    return lval;    // Typically a value like -509.897
}   // end calcNaiveLogFeaturePrior



double DiscreteNaiveBayes::calcLogPosterior( int cid, const cv::Mat &x)
{
    assert( cCounts_.count(cid) == 1);
    assert( frowDef_ == x.rows);
    assert( fcolDef_ == x.cols * x.channels());
    // Sum of logs avoids underflow
    assert( cTotals_ >= cCounts_[cid]);

    const double classPrior = (double)cCounts_[cid]/cTotals_;
    const double logFeatGivenClass = calcNaiveLogLikelihood( x, cid);
    const double logFeatPrior = calcNaiveLogFeaturePrior(x);

    /*
    cerr << "P(C=c) = " << classPrior << endl;
    cerr << "P(F=x|C=c) = " << exp( logFeatGivenClass) << " <-- exp( " << logFeatGivenClass << ")" << endl;
    cerr << "P(F=x) = " << exp( logFeatPrior) << " <-- exp( " << logFeatPrior << ")" << endl;
    cerr << "P(C=x|F=x) = " << (classPrior * exp( logFeatGivenClass) / exp( logFeatPrior)) << endl;
    */

    const double logProb = log( classPrior) + logFeatGivenClass - logFeatPrior;
    //cerr << "P(C=x|F=x) = " << exp( logProb) << endl;
    return logProb;
}   // end calcLogPosterior



int DiscreteNaiveBayes::calcMAP( const cv::Mat &x)
{
    assert( cTotals_ > 0);

    double maxProb = DBL_MIN;
    int bestClass = -1;
    typedef std::pair<int,int> CPair;
    BOOST_FOREACH( const CPair &e, cCounts_)
    {
        const int cid = e.first;
        double logProb = calcLogPosterior( cid, x);

        if ( logProb > maxProb)
        {
            maxProb = logProb;
            bestClass = cid;
        }   // end if
    }   // end foreach

    return bestClass;
}   // end calcMAP



std::map<int, double> DiscreteNaiveBayes::calcClassLikelihoods( const cv::Mat &x)
{
    std::map<int, double> probs;

    double sumProbs = 0;
    typedef std::pair<int,int> CPair;
    BOOST_FOREACH( const CPair &e, cCounts_)
    {
        const int cid = e.first;
        const double logPost = calcLogPosterior( cid, x);
        assert( std::isfinite(logPost));
        probs[cid] = createdFeaturePriors_ ? exp(logPost) : logPost;
        sumProbs += probs[cid];
    }   // end for

    if ( sumProbs > 0.0)
    {
        // Normalise
        BOOST_FOREACH( const CPair &e, cCounts_)
        {
            const int cid = e.first;
            probs[cid] /= sumProbs;
        }   // end foreach
    }   // end if

    return probs;
}   // end calcClassLikelihoods

