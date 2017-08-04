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

#include "MAPEstimator.h"
using RLearning::MAPEstimator;
#include <cassert>
#include <cmath>
#include <cfloat>


// local
void getMinMax( const vector<cv::Mat> &tset, cv::Mat &minVals, cv::Mat &maxVals, cv::Mat &setSums)
{
    assert( !tset.empty());
    const cv::Size sz = tset[0].size(); // All provided matrices must have same size!

    if ( minVals.empty())
        minVals = cv::Mat::ones( sz, CV_64FC1) * DBL_MAX;
    if ( maxVals.empty())
        maxVals = cv::Mat::ones( sz, CV_64FC1) * DBL_MIN;
    if ( setSums.empty())
        setSums = cv::Mat::zeros( sz, CV_64FC1);

    assert( minVals.type() == CV_64FC1);
    assert( minVals.size() == sz);
    assert( maxVals.type() == CV_64FC1);
    assert( maxVals.size() == sz);
    assert( setSums.type() == CV_64FC1);
    assert( setSums.size() == sz);

    BOOST_FOREACH( const cv::Mat &m, tset)
    {
        assert( m.size() == sz);
        assert( m.type() == CV_64FC1);

        for ( int i = 0; i < sz.height; ++i)
        {
            const double *mRow = m.ptr<double>(i);
            double *ssRow = setSums.ptr<double>(i);
            double *mnRow = minVals.ptr<double>(i);
            double *mxRow = maxVals.ptr<double>(i);

            for ( int j = 0; j < sz.width; ++j)
            {
                const double v = mRow[j];
                ssRow[j] += v;
                if ( v < mnRow[j])
                    mnRow[j] = v;
                if ( v > mxRow[j])
                    mxRow[j] = v;
            }   // end for - cols
        }   // end for - rows
    }   // end foreach
}   // end getMinMax



MAPEstimator::MAPEstimator( const vector<TrainingSet> &cs, int nbins) : nbins_(nbins)
{
    int totalCount = cs.size();   // Adding 1 per set initially for Laplacian smoothing

    cv::Mat setSums;
    BOOST_FOREACH( const TrainingSet &tset, cs)
    {
        getMinMax( tset, minVals_, maxVals_, setSums);
        totalCount += tset.size();  // Update total example count
    }   // end foreach

    // minVals and maxVals are now the per element min and max values over all training sets.
    // setSums is ignored for now

    // Get CV_64FC(nbins_) likelihood matrices
    BOOST_FOREACH( const TrainingSet &tset, cs)
    {
        cv::Mat cprops = createBinnedDistribution( tset); // P(X|C_i)
        props_.push_back( cprops);
        priors_.push_back( (double)(tset.size() + 1) / totalCount); // Prior probability for class
    }   // end foreach
}   // end ctor



double MAPEstimator::estimate( const cv::Mat &x, int &c) const
{
    const int numSets = props_.size();
    if ( numSets == 0)
    {
        c = -1;
        return 0;
    }   // end if

    static const double eps = 1e-30;

    double topPost = 0;
    double postSum = eps * numSets;

    for ( int i = 0; i < numSets; ++i)
    {
        double post_i = exp( log( priors_[i]) + calcLogLikelihood( x, props_[i]));
        postSum += post_i;
        if ( post_i > topPost)
        {
            topPost = post_i;
            c = i;
        }   // end if
    }   // end for

    return (topPost + eps) / postSum;
}   // end estimate



int MAPEstimator::estimateProbs( const cv::Mat &x, vector<double> &probs) const
{
    const int numSets = props_.size();
    if ( numSets == 0)
        return -1;

    probs.clear();

    double topPost = 0;
    int c = -1;

    static const double eps = 1e-30;

    double postSum = eps * numSets;

    for ( int i = 0; i < numSets; ++i)
    {
        double post_i = exp( log( priors_[i]) + calcLogLikelihood( x, props_[i]));
        postSum += post_i;
        if ( post_i > topPost)
        {
            topPost = post_i;
            c = i;
        }   // end if

        probs.push_back( post_i);
    }   // end for

    for ( int i = 0; i < numSets; ++i)
        probs[i] = (probs[i] + eps) / postSum;

    return c;
}   // end estimateProbs



// private
cv::Mat MAPEstimator::createBinnedDistribution( const vector<cv::Mat> &tset)
{
    const int nbins = nbins_;

    const cv::Size sz = minVals_.size();

    cv::Mat cnts( sz, CV_32SC(nbins));   // Laplacian smoothing - set all to 1
    cv::Mat tots( sz, CV_32SC1);    // Set all to nbins
    for ( int i = 0; i < sz.height; ++i)
    {
        int *cntsRow = cnts.ptr<int>(i);
        int *totsRow = tots.ptr<int>(i);

        for ( int j = 0; j < sz.width; ++j)
        {
            totsRow[j] = nbins;
            int *cntsVal = &cntsRow[j*nbins];
            for ( int k = 0; k < nbins; ++k)
                cntsVal[k] = 1;
        }   // end for - cols
    }   // end for - rows

    // Add the positive training instances
    BOOST_FOREACH( const cv::Mat &x, tset)
    {
        for ( int i = 0; i < sz.height; ++i)
        {
            const double *xRow = x.ptr<double>(i);

            const double *maxValsRow = maxVals_.ptr<double>(i);
            const double *minValsRow = minVals_.ptr<double>(i);

            int *cntsRow = cnts.ptr<int>(i);
            int *totsRow = tots.ptr<int>(i);

            for ( int j = 0; j < sz.width; ++j)
            {
                const double mx = maxValsRow[j];
                const double mn = minValsRow[j];
                const int bin = DiscreteNaiveBayes::binValue( nbins, mn, mx, xRow[j]);
                cntsRow[j*nbins + bin]++;
                totsRow[j]++;
            }   // end for - cols
        }   // end for - rows
    }   // end foreach

    cv::Mat props( sz, CV_64FC(nbins));

    // Normalise counts to create props
    for ( int i = 0; i < sz.height; ++i)
    {
        const int *cntsRow = cnts.ptr<int>(i);
        const int *totsRow = tots.ptr<int>(i);
        double *propsRow = props.ptr<double>(i);

        for ( int j = 0; j < sz.width; ++j)
        {
            const int tot = totsRow[j];
            const int *cntsVal = &cntsRow[j*nbins];
            double *propsVal = &propsRow[j*nbins];

            for ( int k = 0; k < nbins; ++k)
            {
                propsVal[k] = (double)cntsVal[k] / tot;
            }   // end for - bins
        }   // end for - cols
    }   // end for - rows

    return props;
}   // end createBinnedDistribution



// private
double MAPEstimator::calcLogLikelihood( const cv::Mat &x, const cv::Mat &props) const
{
    assert( x.type() == CV_64FC1);
    const cv::Size sz = x.size();
    assert( sz == props.size());

    const int nbins = nbins_;

    double posterior = 0;

    for ( int i = 0; i < sz.height; ++i)
    {
        const double *xRow = x.ptr<double>(i);
        const double *minValsRow = minVals_.ptr<double>(i);
        const double *maxValsRow = maxVals_.ptr<double>(i);
        const double *propsRow = props.ptr<double>(i);

        for ( int j = 0; j < sz.width; ++j)
        {
            const int bin = DiscreteNaiveBayes::binValue( nbins, minValsRow[j], maxValsRow[j], xRow[j]);
            posterior += log( propsRow[j*nbins + bin]);
        }   // end for - cols
    }   // end for - rows

    return posterior;
}   // end calcLogLikelihood
