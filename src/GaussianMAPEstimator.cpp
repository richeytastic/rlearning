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

#include "GaussianMAPEstimator.h"
using RLearning::GaussianMAPEstimator;
//using RLearning::GaussianMAPEstimator::NormalParams;
#include <cassert>
#include <cmath>
#include <cfloat>



// static
// Calculate the normal distribution parameters for the training set.
void GaussianMAPEstimator::calcTrainingSetParams( const vector<cv::Mat> &tset, NormalParams &np)
{
    const int tsz = tset.size();

    assert( tsz > 1);   // Dataset must be more than a single point
    assert( tset[0].type() == CV_64FC1);

    const cv::Size xsz = tset[0].size();

    // Get the per element means
    np.means = cv::Mat::zeros( xsz, CV_64FC1);
    BOOST_FOREACH( const cv::Mat &m, tset)
    {
        assert( m.size() == xsz);
        np.means += m;
    }   // end foreach
    np.means /= tsz;

    // Get per element variances
    cv::Mat vars = cv::Mat::zeros( xsz, CV_64FC1);
    BOOST_FOREACH( const cv::Mat &m, tset)
    {
        const cv::Mat dlta = np.means - m;
        cv::Mat sqdlta;
        cv::pow( dlta, 2, sqdlta);
        vars += sqdlta;
    }   // end foreach

    //np.means = RLearning::calcMedian( tset);    // Set to median instead (more robust to outliers)

    vars /= (tsz - 1);  // Sample bias
    cv::sqrt( vars, np.stdev); // Standard deviation
    //np.stdev /= 2;
}   // end calcTrainingSetParams



GaussianMAPEstimator::GaussianMAPEstimator( const vector<TrainingSet> &cs)
{
    assert( cs.size() > 1);
    int totalCount = cs.size();   // Adding 1 per set initially for Laplacian smoothing

    BOOST_FOREACH( const TrainingSet &tset, cs)
        totalCount += tset.size();  // Update total example count

    // Get CV_64FC(nbins_) likelihood matrices
    BOOST_FOREACH( const TrainingSet &tset, cs)
    {
        NormalParams np;
        calcTrainingSetParams( tset, np);
        params_.push_back( np);
        // Set the class prior probability (with Laplace smoothing of 1)
        priors_.push_back( (double)(tset.size() + 1) / totalCount);
    }   // end foreach
}   // end ctor


// static
double GaussianMAPEstimator::calcLogLikelihood( const cv::Mat &x, const NormalParams &np)
{
    assert( (int)x.total() == x.cols);
    assert( np.means.cols == x.cols);

    const double *xrow = x.ptr<double>(0);
    const double *urow = np.means.ptr<double>(0);
    const double *srow = np.stdev.ptr<double>(0);

    double loglk = - x.cols * log(sqrt(2*M_PI));
    for ( int i = 0; i < x.cols; ++i)
        loglk += - pow( xrow[i] - urow[i], 2)/(2*srow[i]) - log( srow[i]);

    return loglk;
}   // end calcLogLikelihood



double GaussianMAPEstimator::estimate( const cv::Mat &x, int &c) const
{
    const int numSets = priors_.size();
    static const double eps = 1e-30;

    double topPost = 0;
    double postSum = eps * numSets;

    for ( int i = 0; i < numSets; ++i)
    {
        double post_i = exp( log( priors_[i]) + calcLogLikelihood( x, params_[i]));
        postSum += post_i;
        if ( post_i > topPost)
        {
            topPost = post_i;
            c = i;
        }   // end if
    }   // end for

    return (topPost + eps) / postSum;
}   // end estimate



int GaussianMAPEstimator::estimateProbs( const cv::Mat &x, vector<double> &probs) const
{
    const int numSets = priors_.size();
    static const double eps = 1e-30;

    double topPost = 0;
    int c = -1;

    probs.clear();

    double postSum = eps * numSets;

    for ( int i = 0; i < numSets; ++i)
    {
        double post_i = exp( log( priors_[i]) + calcLogLikelihood( x, params_[i]));
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
