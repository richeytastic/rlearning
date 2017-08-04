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

#include "KMeans.h"
using RLearning::KMeans;
#include "Classification.h"
using RLearning::Classification;
#include <cassert>
#include <cmath>
#include <climits>



cv::Mat KMeans::getDistance( const cv::Mat &x, int i) const
{
    assert( i < (int)means.size());
    assert( x.size() == means[i].size());
    assert( x.type() == means[i].type());
    return x - means[i];
}   // end getDistance



// Given a vector of cluster centres and a new instance (nm), find cluster centre
// that nm is closest to (Euclidean) and return index of cluster centre.
int calcClosestCluster( const vector<cv::Mat> &means, const cv::Mat &nm)
{
    int bestIdx = -1;
    double minssd = DBL_MAX;

    const int meansSz = means.size();
    for ( int i = 0; i < meansSz; ++i)
    {
        double ssd = Classification::calcSSD( nm, means[i]);
        if ( ssd < minssd)
        {
            minssd = ssd;
            bestIdx = i;
        }   // end if
    }   // end for

    return bestIdx;
}   // end calcClosestCluster



void KMeans::createCluster( const cv::Mat &seed)
{
    const int i = clusters.size();
    clusters.push_back( Cluster());
    cv::Mat dseed;
    seed.convertTo( dseed, CV_64F);
    clusters[i].push_back( seed);
    means.push_back( dseed);
}   // end createCluster



int KMeans::add( const cv::Mat &fv)
{
    if ( means.empty())
    {
        add( fv, 0);
        return 0;
    }   // end if

    // Dimensions of clusters and img must be the same!
    assert( fv.size() == means[0].size());
    const int cIdx = calcClosestCluster( means, fv);
    add( fv, cIdx);
    return cIdx;
}   // end add



void KMeans::add( const cv::Mat &fv, int cidx)
{
    if ( cidx == (int)means.size())  // Add a new cluster and return
    {
        createCluster( fv);
        return;
    }   // end if

    assert( cidx < (int)means.size());
    assert( fv.size() == means[cidx].size());

    cv::Mat dfv;
    fv.convertTo( dfv, CV_64F);

    // Calculate new cluster centre
    const int fcount = clusters[cidx].size();
    const cv::Mat cSum = means[cidx] * fcount + dfv;
    clusters[cidx].push_back(dfv);
    means[cidx] = cSum / (fcount+1);
}   // end add


vector<cv::Mat> KMeans::calcClusterMedians() const
{
    vector<cv::Mat> medians;
    BOOST_FOREACH( const vector<cv::Mat> &c, clusters)
        medians.push_back( RLearning::calcMedian( c));
    return medians;
}   // end calcClusterMedians



KMeans::KMeans( const vector<cv::Mat> &seeds)
{
    const int sz = seeds.size();
    assert( sz >= 2);   // Must have at least two seeds!

    for ( int i = 0; i < sz; ++i)
    {
        // Seeds must all be of the same dimension
        if ( i > 0)
        {
            assert( seeds[i].size() == seeds[i-1].size());
            assert( seeds[i].type() == seeds[i-1].type());
        }   // end if

        createCluster( seeds[i]);
    }   // end for
}   // end ctor



double KMeans::getSSDs() const
{
    double tot = 0.0;
    const int sz = means.size();
    for ( int i = 0; i < sz; ++i)
        tot += getSSD( i);
    return tot;
}   // end getSSDs



double KMeans::getSSD( int idx) const
{
    double tot = -1;
    if ( idx >= 0 && idx < (int)means.size())
    {
        tot = 0.0;
        BOOST_FOREACH( const cv::Mat &fv, clusters[idx])
            tot += Classification::calcSSD( fv, means[idx]);
    }   // end if
    return tot;
}   // end getSSD

