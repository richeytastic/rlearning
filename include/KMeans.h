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

/**
 * Does k-means clustering on multi-dimensional vectors.
 */

#pragma once
#ifndef RLearning_KMEANS_H
#define RLearning_KMEANS_H

#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <vector>
using std::vector;
#include "PCA.h"

typedef vector<cv::Mat> Cluster;
typedef vector<Cluster> Clusters;


namespace RLearning
{

class KMeans
{
public:
    KMeans(){}   // Number of clusters given by calls to add( const cv::Mat, int)

    // Initialise the clusterer with initial seed values (at least 2).
    explicit KMeans( const vector<cv::Mat> &seeds);

    // Add a vector to the cluster mean that it's closest to
    // and return the index of this cluster.
    int add( const cv::Mat &v);

    // Add a vector to the indexed cluster.
    void add( const cv::Mat &v, int idx);

    inline int getClusterCount() const { return means.size();}
    inline vector<cv::Mat> getClusterMeans() const { return means;}
    inline Clusters getClusters() const { return clusters;}

    vector<cv::Mat> calcClusterMedians() const; // May be more robust to outliers


    // Return total sum of square differences of all vectors from all clusters.
    double getSSDs() const;

    // Return the sum of square differences from all vectors from a single cluster.
    // If cluster ID is invalid, a negative value is returned.
    double getSSD( int idx) const;

    // Return the distance x - u for cluster i
    cv::Mat getDistance( const cv::Mat &x, int i) const;

private:
    vector<cv::Mat> means;    // Cluster centres
    Clusters clusters;        // Each cluster's vectors

    void createCluster( const cv::Mat &);
};  // end class

}   // end namespace

#endif
