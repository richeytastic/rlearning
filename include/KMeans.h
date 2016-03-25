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
