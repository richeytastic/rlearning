/**
 * Principal Components Analysis and related utility functions.
 * Richard Palmer
 * Jan 2013
 */

#ifndef RLEARNING_PCA_H
#define RLEARNING_PCA_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <boost/foreach.hpp>
#include <iostream>


namespace RLearning
{

// Flatten the provided multi-dimensional matrices so that they reside as
// single channel column vectors inside the returned (continuous) matrix.
// Each of the provided matrices inside the given vector must have the
// same number of dimensions! Returned matrix is type CV_32FC1 and is
// suitable for use in PCA and related functions (below).
cv::Mat flattenToColumnVectors( const std::vector<cv::Mat> &data);

// Separate the rows of rowVecs into individual cv::Mat instances in parameter vs.
void separateRowVectors( const cv::Mat &rowVecs, std::vector<cv::Mat> &vs);

cv::Mat calcMedian( const std::vector<cv::Mat> &data);

// Calculate the means of the provided data (data dimensions must be the same).
// Returned matrix is CV_32FC(N) where N is the number of channels in each of
// the data matrices. The returned matrix has the same dimensions as the data.
cv::Mat calcMeans( const std::vector<cv::Mat> &data);

// Calculate the means from a bunch of multi-dimensional data points stored
// as column vectors in the given matrix. Number of data dimensions is found
// as the rows of parameter data. Number of data items is as the columns.
// Returned matrix is a single column vector with dimensions as row count.
cv::Mat calcMeans( const cv::Mat &colVecs);

// Calculate the covariance matrix of the provided data. Returned matrix is
// square (NxN dimensions) where N is the length of each reshaped data matrix
// into a single row/column vector.
// Parameter sampleBias (default true) scales the covariance matrix by 1/(M-1)
// with M data points if true and scales with 1/M otherwise.
// If parameter means is not provided, the means are calculated.
cv::Mat calcCovariance( const std::vector<cv::Mat> &data,
        bool sampleBias=true, cv::Mat means=cv::Mat());

cv::Mat calcCovariance( const cv::Mat &colVecs,
        bool sampleBias=true, cv::Mat means=cv::Mat());

// Print the given matrix.
void printMatrix( const cv::Mat &m, std::ostream &os);

// Print a bunch of vectors (stored as columns in m)
// Vectors are printed in rows to save space.
void printColumnVectors( const cv::Mat &m, std::ostream &os);

// Write a bunch of data points to an output stream - one row per data point.
// Allows for multiple dimensions (each dimension value separated by a space).
// Default point ordering is as column vectors in the given matrix.
void writePoints( std::ostream &os, const cv::Mat_<float> &data, bool inColOrder=true);


class PCA
{
public:
    PCA( const std::vector<cv::Mat> &data, bool useSampleBias=true);
    PCA( const cv::Mat &colData, bool useSampleBias=true);

    cv::Mat getMeans();

    // Calculate and return the covariance matrix (always single channel)
    cv::Mat getCovariance();

    // Calculate the unit length eigenvectors and place into out parameter rowVecs.
    // Returns the associated eigenvalues as a column vector in descending order.
    cv::Mat calcEigenvectors( cv::Mat &rowVecs);

    // Using the provided basis vectors (all should be unit length), project the
    // original data into the new basis space and place the resulting data vectors
    // as columns into out parameter outColumnData. If fewer basis vectors than those
    // calculated by calcEigenvectors are used, the original data will be projected
    // into a lower dimension (with some information being lost).
    // This function simply applies the matrix operation:
    // outColumnData = rowBasisVecs x originalDataAsColumnVectors
    void project( const cv::Mat &rowBasisVecs, cv::Mat &outColumnData) const;
    cv::Mat project( const cv::Mat &rowBasisVecs) const;

    // Back-project the provided column vector data into the original space
    // so that the original data can be reconstructed. Use the same basis vectors
    // as in the project function. This function applies the matrix operation:
    // outColData = (projectedColumnData.t() x rowBasisVecs).t()
    void reconstruct( const cv::Mat &projectedColumnData, const cv::Mat &rowBasisVecs, cv::Mat &outColData) const;
    cv::Mat reconstruct( const cv::Mat &projectedColumnData, const cv::Mat &rowBasisVecs) const;

private:
    bool useSampleBias_;    // If true, scale covariance matrix by 1/(n-1) for n data points
    int origChannels_;  // Original number of channels in each datum
    int origRows_;  // Original number of rows in each datum
    cv::Mat colVecs_;   // Original data as column vectors
    cv::Mat means_;
    cv::Mat covMat_;
    cv::Mat eigenRows_; // Eigenvectors as rows
    cv::Mat eigenVals_; // Associated eigenvalues as column vector

    void calcMeans();
};  // end class


}   // end namespace

#endif
